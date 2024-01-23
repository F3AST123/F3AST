#!/usr/bin/env python3
""" Training for F3AST """
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
from sklearn.metrics import average_precision_score
from tabulate import tabulate
import collections
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
import torchvision.models.video as models
from itertools import groupby
from collections import OrderedDict
import timm
from tqdm import tqdm
import cv2
import copy

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.seq2seq import EncoderTransformer, DecoderTransformer, beam_decode
from model.slowfast import ResNet3dSlowFast
from dataset.frame import ActionSeqDataset, ActionSeqVideoDataset
from util.eval import process_frame_predictions, edit_score
from util.io import load_json, store_json, store_gz_json, clear_files
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs, compute_average_precision, acc_iou, mean_category_acc, success_rate
import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM_FRAMES = 2000000
BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 4
HIDDEN_DIM = 768
MASK_RATIO = 0.15

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # From torchvision
            'rn50_tsm',
            'rny002_gsm',
            'slowfast'
        ], help='architecture for feature extraction')
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='f3ast',
        choices=['gru', 'vanilla', 'f3ast'])

    parser.add_argument('--use_local', type=bool, default=False)
    parser.add_argument('--local_frame_dir', type=str, default=None,
                        help='Path to frames with objects of interest')
    parser.add_argument('--max_seq_len', type=int, default=30)
    parser.add_argument('--clip_len', type=int, default=128)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--mask_ratio', type=int, default=0.15)
    parser.add_argument('--sparse_att_mask', type=bool, default=True)
    parser.add_argument('--with_tolerance', type=int, default=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int, default=30)
    parser.add_argument('--criterion', choices=['acc', 'loss'], default='acc')

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    return parser.parse_args()


class F3AST(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, use_local=False, sparse_att_mask=True,
                     step=1, max_seq_len=30, device='cuda'):
            super().__init__()
            is_rgb = True
            self._device = device
            self._num_classes = num_classes

            if 'rn50' in feature_arch:
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                glb_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                glb_feat_dim = features.fc.in_features
                glb_feat.fc = nn.Identity()

                # local feature embed
                if use_local:
                    lcl_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                    lcl_feat_dim = lcl_feat.head.fc.in_features
                    lcl_feat.head.fc = nn.Identity()

            elif 'rny002' in feature_arch:
                rny_name = {'rny002': 'regnety_002'}[feature_arch.rsplit('_', 1)[0]]
                glb_feat = timm.create_model(rny_name, pretrained=True)
                glb_feat_dim = glb_feat.head.fc.in_features
                glb_feat.head.fc = nn.Identity()

                # local feature embed
                if use_local:
                    lcl_feat = timm.create_model(rny_name, pretrained=True)
                    lcl_feat_dim = lcl_feat.head.fc.in_features
                    lcl_feat.head.fc = nn.Identity()

            elif 'slowfast' in feature_arch:
                glb_feat = ResNet3dSlowFast(None, resample_rate=4, speed_ratio=4, fusion_kernel=7, slow_upsample=4)
                glb_feat.load_pretrained_weight()
                glb_feat_dim = 2304

                # local feature embed
                if use_local:
                    lcl_feat = ResNet3dSlowFast(None, resample_rate=4, speed_ratio=4, fusion_kernel=7, slow_upsample=4)
                    lcl_feat.load_pretrained_weight()
                    lcl_feat_dim = 2304

            else:
                raise NotImplementedError(feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = clip_len
            if feature_arch.endswith('_tsm'):
                make_temporal_shift(glb_feat, clip_len, is_gsm=False, step=step)
                if use_local:
                    make_temporal_shift(lcl_feat, clip_len, is_gsm=False, step=step)
                self._require_clip_len = clip_len
            elif feature_arch.endswith('_gsm'):
                make_temporal_shift(glb_feat, clip_len, is_gsm=True)
                if use_local:
                    make_temporal_shift(lcl_feat, clip_len, is_gsm=False, step=step)
                self._require_clip_len = clip_len

            self._glb_feat = glb_feat  # global feature extractor
            self._feat_dim = glb_feat_dim
            self._use_local = use_local
            if use_local:  # local feature extractor
                self._lcl_feat = lcl_feat
                self._feat_dim += lcl_feat_dim
            self._is_slowfast = 'slowfast' in feature_arch
            self._sparse_att_mask = sparse_att_mask
            self._max_seq_len = max_seq_len

            # encoder-decoder
            d_model = min(HIDDEN_DIM, self._feat_dim)
            self._encoder = EncoderTransformer(self._feat_dim, d_model, dim_feedforward=1024, num_layers=1, dropout=0.1)
            self._decoder = DecoderTransformer(d_model, num_classes, dim_feedforward=1024, num_layers=2, dropout=0.1)

            # learn soft attention for encoder
            if sparse_att_mask:
                self.learn_vid_att = torch.nn.Embedding(clip_len * clip_len, 1)
                self.sigmoid = torch.nn.Sigmoid()

        def forward(self, frame, lcl_frame, tgt=None, src_key_padding_mask=None):
            batch_size, true_clip_len, channels, height, width = frame.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert true_clip_len <= self._require_clip_len, \
                    'Expected {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    frame = F.pad(
                        frame, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            src = None
            # global visual embedding
            if self._is_slowfast:
                im_feat = self._glb_feat(frame.transpose(1, 2)).transpose(1, 2)
            else:
                im_feat = self._glb_feat(
                    frame.view(-1, channels, height, width)
                ).reshape(batch_size, clip_len, -1)
            src = im_feat

            if self._use_local:
                # local feature embedding
                lcl_feat = self._lcl_feat(lcl_frame.transpose(1, 2)).transpose(1, 2)
                src = torch.cat((src, lcl_feat), dim=2) if src is not None else lcl_feat
            
            # learn encoder attention mask
            if self._sparse_att_mask:
                learn_att = self.learn_vid_att.weight.reshape(clip_len, clip_len)
                learn_att = self.sigmoid(learn_att)
                diag_mask = torch.diag(torch.ones(clip_len)).to(self._device)
                video_attention = (1. - diag_mask) * learn_att
                learn_att = diag_mask + video_attention
            else:
                video_attention = None
                learn_att = None

            # transformer encoder
            enc_feat = self._encoder(src, mask=learn_att, src_key_padding_mask=src_key_padding_mask)
            if tgt is None:
                tgt, tgt_prob = beam_decode(enc_feat, self._decoder, self._num_classes, mask=learn_att,
                                            src_key_padding_mask=src_key_padding_mask,
                                            max_length=self._max_seq_len)
                return tgt, tgt_prob

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1], device=self._device).bool()
            tgt_key_padding_mask = torch.zeros(tgt.size(), dtype=torch.bool, device=self._device)
            tgt_key_padding_mask[tgt == 0] = True
            seq_pred, dec_feat = self._decoder(tgt, enc_feat, tgt_mask=tgt_mask,
                                               memory_key_padding_mask=src_key_padding_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask)

            return seq_pred, video_attention

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, use_local=False,
                 sparse_att_mask=True, max_seq_len=30, device='cuda', multi_gpu=False):
        self._device = device
        self._multi_gpu = multi_gpu
        self._model = F3AST.Impl(num_classes, feature_arch, temporal_arch, clip_len, use_local=use_local,
                                 sparse_att_mask=sparse_att_mask, step=step, max_seq_len=max_seq_len)

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, mask_ratio=0.0):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame, lcl_frame, src_key_padding_mask = loader.dataset.load_frame_gpu(batch, self._device)

                # training and apply random masking
                if optimizer is not None and mask_ratio > 0:
                    for i in range(src_key_padding_mask.size(0)):
                        unmasked_indices = (src_key_padding_mask[i] == False).nonzero(as_tuple=True)[0]
                        num_unmasked = len(unmasked_indices)
                        num_to_convert = int(mask_ratio * num_unmasked)
                        # Randomly select indices of the 'False' elements to change to 'True'
                        indices_to_convert = unmasked_indices[torch.randperm(num_unmasked)[:num_to_convert]]
                        src_key_padding_mask[i, indices_to_convert] = True

                label = batch['label'].to(self._device)

                with torch.cuda.amp.autocast():
                    seq_pred, vid_atten = self._model(frame, lcl_frame, label[:, :-1], src_key_padding_mask)
                    seq_len = [len(b[b > 0]) for b in label[:, 1:]]
                    for b_id, b in enumerate(seq_len):
                        seq_pred[b_id, b:] = 0
                    seq_pred_pack = pack_padded_sequence(seq_pred, seq_len, batch_first=True, enforce_sorted=False).data
                    label_pack = pack_padded_sequence(label[:, 1:], seq_len, batch_first=True, enforce_sorted=False).data

                    # seq cross entropy
                    loss = F.cross_entropy(seq_pred_pack, label_pack)
                    # encoder attention mask sparsity loss
                    if vid_atten is not None:
                        loss += torch.mean(torch.abs(vid_atten))
                        
                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, frame, lcl_frame, src_key_padding_mask, use_amp=True):
        if not isinstance(frame, torch.Tensor):
            frame = torch.FloatTensor(frame)
            lcl_frame = torch.FloatTensor(lcl_frame)
        if len(frame.shape) == 4: # (L, C, H, W)
            frame = frame.unsqueeze(0)
        if len(lcl_frame.shape) == 4: # (L, C, H, W)
            lcl_frame = lcl_frame.unsqueeze(0)
        frame = frame.to(self._device)
        lcl_frame = lcl_frame.to(self._device)
        
        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                seq_tgt, seq_tgt_prob = self._model(frame, lcl_frame, src_key_padding_mask=src_key_padding_mask)
            return seq_tgt.cpu().numpy(), seq_tgt_prob.cpu().numpy()


def evaluate(model, dataset, split, classes, with_tolerance=False, device='cuda'):
    sets = []

    if with_tolerance:
        # FineTennis tolerance
        sets = [{4, 5}, {5, 6}]
        for start in range(8, 31, 6):
            sets.append({start, start + 2})
            sets.append({start, start + 3})
            sets.append({start + 1, start + 2})
            sets.append({start + 1, start + 4})
            sets.append({start + 2, start + 3})
            sets.append({start + 2, start + 4})
        sets += [{32, 33}, {33, 34}]
        for start in range(36, 59, 6):
            sets.append({start, start + 2})
            sets.append({start, start + 3})
            sets.append({start + 1, start + 2})
            sets.append({start + 1, start + 4})
            sets.append({start + 2, start + 3})
            sets.append({start + 2, start + 4})

    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (np.zeros(2 * video_len, int),
                            np.zeros((2 * video_len, len(classes) + 1), np.float32))

    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):
        src_key_padding_mask = clip['src_key_padding_mask'].to(device)
        if batch_size > 1:
            # Batched by dataloader
            batch_pred_seq, batch_pred_seq_prob = model.predict(clip['frame'], clip['local_frame'], src_key_padding_mask)
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                classes, scores = pred_dict[video]
                pred_seq = batch_pred_seq[i]
                pred_seq_prob = batch_pred_seq_prob[i]
                pred = pred_seq[(pred_seq != 1) & (pred_seq != 2) & (pred_seq != 0)]
                pred_prob = pred_seq_prob[(pred_seq != 1) & (pred_seq != 2) & (pred_seq != 0)]
                start = max(0, clip['start'][i].item())
                end = start + len(pred)
                if start < end:
                    classes[start:end] += pred
                    scores[start:end] += pred_prob

    # accuracy, edit score, success rate, mean Average Precision
    edit_scores = []
    y_true, y_pred = None, None
    acc, num_samples, num_actions = 0, 0, 0
    for video, (pred, pred_prob) in sorted(pred_dict.items()):
        gt = dataset.get_labels(video)
        pred_prob = pred_prob[pred != 0]
        pred = pred[pred!=0]
        gt = gt[gt!=0]

        # sucess rate
        if len(pred) == len(gt):
            success = True
            for j in range(len(pred)):
                if pred[j] != gt[j] and {pred[j], gt[j]} not in sets:
                    success = False
                    break
            if success:
                sr += 1

        # edit score
        edit_scores.append(edit_score(pred, gt, sets))

        max_length = max(len(pred), len(gt))
        min_length = min(len(pred), len(gt))
        pred = np.pad(pred, (0, max_length - len(pred)), 'constant')
        gt = np.pad(gt, (0, max_length - len(gt)), 'constant')

        if y_true is None and y_pred is None:
            # remove background, start, end
            y_true = gt[:min_length] - 3
            y_pred = pred_prob[:min_length, 3:]
        else:
            # remove background, start, end
            y_true = np.concatenate((y_true, gt[:min_length] - 3), axis=0)
            y_pred = np.concatenate((y_pred, pred_prob[:min_length, 3:]), axis=0)

        num_samples += 1

        # accuracy
        for j in range(max_length):
            if pred[j] != 0 and gt[j] != 0:
                num_actions += 1
                if pred[j] == gt[j]:
                    acc += 1
                # if allow tolerance
                elif {pred[j], gt[j]} in sets:
                    acc += 1

    # if some classes do not exist
    mask = [0] * y_pred.shape[1]
    for i in range(y_pred.shape[1]):
        if i not in y_true:
            mask[i] = 1
            temp_pred = np.zeros((1, y_pred.shape[1]))
            temp_pred[:, i] = 1
            y_true = np.concatenate((y_true, np.array([i])), axis=0)
            y_pred = np.concatenate((y_pred, temp_pred), axis=0)

    # average precision
    APs = average_precision_score(y_true, y_pred, average=None)

    print('Acc:', acc / num_actions)
    print('Edit:', sum(edit_scores) / len(edit_scores))
    print('SR:', sr / num_samples)
    print('mAP:', np.ma.masked_array(APs, mask=mask).mean())
    return acc / num_actions


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'acc':
        key = 'val_acc'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'classes.txt'))

    dataset_len = EPOCH_NUM_FRAMES // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }

    print('Dataset size:', dataset_len)
    train_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.clip_len, dataset_len, is_eval=False,
        local_frame_dir=args.local_frame_dir, max_seq_len=args.max_seq_len,
        **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, dataset_len // 4,
        local_frame_dir=args.local_frame_dir, max_seq_len=args.max_seq_len,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'acc':
        # Only perform acc evaluation during training if criterion is acc
        val_data_frames = ActionSeqVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.clip_len, local_frame_dir=args.local_frame_dir,
            max_seq_len=args.max_seq_len, crop_dim=args.crop_dim, stride=args.stride, overlap_len=0)

    return classes, train_data, val_data, None, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'use_local': args.use_local,
        'with_tolerance': args.with_tolerance,
        'stride': args.stride,
        'mask_ratio': args.mask_ratio,
        'num_epochs': num_epochs,
        'max_seq_len': args.max_seq_len,
        'sparse_att_mask': args.sparse_att_mask,
        'warm_up_epochs': args.warm_up_epochs,
        'local_frame_dir': args.local_frame_dir,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, train_data_frames, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = F3AST(len(classes) + 1, args.feature_arch, args.temporal_arch, clip_len=args.clip_len, step=args.stride,
                  multi_gpu=args.gpu_parallel, use_local=args.use_local, sparse_att_mask=args.sparse_att_mask,
                  max_seq_len=args.max_seq_len)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'acc' else float('inf')
    best_loss, stop_criterion = float('inf'), 0

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(
            args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(
            train_loader, optimizer, scaler, lr_scheduler=lr_scheduler,
            acc_grad_iter=args.acc_grad_iter, mask_ratio=args.mask_ratio)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_acc = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'acc':
            if epoch >= args.start_val_epoch:
                val_acc = evaluate(model, val_data_frames, 'VAL', classes, with_tolerance=args.with_tolerance)
                if args.criterion == 'acc' and val_acc > best_criterion:
                    best_criterion = val_acc
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'val_acc': val_acc})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on hold out splits
        eval_splits += ['test']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSeqVideoDataset(classes, split_path, args.frame_dir, args.clip_len, overlap_len=0, 
                                                   local_frame_dir=args.local_frame_dir, max_seq_len=args.max_seq_len,
                                                   crop_dim=args.crop_dim, stride=args.stride)
                split_data.print_info()
                evaluate(model, split_data, split.upper(), classes, with_tolerance=args.with_tolerance)


if __name__ == '__main__':
    main(get_args())
