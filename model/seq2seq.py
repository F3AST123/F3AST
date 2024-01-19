import math
import random
import copy
import time
import operator
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from .blocks import (TransformerBlock, MaskedConv1D, ConvBlock, LayerNorm, LocalMHA, FeedForward)
from queue import PriorityQueue


class Beam:
    def __init__(self, prevNode, wordId, wordp, logProb, length):

        self.prevNode = prevNode
        self.wordId = wordId
        self.wordp = wordp
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        T = 0.1
        # prevent overflow
        return self.logp / float((self.length - 1)**T + 1e-6) + alpha * reward


def beam_decode(encoder_outputs,
                decoder,
                num_classes,
                beam_width=5,
                n_top=1,
                max_length=20,
                mask=None,
                memory_mask=None,
                src_key_padding_mask=None):
    bs, _, _ = encoder_outputs.size()
    device = encoder_outputs.device

    decoded_batch = []
    decoded_prob_batch = []
    sos_token = 1
    eos_token = 2

    # mask = (~src_key_padding_mask).long().unsqueeze(dim=1)
    # encoder_outputs, _ = encoder(src, mask)
    # encoder_outputs = encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
    # encoder_outputs, _ = encoder(src)
    # encoder_outputs = src

    # decoding goes sentence by sentence
    for idx in range(bs):
        encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)  # (1, time_step, out_channels)
        decoder_input = torch.LongTensor([[sos_token]]).to(device)
        
        decoder_input_prob = torch.zeros(1, 1, num_classes).to(device)
        decoder_input_prob[0, 0, sos_token] = 1

        # number of sentences to generate
        endnodes = []
        number_required = min(n_top + 1, n_top - len(endnodes))

        # starting node - prevNode, wordId, logProb, length
        node = Beam(None, decoder_input, decoder_input_prob, 0, 1)
        beam_nodes = PriorityQueue()
        # add length to avoid tie-breaking
        beam_nodes.put((-node.eval(), time.time(), node.length, node))
        qsize = 1

        # start beam search
        while True:
            if qsize > 2000:
                break
            if len(decoder_input[0]) > max_length:
                break

            scores, t, l, n = beam_nodes.get()
            decoder_input = n.wordId
            decoder_input_prob = n.wordp

            if n.wordId[0, -1].item() == eos_token and n.prevNode:
                endnodes.append((scores, None, None, n))

                # if we reached maximum sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # feeding to decoder
            decoder_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input[0]), device=device).bool()
            if memory_mask is not None:
                temp_memory_mask = memory_mask[:len(decoder_input[0]), :]
            else:
                temp_memory_mask = memory_mask
            if src_key_padding_mask is not None:
                decoder_output, _ = decoder(decoder_input, encoder_output,
                                            memory_mask=temp_memory_mask,
                                            memory_key_padding_mask=src_key_padding_mask[idx].unsqueeze(0),
                                            tgt_mask=decoder_mask)
            else:
                decoder_output, _ = decoder(decoder_input, encoder_output,
                                            memory_mask=temp_memory_mask,
                                            tgt_mask=decoder_mask)
            output = F.log_softmax(decoder_output, dim=-1)

            # get beam_width most value
            log_prob, indexes = output[-1].data.topk(beam_width)
            # print('log_prob:', log_prob)
            # print('indexes:', indexes)
            # print('output:', F.softmax(decoder_output, dim=-1).shape)
            output_prob = F.softmax(decoder_output, dim=-1)[-1]
            next_nodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[-1][new_k].view(1, -1)
                log_p = log_prob[-1][new_k].view(1, -1)
                t_prob = output_prob[-1].view(1, 1, -1)
                # decoded_t = indexes[0][new_k].view(1, -1)
                # log_p = log_prob[0][new_k].view(1, -1)
                # print(decoded_t)
                # print(torch.cat([decoder_input, decoded_t], dim=1).shape)
                # print(decoder_input_prob.shape, t_prob.shape)
                # print(torch.cat([decoder_input_prob, t_prob], dim=1).shape)
                node = Beam(n, torch.cat([decoder_input, decoded_t], dim=1),
                            torch.cat([decoder_input_prob, t_prob], dim=1),
                            n.logp + log_p, n.length + 1)
                # node = Beam(n, decoded_t, n.logp + log_p, n.length + 1)
                score = -node.eval()
                # print(score)
                next_nodes.append((score, node))

            # put nodes into queue
            for i in range(len(next_nodes)):
                score, nextnode = next_nodes[i]
                # print('PUT:', score, nextnode.wordId)
                beam_nodes.put((score, time.time(), nextnode.length, nextnode))

            qsize += len(next_nodes) - 1

        # choose best paths, back trace
        if len(endnodes) == 0:
            endnodes = [beam_nodes.get() for _ in range(n_top)]

        # for score, _, _, n in endnodes:
        #     print('ENCODES:', score, n.wordId)

        utterances = []
        decoded_prob = []
        for score, _, _, n in sorted(endnodes, key=operator.itemgetter(0)):
            # print(n.wordId, score)
            # print(n.wordp)
            utterances.append(n.wordId)
            decoded_prob.append(n.wordp)

        # padded = F.pad(input=lengths, pad=(0, 0, 0, max_length - len(lengths[0])), mode='constant', value=0)
        # length_batch.append(padded)
        for i in range(n_top):
            padded = F.pad(input=utterances[i], pad=(0, max_length - len(utterances[i][0])), mode='constant', value=0)
            decoded_batch.append(padded)
            padded_prob = F.pad(input=decoded_prob[i], pad=(0, 0, 0, max_length - len(utterances[i][0])), mode='constant', value=0)
            # print(padded_prob.shape)
            decoded_prob_batch.append(padded_prob)
    tgt = torch.cat(decoded_batch, 0)
    tgt_prob = torch.cat(decoded_prob_batch, 0)
    
    return tgt, tgt_prob


def beam_decode_rnn(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.2,
                 maxlen: int = 200):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        # pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        # pos_embedding = torch.zeros((maxlen, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size), requires_grad=False)
        self.pos_embedding[:, :, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, :, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)].detach())


# # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float = 0.5,
#                  maxlen: int = 5000):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(-torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         # pos = torch.arange(0, maxlen).unsqueeze(1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         # pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size), requires_grad=False)
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)
#
#     def forward(self, token_embedding: Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
#         # return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)].detach())
    

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Encoder Transformer
class EncoderTransformer(nn.Module):
    def __init__(self, feat_dim, d_model, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu',
                 kernal_size=3, device=None):
        super(EncoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.gelu = nn.GELU()
        self.conv = nn.Conv1d(d_model, d_model, kernal_size, stride=1, padding=kernal_size//2)
        self.require_fc = False
        # print(feat_dim, d_model)
        if feat_dim > d_model:
            self.require_fc = True
            self.fc = nn.Linear(feat_dim, d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        if self.require_fc:
            # print(src.shape)
            src = self.gelu(self.fc(src))
        src = self.gelu(self.conv(src.transpose(1, 2)).transpose(1, 2))
        src = self.positional_encoding(src)
        return self.encoder(src, mask, src_key_padding_mask, is_causal)


# Decoder Transformer
class DecoderTransformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu', device=None):
        super(DecoderTransformer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.tgt_tok_emb(tgt)
        tgt = self.positional_encoding(tgt)
        outs_feat = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        outs = self.generator(outs_feat)
        return outs, outs_feat


# RNN Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, num_layers=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return hidden

# RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, num_layers=1):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        # print(embedded.shape, context.shape)
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(output)

        return prediction, hidden


class Seq2Seq_RNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=1):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=1):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# Reconstruct Decoer Transformer
class ReconstDecoder(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu', device=None):
        super(ReconstDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.positional_encoding(tgt)
        outs_feat = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        return outs_feat


class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        d_model,                  # input feature dimension
        n_embd=768,                # embedding dimension (after convolution)
        dim_feedforward=512,
        n_head=8,                # number of head for self-attention in transformers
        n_embd_ks=3,             # conv kernel size of the embedding network
        max_len=30,               # max sequence length
        arch = (1, 1),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = 19,     # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = True,     # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        self.d_model = d_model
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.gelu = nn.GELU()
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            d_model = n_embd if idx > 1 else d_model
            self.embd.append(
                MaskedConv1D(
                    d_model, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding
        if self.use_abs_pe:
            self.positional_encoding = PositionalEncoding(d_model)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    dim_feedforward, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size,
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.transpose(1, 2).size()
        mask = (~mask).unsqueeze(dim=1)

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.gelu(self.embd_norm[idx](x))

        # positional embedding
        if self.use_abs_pe:
            x = self.positional_encoding(x.transpose(1, 2)).transpose(1, 2)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        print(x.shape)
        exit(0)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks

