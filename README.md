# Analyzing Fast, Frequent, and Fine-Grained Actions in Videos using End-to-End Sequence Modeling
## Overview
Our paper focuses on analyzing fast, frequent, and fine-grained action sequences from videos, which is an important and challenging task for both computer vision and multi-modal LLMs. Existing methods are inadequate in solving this particular problem. For instance, in sports broadcast videos, ambiguous temporal boundaries and transitions between actions make it hard and unnecessary to localize or segment timestamps precisely. Instead, we focus on accurately recognizing fine-grained action types. Besides, frame-wise temporal supervision may even hinder the performance due to over-segmentation. Moreover, having a large number of actions in a short time is especially challenging for video captioning. We build a new large-scale dataset $FineTennis$, motivated by sports analytics tasks, that shows the above challenges. We propose $F^3AST$, a sequence-to-sequence translation model that can effectively recognize fast, frequent, and fine-grained action sequences from videos. $F^3AST$ is efficient and can be trained end-to-end on a single GPU. Finally, we demonstrate that $F^3AST$ significantly outperforms existing methods on 4 fine-grained action sequence datasets.

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the [data](https://github.com/F3AST123/F3AST/tree/main/data) directory for pre-processing and setup instructions.

## Basic usage
To train a model, use `python3 train_vid2seq.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: supports finetennis, badmintonDB, finediving, finegym
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., slowfast)

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

### Trained models
Models and configurations can be found in [f3ast-model](https://github.com/F3AST123/F3AST/tree/main/f3ast-model). Place the checkpoint file and config.json file in the same directory.

To perform inference with an already trained model, use `python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save`. This will output results for 4 evaluation metrics (accuracy, edit score, success rate, and mean Average Precision).

## Data format
Each dataset has plaintext files that contain the list of classes and events in each video: `classes.txt`

This is a list of the class names, one per line: `{split}.json`

This file contains entries for each video and its contained events.
```
[
    {
        "video": VIDEO_ID,
        "num_frames": 518,                 // Video length
        "events": [
            {
                "frame": 100,               // Frame
                "label": CLASS_NAME,        // Event class
            },
            ...
        ],
        "fps": 25,
        "width": 1280,      // Metadata about the source video
        "height": 720
    },
    ...
]
```
**Frame directory**

We assume pre-extracted frames, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <frame_dir>/<video_id>/<frame_number>.jpg. For example,
```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```
Similar format applies to the frames containing objects of interest.







