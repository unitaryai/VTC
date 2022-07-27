# VTC: Code and Models for Improving Video-Text Retrieval with User Comments

![vtc_fig.png](vtc_fig.png?style=centerme)

This repository is the official implementation of "VTC: Improving Video-Text Retrieval with User Comments". Code to to download the dataset can be found [here](https://github.com/unitaryai/VTC-dataset).

Our main contributions:

1) We introduce a new dataset VTC of videos, titles, and comments.

2) We introduce a new hierarchical attention method that learns how to identify relevant
auxiliary information, and that can learn representations that even generalize to
other datasets.

3) We quantify the value of
the comments modality for video-text learning and we show that by using comments, our method is able
to learn better, more contextualised, representations for image, video and audio representations.

Project template is from https://github.com/victoresque/pytorch-template.

## Installation

In order to install the conda environment, [Anaconda](https://conda.io/docs/user-guide/install/download.html) will need to be installed first.


```bash
# Clone the repository
git clone https://github.com/unitaryai/VTC
cd VTC

# Create a new conda environment with dependencies
conda env create -f environment.yml

```


## Training

All relevant configs will need to be updated with the right paths to the downloaded files e.g. dataset csv file, root directory where the images/videos are or model paths.

Generally, the Context Adapter Module models in the paper are trained using:
```bash
# image preview
python train.py --config "configs/pretrained_clip_comments_attention.jsonc"

# video
python train.py --config "configs/pretrained_clip_timesformer_comments_attention.jsonc"

```
For more configurable flag options run `python train.py --help`.

To replicate our experiments in the paper, run the following scripts:

```bash
# training image baselines (Table 2)
bash experiments/train/image_baselines.sh
# training CAM module with varying number of comments (Figure 4)
bash experiments/train/image_vary_num_comments.sh
# training video models using timesformer (Table 7)
bash experiments/train/video_timesformer.sh
```
When training the frozen models, we recommend saving the embeddings for the visual side to speed up the training. This can be done by running:
```bash
python scripts/get_clip_vit_embeddings.py
```
The saved embeddings can be passed to the dataset loader in the config as `cached_vision_features` or simply passed as an additional flag when running with `python train.py --cached_vision_features $SAVED_FEATURES_PATH`.

For audio experiments, first clone the GDT repository and download the model weights trained on the IG65M dataset.
```bash
git clone https://github.com/facebookresearch/GDT.git
wget https://dl.fbaipublicfiles.com/GDT/gdt_IG65M.pth
```
Then compute the audio embeddings for the videos in the VTC dataset that will be used in the training.
```bash
python scripts/get_audio_embeddings.py
```
## Evaluation

To evaluate a model, update the relevant configs with the right paths to the downloaded files e.g. dataset csv file, root directory where the images/videos are or model paths. Then you can run:

```bash
# image
python evaluation/eval.py --config "configs/pretrained_clip_comments_attention.jsonc" --resume $CKPT_PATH

# video
python evaluation/eval.py --config "configs/pretrained_clip_timesformer_comments_attention.jsonc" --resume $CKPT_PATH

```

For more configurable flag options run `python evaluation/eval.py --help`.

To replicate our experiments in the paper, run the following scripts:
```bash
# testing image baselines (Table 2)
bash experiments/eval/image_baselines.sh
# testing CAM module with varying number of comments (Figure 4)
bash experiments/eval/image_vary_num_comments.sh
# testing video models using timesformer (Table 7) and testing on different datasets (VTC, KineticsComments, Livebot)
bash experiments/eval/video_timesformer.sh
```


## Results &  Pre-trained Models

We will be releasing pre-trained models soon.
### Combining Modalities
We show that our method is robust to different combinations of modalities, both at train and at test time.


| training   | inference  | TVR R@1  | TVR R@10   | VTR R@1 | VTR R@10 |
|--------------------------|------------------|-------------------|------------------|-------------------|--------------------------|
| CLIP                 | img+title            | 11.1                                                  | 26.0                                                  | 11.1          | 25.3           |
| img+title            | img+title            | 15.5                                                  | 34.9                                                  | 14.4          | 33.4           |
<br>
| img+title+cmts       | img+title            | 15.5                                                  | 34.5                                                  | 14.4          | 33.3           |
| img+title+cmts       | img+title+cmts       | 18.0                                                  | 43.2                                                  | 18.7          | 43.9           |
| img+title+cmts     | img+title            | 14.9                                                  | 34.2                                                  | 14.2          | 32.9           |
|img+title+cmts     | img+title+cmts       | 28.2                                                  | 51.2                                            | 25.1          | 49.9           |
<br>
| img+title+cmts+audio | img+title            | 15.4                                                  | 34.0                                                  | 14.3          | 32.9           |
| img+title+cmts+audio | img+title+audio      | 15.8                                                  | 36.9                                                  | 12.2          | 30.4           |
| img+title+cmts+audio | img+title+cmts+audio | 19.6                                                  | 45.6                                                  | 20.6          | 47.2           |

### Video results
These experiments are using video frames and were trained adapting the video branch with comments, with either one or eight frames from the video. Showing Recall@10.


| inference | #frames | VTC VTR | VTC TVR| KineticsComms VTR | KineticsComms TVR | LiveBotEN VTR | LiveBotEN TVR |
|--------------------|-------------------|---------------------------------------|--------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| video              | 1                 | 28.9                                  | 28.3                                       | 48.8                                   | 46.9         | 48.0         | 49.0         |
| video+comments     | 1                 | 40.8                                  | 41.0                                       | 61.1                                   | 59.2         | 64.0         | 64.0         |
| mean-pooling       | 8                 | 19.3                                  | 24.2                                       | 54.1                                   | 49.8         | 69.0         | 66.0         |
| video              | 8                 | 28.9                                  | 27.6                                       | 56.9                                   | 55.8         | 70.0         | 72.0         |
| video+comments     | 8                 | 41.5                                  | 41.9                                       | 68.0                                   | 66.1         | 69.0         | 80.0         |


## Citation


```text
@inproceedings{hanu2022vtc,
    title={{VTC: Improving Video-Text Retrieval with User Comments}},
    author={Laura Hanu and James Thewlis and Yuki M. Asano and Christian Rupprecht},
    booktitle={ECCV},
    year={2022}
}
```