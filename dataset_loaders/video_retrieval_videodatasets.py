import glob
import json
import os
import pickle
import warnings
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import clip
import ffmpeg
import pandas as pd
import torch
import torchvision
from clip.simple_tokenizer import SimpleTokenizer
from einops.layers.torch import Rearrange
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

VIDEO_AUG = transforms.Compose(
    [
        Rearrange("t h w c -> t c h w"),
        transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
            ]
        ),
        Rearrange("t c h w -> t h w c"),
    ]
)


def _tokenize_max_len(texts, max_len=77):
    tokenizer = SimpleTokenizer()
    if isinstance(texts, str):
        texts = [texts]
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), max_len, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) >= max_len:
            result[i, :max_len] = torch.tensor(tokens[: max_len - 1] + [eot_token])
        else:
            result[i, : len(tokens)] = torch.tensor(tokens)
    return result


def _read_video_train(video_path):
    frame_strides = (8, 16, 16, 24)
    reference_fps = 30
    nframes = 8

    prob = ffmpeg.probe(video_path)
    video_length = float(prob["streams"][0]["duration"])

    frame_stride = frame_strides[torch.randint(0, len(frame_strides), [])]

    segment_duration_sec = nframes / (reference_fps / frame_stride)

    ffmpeg_start_time = 0
    tb = Fraction(1, 1000)

    start_lower = ffmpeg_start_time
    start_upper = max(0, video_length - segment_duration_sec)
    segment_start_sec = (start_lower - start_upper) * torch.rand(
        []
    ).item() + start_upper

    segment_end_sec = segment_start_sec + segment_duration_sec

    video_start = int(segment_start_sec / tb)
    video_end = int(segment_end_sec / tb)

    vid, _, _ = torchvision.io._read_video_from_file(
        video_path,
        seek_frame_margin=5,
        video_width=300,
        video_height=0,
        read_audio_stream=False,
        video_timebase=tb,
        video_pts_range=(video_start, video_end),
    )

    if vid.shape[0] == 0:
        print("Video read failed", video_path)
        vid = torch.zeros(8, 300, 300, 3, dtype=torch.uint8)

    idxs = torch.floor(torch.linspace(0, len(vid) - 1, nframes)).to(torch.int64)
    vid = torch.index_select(vid, 0, idxs)

    vid = VIDEO_AUG(vid)

    return vid


class VideoDatasetMSRVTT(Dataset):
    """Video loader for MSR-VTT dataset with CLIP preprocessing.
    TODO: add option for other preprocessing.

    Args:
        root (string): Path to the directory containing MSRVTT (with *_videodatainfo.json,
            TrainValVideo/, TestVideo/)

        train (bool): True will load training set, False validation or test set depending
            on split

        split (string): Which train/test split to use. One of:
            - 'jsfusion': Called 1k-A in Collaborative Experts, this is the split used in
                JSFusion (Yu et al ECCV18) and has 9000 train videos and 1000 val videos, with one
                arbitary caption per video used at evaluation time. We reuse the
                caption indices from jsfusion_val_caption_idx.pkl as in Collaborative
                Experts
            - 'miech': Called 1k-B in Collaborative Experts, this split is used in Miech et
                al (arXiv:1804.02516) and has 6656 train videos and 1000 test videos.
                At evaluation we use the first caption as in Collaborative Experts code.
            - 'full-val': Official split of 6513 train and 497 val, all captions
            - 'full-test': Official split of 6513 train and 2990 test, all captions
    """

    def __init__(self, root="/data/MSRVTT", train=True, split=None, augment=False):
        self.train = train
        self.augment = augment

        ce_meta_dir = Path("dataset_loaders/msrvtt_meta/")
        json_files = ["train_val_videodatainfo.json", "test_videodatainfo.json"]
        video_folders = ["TrainValVideo", "TestVideo"]

        caption_indices_file = None
        if split == "miech":
            txt_file = "train_list_miech.txt" if train else "test_list_miech.txt"
        elif split == "jsfusion":
            txt_file = "train_list_jsfusion.txt" if train else "val_list_jsfusion.txt"
            if not train:
                # Found this file in the old version of the tarball
                # from github details link above
                caption_indices_file = "jsfusion_val_caption_idx.pkl"
        elif split == "full-val":
            txt_file = "train_list_full.txt" if train else "val_list_full.txt"
        elif split == "full-test":
            txt_file = "train_list_full.txt" if train else "test_list_full.txt"
        else:
            raise Exception("Unknown MSRVTT split")

        txt_file = ce_meta_dir / txt_file
        with open(txt_file, "r") as f:
            video_ids = [x.strip() for x in f.read().split("\n") if x.strip() != ""]

        print("MSRVTT split %s, %d files" % (split, len(video_ids)))

        sent_dict = defaultdict(lambda: [])
        for json_file in json_files:
            json_file_abs = os.path.join(root, json_file)
            metadata = json.load(open(json_file_abs))

            for s in metadata["sentences"]:
                sent_dict[s["video_id"]].append(s["caption"])

        video_file_dict = {}
        for vf in video_folders:
            vf_abs = os.path.join(root, vf)
            mp4s = [x for x in os.listdir(vf_abs) if x.endswith(".mp4")]
            for m in mp4s:
                video_file_dict[m.split(".")[0]] = os.path.join(vf_abs, m)

        if caption_indices_file is not None:
            with open(ce_meta_dir / caption_indices_file, "rb") as f:
                caption_indices = pickle.load(f)

            for c, i in caption_indices.items():
                sent_dict[c] = [sent_dict[c][i]]

        if not train and split == "miech":
            # Use first caption
            for k, v in sent_dict.items():
                sent_dict[k] = [sent_dict[k][0]]

        self.video_files = []
        for v_id in video_ids:
            self.video_files.append(video_file_dict[v_id])

        self.captions = sent_dict

        self.preprocess = CLIP_TRANSFORM

    def __len__(self):
        if self.augment and self.train:
            # Fake longer length to avoid too many val passes
            return 5 * len(self.video_files)
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx % len(self.video_files)]
        vid_id = video_path.split("/")[-1:][0][:-4]

        if self.augment:
            vid = _read_video_train(video_path)
        else:
            vid, _, _ = torchvision.io._read_video_from_file(
                video_path, read_audio_stream=False
            )
        images = []
        for frame in vid:
            images.append(
                self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
            )

        vid = torch.stack(images)
        title = self.captions[vid_id]

        if self.augment:
            if not self.train:
                warnings.warn(
                    "MSRVTT: augment is true, but not using training set -- "
                    " the output will not be deterministic"
                )
            captions = self.captions[vid_id]
            idxs = torch.multinomial(torch.ones(len(captions)), len(captions))
            captions = [captions[idx] for idx in idxs[0:6]]
            title = captions[0]
            fake_comments = captions[1:]
            assert len(fake_comments) == 5

            fake_comments = clip.tokenize(fake_comments)
            text = clip.tokenize([title])[0]

            out = vid, text, fake_comments, {}
        else:
            title = self.captions[vid_id]
            fake_comments = []

            try:
                text = clip.tokenize(title)
            except Exception as e:
                print(f"Failed to tokenize {title}", str(e))
                text = clip.tokenize(title[:20])

            out = vid, text, vid_id
        return out


class VideoDatasetMSVD(Dataset):
    """Video loader for MSVD with CLIP preprocessing.

    Args:
        root (string): Path to the directory containing MSVD (with YouTubeClips/*.avi)
        from https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar

        train (bool): True will load training set, False validation or test set depending
            on split

        split (string): Which train/test split to use. One of:
            - 'val': 1200 train videos, 100 val videos
            - 'test': 1200 train videos, 670 test videos
    """

    def __init__(self, root="/data/MSVD", train=True, split=None, augment=False):
        root = Path(root)
        self.root = root
        self.train = train
        self.augment = augment

        ce_meta_dir = Path("dataset_loaders/msvd_meta/")

        if split == "val":
            txt_file = "train_list.txt" if train else "val_list.txt"
        elif split == "test":
            txt_file = "train_list.txt" if train else "test_list.txt"
        else:
            raise Exception("Unknown MSVD split")

        caption_file = "raw-captions.pkl"
        with open(ce_meta_dir / caption_file, "rb") as f:
            self.captions = pickle.load(f)

        with open(ce_meta_dir / txt_file, "r") as t:
            lines = t.read().split("\n")
            self.video_ids = [l.strip() for l in lines if l.strip() != ""]

        self.video_files = []

        nmissing = 0

        for v in self.video_ids:
            vfile = root / "YouTubeClips" / (v + ".avi")

            if vfile.exists():
                self.video_files.append(str(vfile))
            else:
                nmissing += 1

        self.preprocess = CLIP_TRANSFORM

        print(len(self.video_files), "loaded files", nmissing, "missing files")
        assert nmissing == 0

    def __len__(self):
        if self.augment and self.train:
            # Fake longer length to avoid too many val passes
            return 5 * len(self.video_files)
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx % len(self.video_files)]
        vid_id = video_path.split("/")[-1:][0][:-4]

        if self.augment:
            vid = _read_video_train(video_path)
        else:
            vid, _, _ = torchvision.io._read_video_from_file(
                video_path, read_audio_stream=False
            )
        images = []
        for frame in vid:
            images.append(
                self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
            )

        vid = torch.stack(images)

        if self.augment:
            if not self.train:
                warnings.warn(
                    "MSVD: augment is true, but not using training set -- "
                    " the output will not be deterministic"
                )
            captions = self.captions[vid_id]
            captions = [" ".join(s) for s in captions]

            idxs = torch.multinomial(torch.ones(len(captions)), len(captions))
            captions = [captions[idx] for idx in idxs[0:6]]
            title = captions[0]
            fake_comments = captions[1:]
            assert len(fake_comments) == 5

            fake_comments = clip.tokenize(fake_comments)
            text = clip.tokenize([title])[0]

            out = vid, text, fake_comments, {}
        else:
            title = self.captions[vid_id]
            title = [" ".join(s) for s in title]

            try:
                text = clip.tokenize(title)
            except Exception as e:
                print(f"Failed to tokenize {title}", str(e))
                text = clip.tokenize(title[:20])

            out = vid, text, vid_id

        return out


class VideoDatasetActivityNet(Dataset):
    """Video loader for ActivityNet with CLIP preprocessing.

    Loads ActivityNet videos preprocessed with CLIP

    Details:
        https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/activity-net

    Args:
        root (string): Path to the directory containing ActivityNet
            features from Collaborative Experts corresponding to
            ``data/activity-net/structured-symlinks`` from the downloaded
            tarball

        train (bool): True will load training set with 10009 videos, False validation set

        split (string): Which train/test split to use. One of:
            - 'val': uses val_1 with 4917 videos
            - 'test': uses val_2 with 4885 videos


        train_caption_sampling (string):
            One of ``['all', 'first', 'random']`` specifying how to sample captions
            at train time, since there are multiple captions per video.

            For the different options the caption embeddings returned by __getitem__ are as follows:

            - 'all': [n_captions, caption_feature_dim] tensor containg embeddings
                of all the captions for the video
            - 'first': [caption_feature_dim] shape tensor of the first caption
            - 'random': [caption_feature_dim] shape tensor of a random caption

        test_caption_sampling (string):
            As above but at test time
    """

    def __init__(
        self,
        root="/scratch/shared/beegfs/albanie/shared-datasets/activity-net/",
        train=True,
        split=None,
    ):
        root = Path(root)
        self.root = root
        self.train = train

        ce_meta_dir = Path(
            "/scratch/shared/beegfs/albanie/shared-datasets/activity-net/structured-symlinks/"
        )

        if split == "val":
            txt_file = "train_list.txt" if train else "val_1_list.txt"
        elif split == "test":
            txt_file = "train_list.txt" if train else "val_2_list.txt"
        else:
            raise Exception("Unknown Activitynet split")

        caption_file = "raw-captions.pkl"
        with open(ce_meta_dir / caption_file, "rb") as f:
            self.captions = pickle.load(f)

        with open(ce_meta_dir / txt_file, "r") as t:
            lines = t.read().split("\n")
            self.video_ids = [l.strip() for l in lines if l.strip() != ""]

        self.video_files = []

        nmissing = 0

        for v in self.video_ids:
            vfile = root / "videos" / (v + ".mp4")

            if vfile.exists():
                self.video_files.append(str(vfile))
            else:
                nmissing += 1

        self.preprocess = CLIP_TRANSFORM

        print(len(self.video_files), "loaded files", nmissing, "missing files")
        assert nmissing == 0

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        vid_id = video_path.split("/")[-1:][0][:-4]

        vid, _, _ = torchvision.io._read_video_from_file(
            video_path, read_audio_stream=False
        )
        images = []
        for frame in vid:
            images.append(
                self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
            )

        vid = torch.stack(images)
        caption = self.captions[vid_id]
        caption = [" ".join(s) for s in caption]

        text = clip.tokenize(caption)

        return vid, text, vid_id


class VideoDatasetK700Comments(Dataset):
    def __init__(
        self,
        root="/data",
        kinetics_csv="/data/oxford_project/kinetics700_havedescs.csv",
        train=False,
        split="test",
    ):
        # This loader is only intended for testing
        assert train is False
        assert split == "test"

        df = pd.read_csv(kinetics_csv)
        self.video_files = []
        self.titles = []
        self.comments = []
        self.descriptions = []
        self.preprocess = CLIP_TRANSFORM

        k400train = glob.glob(
            os.path.join(root, "kinetics400", "train", "**", "*.mp4"), recursive=True
        )
        k700train = glob.glob(
            os.path.join(root, "kinetics700", "train", "**", "*.mp4"), recursive=True
        )

        train_ids = set(x.split("/")[-1].split(".")[0] for x in k700train) | set(
            x.split("/")[-1].split(".")[0] for x in k400train
        )
        assert len(train_ids) == 529841

        for _, row in df.iterrows():
            # Make sure we don't use any video id that has been used for training
            # this is a bit complicated because there is the potential for the same
            # video to be in different sets in different kinetics versions, or
            # for multiple clips from the same video to be in different sets (yet
            # they would have the same title and comments)

            is_val = (
                "/test/" in row.video_path
                and row.kinetics_id not in train_ids
                and row.title_lang == "en"
                and not pd.isna(row.comments)
                and len(json.loads(row.comments)) >= 3
            )

            if is_val:
                self.video_files.append(os.path.join(root, row.video_path))
                self.titles.append(row.title)
                self.comments.append(json.loads(row.comments))
                self.descriptions.append(row.description)

        print(len(self.video_files), "kinetics comments val files")

    def __getitem__(self, index):
        video_path = self.video_files[index]
        title = self.titles[index]
        comments = self.comments[index]
        vid_id = video_path.split("/")[-1].split(".")[0]

        vid, _, _ = torchvision.io._read_video_from_file(
            video_path, read_audio_stream=False
        )
        images = []
        for frame in vid:
            images.append(
                self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
            )

        vid = torch.stack(images)
        title_tok = _tokenize_max_len(title)
        comments_tok = _tokenize_max_len(comments)

        return vid, title_tok, comments_tok, vid_id

    def __len__(self):
        return len(self.video_files)


if __name__ == "__main__":
    x = VideoDatasetK700Comments(
        train=False,
        split="test",
        root="/data",
        kinetics_csv="/data/oxford_project/kinetics700_havedescs.csv",
    )
    y = x[1]
