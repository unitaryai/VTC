import ast
import glob
import json
import os
from fractions import Fraction

import clip
import numpy as np
import pandas as pd
import torch
import torchvision
from clip.simple_tokenizer import SimpleTokenizer
from einops.layers.torch import Rearrange
from PIL import Image
from rake_nltk import Rake
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset_loaders.video_retrieval_videodatasets import VideoDatasetMSRVTT  # noqa
from dataset_loaders.video_retrieval_videodatasets import (
    VideoDatasetK700Comments,
    VideoDatasetMSVD,
    _tokenize_max_len,
)

__all__ = [
    "FeaturesDataset",
    "ImTextDataset",
    "VideoDatasetFirst1800",
    "VideoDatasetFirst32",
    "VideoDatasetK700Comments",
    "VideoDatasetLivebot",
    "VideoDatasetMSRVTT",
    "VideoDatasetMSVD",
    "VideoDatasetReddit",
    "VideoDatasetSegments",
]

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

IMG_AUG = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
            ]
        ),
    ]
)

# Transforms on [t,h,w,c] uint8 videos
# "Deterministic or random transformations applied on the batch of Tensor Images
# identically transform all the images of the batch."
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

BOT_TEXT_TO_AVOID = [
    "i am a bot",
    "i'm a bot",
    "this is a bot",
    "redditspeedbot",
    "this bot",
    "look at my programming",
    "look at my source code on github",
    "this is a manual removal by a *human moderator*",
    "your post was removed",
    "this post was removed",
    "your post has been removed",
    "community moderation bot",
    "unfortunately it has been removed",
    "thank you for your submission",
    "your submission has been removed",
    "if you feel this was done in error",
    "your post breaks",
    "has been removed for the following reasons",
    "downvote this comment if",
    "redditdownloader",
    "repostsleuthbot",
    "vreddit",
    "savethisvideo",
    "stabbot",
    "[removed]",
    "[deleted]",
    "[excluído]",
    "savevideo",
    "this comment",
]


def random_blank(strs, p):
    for i in range(len(strs)):
        if torch.rand([]) < p:
            strs[i] = ""
    return strs


def partition_dataframe(df, root=None, split=None):
    """
    Partition into train/test/val

    df: Pandas dataframe from CSV
    split: 'train', 'test' or 'val'
    """

    mp4s = df.video_path.tolist()
    ids = [x.split("/")[-1].split(".")[0] for x in mp4s]

    # The least significant digit of the base36 id is quasi-random
    # so use it to partition into train and test
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    digit_split = {}

    digit_split["test"] = set(digits[0:4])
    digit_split["val"] = set(digits[4:8])
    digit_split["train"] = set(digits[8:])

    if root is not None:
        # Check for missing files
        available_mp4s = glob.glob(os.path.join(root, "**/*.mp4"), recursive=True)
        available_ids = set(x.split("/")[-1].split(".")[0] for x in available_mp4s)

        # Corrupt jhgxv7.mp4
        available_ids -= {"jhgxv7"}

        print(
            "CSV: %d Available on Disk: %d"
            % (len(ids), len(set(ids).intersection(available_ids)))
        )

        keep = [id[-1] in digit_split[split] and id in available_ids for id in ids]
    else:
        keep = [id[-1] in digit_split[split] for id in ids]

    return df[keep]


def load_features(df, path):
    # Load from PTH
    features_stored = torch.load(path)

    if "reddit_id_to_comment_id" in features_stored:
        # Handle comments
        reddit_ids = list(features_stored["reddit_id_to_comment_id"].keys())
        embeddings = features_stored["embeddings"]
        lookup = {int(el): i for i, el in enumerate(reddit_ids)}
        sel = [lookup[rid] for rid in df.reddit_id]
        # embeddings is a list of lists of zero-or-more tensors
        feats = [embeddings[s] for s in sel]
        assert len(feats) == len(df)
        return feats
    else:
        # Handle not
        assert features_stored["reddit_ids"].dtype is torch.int64
        assert features_stored["embeddings"].dtype is torch.float32
        lookup = {int(el): i for i, el in enumerate(features_stored["reddit_ids"])}
        sel = [lookup[rid] for rid in df.reddit_id]
        feats = features_stored["embeddings"][sel]
        assert feats.shape[0] == len(df)
        return feats


def filter_by_k_comments(df, k=3, limit=None):
    filtered_ids = []
    for _, row in tqdm(df.iterrows()):
        if len(ast.literal_eval(row.comments)) >= k:
            filtered_ids.append(row.reddit_id)

    new_df = df[df.reddit_id.isin(filtered_ids)]
    if limit is not None and len(filtered_ids) > limit:
        # random_state=1 to ensure reproducibility
        new_df = new_df.sample(n=limit, random_state=1)
    return new_df


class VisionTitleCommentDatasetBase(Dataset):
    def split_dataset(
        self, csv_file, df, train, test, test_on_over_k_comms=None, test_set_limit=None
    ):
        if test:
            assert not train
            new_df = partition_dataframe(df, split="test")
        else:
            new_df = partition_dataframe(df, split="train" if train else "val")
        if test_on_over_k_comms is not None and not train:
            new_df = filter_by_k_comments(
                new_df, test_on_over_k_comms, limit=test_set_limit
            )
        return new_df

    def should_add_comments(self, add_comments, train):
        cases = {
            "always": [True, True],
            "train_only": [False, True],
            "never": [False, False],
        }

        return cases[add_comments][int(train)]

    def _tokenise(self, texts, max_len=77):
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), max_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) >= max_len:
                # summarise text by extracting keywords
                self.rake.extract_keywords_from_text(texts[i])
                a = self.rake.get_ranked_phrases()
                tokens = [sot_token] + self.tokenizer.encode(" ".join(a)) + [eot_token]
                if len(tokens) >= max_len:
                    result[i, :max_len] = torch.tensor(
                        tokens[: max_len - 1] + [eot_token]
                    )
                else:
                    result[i, : len(tokens)] = torch.tensor(tokens)
            else:
                result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def preprocess_comments(self, comments, sampling=None, num_comms=2):
        if num_comms == 0:
            return []
        # TODO have comm ids in separate column
        if len(comments) > 0 and isinstance(comments[0], tuple):
            comments = [
                comm[0]
                for comm in comments
                if all(s not in comm[0].lower() for s in BOT_TEXT_TO_AVOID)
            ]
        else:
            comments = [
                comm
                for comm in comments
                if all(s not in comm.lower() for s in BOT_TEXT_TO_AVOID)
            ]

        if len(comments) >= num_comms:
            if sampling == "random":
                idxs = torch.multinomial(torch.ones(len(comments)), len(comments))
                comments = [comments[idx] for idx in idxs[:num_comms]]

            elif sampling is None:
                comments = comments[:num_comms]
        while len(comments) < num_comms:
            comments.append("")

        return comments

    def _load_reddit(self, df, file_extension=".mp4"):
        files = [x[len("results/") : -4] + file_extension for x in df.video_path]
        exists = np.array([os.path.exists(os.path.join(self.root, x)) for x in files])
        if exists.sum() != len(files):
            print("%d files found out of %d in CSV" % (exists.sum(), len(files)))
        df = df[exists]
        files2 = [x[len("results/") : -4] + file_extension for x in df.video_path]
        self.filenames += [os.path.join(self.root, x) for x in files2]
        self.ids += df.reddit_id.to_list()
        self.titles += df.title.to_list()
        self.video_lengths += df.video_length.to_list()
        self.comments += [ast.literal_eval(c) for c in df.comments]

        print(len(self.ids), "reddit videos")

    def _load_kinetics(self, df):
        nk = 0
        for ki in range(len(df)):
            row = df.iloc[ki]
            vp = os.path.join(self.kinetics_root, row.video_path)
            split_k700 = row.split_k700
            split_k400 = row.split_k400

            istrain = (
                split_k700 == "train"
                and (split_k400 == "train" or pd.isna(split_k400))
                and "/train/" in row.video_path
            )

            if istrain and os.path.exists(vp):
                self.filenames.append(vp)
                self.ids.append(-1)
                self.titles.append(row.title_en)
                self.video_lengths.append(row.video_length)
                comms = [] if pd.isna(row.comments) else json.loads(row.comments)

                if not pd.isna(row.description_en):
                    desc_sentences = [
                        x.strip() for x in row.description_en.split(".") if len(x) > 60
                    ]
                    comms.extend(desc_sentences)

                self.comments.append(comms)

                nk += 1
        print(nk, "kinetics videos")
        assert nk > 400000

    def _load_howto100m(self, df):
        nk = 0
        for ki in range(len(df)):
            row = df.iloc[ki]
            vp = os.path.join(self.howto100m_root, row.video_path)

            if os.path.exists(vp):
                self.filenames.append(vp)
                self.ids.append(-1)
                self.titles.append(row.title)
                self.video_lengths.append(row.video_length)

                comms = [] if pd.isna(row.comments) else json.loads(row.comments)

                if not pd.isna(row.description):
                    desc_sentences = [
                        x.strip() for x in row.description.split(".") if len(x) > 60
                    ]
                    comms.extend(desc_sentences)

                self.comments.append(comms)

                nk += 1
        print(nk, "howto100m videos")
        assert nk > 80000

    def _read_video(self, idx):
        id = self.ids[idx]
        video_path = self.filenames[idx]
        video_length = min(60, self.video_lengths[idx])
        frame_stride = self.frame_strides[torch.randint(0, len(self.frame_strides), [])]

        segment_duration_sec = self.nframes / (self.reference_fps / frame_stride)

        # Often the reddit videos have an offset of 1.4s to the
        # start time meaning that there are no video frames
        # in the range (0, 1.4).
        # The offset can be obtained with ffprobe, eg:
        #
        #     prob = ffmpeg.probe(video_path)
        #     start_time = float(prob["streams"][0]["start_time"])
        #
        # unfortunately this is not exposed in the torchvision api
        # and calling ffprobe makes things slow, so for now just
        # assume 1.4 for all videos (TODO: precompute for all videos)
        ffmpeg_start_time = 0 if id == -1 else 1.4

        # For simplicity just use milliseconds as the timebase
        # (which defines the unit used in video_pts_range, given
        # that the range must be integers)
        #
        # The video's native timebase is obtained with
        #
        #     prob = ffmpeg.probe(video_path)
        #     tb = Fraction(prob["streams"][0]["time_base"])
        #
        # although the given timebase is converted to ffmpeg's
        # internal AV_TIME_BASE so the choice is somewhat arbitrary
        tb = Fraction(1, 1000)

        if self.train:
            start_lower = ffmpeg_start_time
            start_upper = max(0, video_length - segment_duration_sec)
            segment_start_sec = (start_lower - start_upper) * torch.rand(
                []
            ).item() + start_upper
        else:
            segment_start_sec = 0

        segment_end_sec = segment_start_sec + segment_duration_sec

        video_start = int(segment_start_sec / tb)
        video_end = int(segment_end_sec / tb)

        # For now use this private method since it allows resizing
        # on the ffmpeg side which is faster.
        # A large seek_frame_margin seems to be needed
        # to seek accurately
        vid, _, _ = torchvision.io._read_video_from_file(
            video_path,
            seek_frame_margin=5,
            video_width=self.video_read_width,
            video_height=self.video_read_height,
            read_audio_stream=False,
            video_timebase=tb,
            video_pts_range=(video_start, video_end),
        )

        if vid.shape[0] == 0:
            print("Zero len vid, trying fallback", video_path)
            vid, _, _ = torchvision.io._read_video_from_file(
                video_path,
                video_width=self.video_read_width,
                video_height=self.video_read_height,
                read_audio_stream=False,
                video_timebase=Fraction(1),
                video_pts_range=(0, 5),
            )

        if vid.shape[0] == 0:
            print("Fallback failed", video_path)
            vid = torch.zeros(8, 300, 300, 3, dtype=torch.uint8)

        idxs = torch.floor(torch.linspace(0, len(vid) - 1, self.nframes)).to(
            torch.int64
        )
        vid = torch.index_select(vid, 0, idxs)

        vid = self.video_tfm(vid)

        return vid


class VideoDatasetSegments(VisionTitleCommentDatasetBase):
    """
    A video loader that selects a random segment from
    each video and does data augmentation through cropping,
    variable speed and color jitter.

    Returns frames in [t h w c] order
    """

    def __init__(
        self,
        csv_file,
        root,
        train=True,
        test=False,
        add_comments="train_only",
        num_comms=2,
        comment_sampling="random",
        use_kinetics_train=None,
        kinetics_csv=None,
        kinetics_root=None,
        use_howto100m_train=None,
        howto100m_csv=None,
        howto100m_root=None,
        first_frame_only=False,
        test_on_over_k_comms=None,
        test_set_limit=None,
    ):
        self.train = train
        self.root = root
        self.kinetics_root = kinetics_root
        self.howto100m_root = howto100m_root
        self.num_comms = num_comms
        self.comment_sampling = comment_sampling if train else None
        self.first_frame_only = first_frame_only
        self.test_on_over_k_comms = test_on_over_k_comms
        self.test_set_limit = test_set_limit

        self.add_comments = self.should_add_comments(add_comments, train)

        self.video_read_height = 300
        self.video_read_width = 0
        self.nframes = 8
        self.reference_fps = 30

        self.frame_tfm = CLIP_TRANSFORM
        self.tokenizer = SimpleTokenizer()
        self.rake = Rake()

        if self.train:
            self.video_tfm = VIDEO_AUG
            self.frame_strides = (4, 8, 16, 32)
        else:
            self.video_tfm = transforms.Compose([])
            self.frame_strides = (16,)

        self.ids = []
        self.filenames = []
        self.titles = []
        self.video_lengths = []
        self.comments = []

        # Always use reddit for val
        use_reddit = (not train) or (
            use_kinetics_train != "only" and use_howto100m_train != "only"
        )
        use_kinetics = train and use_kinetics_train in ("combine", "only")
        use_howto100m = train and use_howto100m_train in ("combine", "only")
        assert not (use_kinetics_train == "only" and use_howto100m_train == "only")

        if use_reddit:
            df = pd.read_csv(csv_file)
            df = self.split_dataset(
                csv_file,
                df,
                train,
                test,
                test_on_over_k_comms=test_on_over_k_comms,
                test_set_limit=test_set_limit,
            )
            self._load_reddit(df)

        if use_kinetics:
            df_kinetics = pd.read_csv(kinetics_csv)
            self._load_kinetics(df_kinetics)

        if use_howto100m:
            df_howto100m = pd.read_csv(howto100m_csv)
            self._load_howto100m(df_howto100m)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        id = self.ids[idx]
        title = self.titles[idx]
        comments = self.comments[idx]

        vid = self._read_video(idx)

        images = [self.frame_tfm(Image.fromarray(frame.numpy())) for frame in vid]
        vid = torch.stack(images)

        if self.first_frame_only:
            vid = images[0]

        title_tok = self._tokenise([title])[0]

        if self.add_comments:
            comments = self.preprocess_comments(
                comments, sampling=self.comment_sampling, num_comms=self.num_comms
            )

            comments_tok = self._tokenise(comments)

            if torch.rand([]) < 0.0001:
                print(
                    "Debug dataloader -- title:",
                    title,
                    "comms:",
                    comments,
                )
        else:
            comments_tok = self._tokenise([""])

        meta = {"id": id}
        return vid, title_tok, comments_tok, meta


class VideoDatasetFirst32(Dataset):
    """A simple video loader that just returns the first 32 frames
    rescaled to 128x172 ignoring aspect ratio and doesn't do
    any frame rate resampling

    Tensor is padded with black frames if there are under 32
    frames

    Returns frames in [c t h w] order
    """

    def __init__(
        self,
        csv_file,
        root,
        text_features=None,
        train=True,
        should_partition_dataframe=True,
        clip_preprocess=False,
    ):
        self.train = train

        self.height = 128
        self.width = 171
        self.nframes = 32

        df = pd.read_csv(csv_file)
        if should_partition_dataframe:
            df = partition_dataframe(df, root=root, split="train" if train else "val")

        self.video_files = []

        for i in range(len(df)):
            vp = df.video_path.iloc[i][len("results/") :]
            vp = os.path.join(root, vp)
            self.video_files.append(vp)

        self.ids = df.reddit_id.to_list()
        self.titles = df.title.to_list()
        self.clip_preprocess = clip_preprocess

        if clip_preprocess:
            self.tfms = CLIP_TRANSFORM
        else:
            self.tfms = transforms.Compose(
                [
                    Rearrange("t h w c -> t c h w"),
                    transforms.ConvertImageDtype(torch.float32),
                    # For ig65m https://github.com/moabitcoin/ig65m-pytorch/blob/master/ig65m/cli/extract.py#L64
                    transforms.Normalize(
                        mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989],
                    ),
                ]
            )

        # Ordering used in ig65m and torchvision
        # https://github.com/pytorch/vision/tree/master/torchvision/models/video
        self.final_rearrange = Rearrange("t c h w -> c t h w")

        if text_features is not None:
            self.text_feats = load_features(df, text_features)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        id = self.ids[idx]
        title = self.titles[idx]

        # For now use this private method since it allows resizing
        # on the ffmpeg side which is faster
        # Get the first 4 seconds which should get us at least 32
        # frames at reasonable frame rates
        vid, _, _ = torchvision.io._read_video_from_file(
            video_path,
            video_width=self.width,
            video_height=self.height,
            read_audio_stream=False,
            video_timebase=Fraction(1),
            video_pts_range=(0, 4),
        )

        vid = vid[0 : self.nframes, ...]

        if vid.shape[0] < self.nframes:
            # Padding
            length = vid.size(0)
            out_tensor = vid.new_full((self.nframes, self.height, self.width, 3), 0.0)
            if length == 0:
                print("Zero length video!", video_path)
            else:
                out_tensor[:length, ...] = vid
            vid = out_tensor

        if self.clip_preprocess:
            images = []
            for frame in vid:
                images.append(self.tfms(Image.fromarray(frame.numpy()).convert("RGB")))
            vid = torch.stack(images)
            try:
                text = clip.tokenize(title)
            except Exception as e:
                print(f"Failed to tokenize {title}", str(e))
                text = clip.tokenize(title[:20])
        else:
            vid = self.tfms(vid)
            vid = self.final_rearrange(vid)
            text = self.text_feats[idx]
        meta = {"id": id}
        return vid, text, meta


class VideoDatasetFirst1800(Dataset):
    """A simple video loader that returns the first 1800 frames
    (which would be the first minute at 30fps), returning
    a shorter tensor if there are not enough frames, but with
    a minimum length of 32 frames padded with black frames.

    Framerate is not resampled

    To emulate preprocessing used in collab experts, videos
    are first resized to height 256 (preserving aspect ratio)
    and then to smaller edge 128 (preserving aspect ratio) and
    then a 112x112 center crop is taken.

    Returns frames in [c t h w] order
    """

    def __init__(self, csv_file, root, train=True, should_partition_dataframe=True):
        self.train = train

        self.video_read_height = 256
        self.height = 128
        self.crop_size = 112
        self.nframes = 1800
        self.min_nframes = 32

        df = pd.read_csv(csv_file)
        if should_partition_dataframe:
            df = partition_dataframe(df, root=root, split="train" if train else "val")

        self.video_files = []

        for i in range(len(df)):
            vp = df.video_path.iloc[i][len("results/") :]
            vp = os.path.join(root, vp)
            self.video_files.append(vp)

        self.tfms = transforms.Compose(
            [
                Rearrange("t h w c -> t c h w"),
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ConvertImageDtype(torch.float32),
                # For ig65m https://github.com/moabitcoin/ig65m-pytorch/blob/master/ig65m/cli/extract.py#L64
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

        # Ordering used in ig65m and torchvision
        # https://github.com/pytorch/vision/tree/master/torchvision/models/video
        self.final_rearrange = Rearrange("t c h w -> c t h w")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]

        # For now use this private method since it allows resizing
        # on the ffmpeg side which is faster
        range_upper_bound = self.nframes // 15  # time if it was 15fps
        vid, _, _ = torchvision.io._read_video_from_file(
            video_path,
            video_width=0,
            video_height=self.video_read_height,
            read_audio_stream=False,
            video_timebase=Fraction(1),
            video_pts_range=(0, range_upper_bound),
        )

        vid = vid[: self.nframes]
        length = vid.size(0)

        if length > 0:
            vid = self.tfms(vid)
        else:
            vid = vid.float()

        if length < self.min_nframes:
            # Padding
            out_tensor = vid.new_full(
                (self.min_nframes, 3, self.crop_size, self.crop_size), 0.0
            )
            if length == 0:
                print("Zero length video!", video_path)
            else:
                out_tensor[:length, ...] = vid
            vid = out_tensor

        vid = self.final_rearrange(vid)

        return vid, {}


def sample_instance(feature_list, sampling):
    """Sample tensor from a list

    Args:
        features_list: List of 1D tensors
        sampling: One of ``['all', 'first', 'random']``

    Returns: Depending on ``sampling``:
        - 'all': [list_len, embedding_size] tensor containg all the
            embeddings stacked
        - 'first': [embedding_size] shape tensor of the first element
        - 'random': [embedding_size] shape tensor of a random element
    """
    assert isinstance(feature_list, list)
    if sampling == "first":
        return feature_list[0]
    elif sampling == "random":
        ri = torch.randint(0, len(feature_list), ())
        return feature_list[ri]
    elif sampling == "all":
        # NB this won't work when doing batching since
        # tensor sizes will vary
        return torch.stack(feature_list)
    else:
        raise Exception("Unknown sampling method")


def sample_if_list(feature_tensor_or_list, sampling):
    """Convenience function that will return a tensor as-is
    but if given a list of tensors will return one given by the
    sampling method.

    Args:
        feature_tensor_or_list: Tensor or list of tensors
        sampling: One of ``['all', 'first', 'random']``
        - all
    """
    if isinstance(feature_tensor_or_list, list):
        return sample_instance(feature_tensor_or_list, sampling)
    elif torch.is_tensor(feature_tensor_or_list):
        return feature_tensor_or_list


class FeaturesDataset(Dataset):
    """Load precomputed reddit features

    Args:
        csv_file: A csv file of reddit posts, which should at minimum have
        "reddit_id" and "video_path" columns. Returned features will be ordered
        as given in this file.

        input_features: Specification of the input features, with possible forms:
            - "filename.pth" : Load a single file
            - ["file1.pth", "file2.pth", ...] : Load multiple files, each
                will be a separate returned input from __getitem__
            - ["a.pth", ["b.pth", "c.pth"], ...] etc : Features given in nested
                lists will be concatenated into one long feature, eg
                feature_a, feature_bc

            Files are in pytorch format, containing a dict
                {"reddit_ids": (torch.int64, shape N)
                 "embeddings": (torch.float32, shape N x embedding_size)}
            Or for comments (multiple comments per reddit id):
                {"reddit_id_to_comment_id": (dict[int, List[str]])
                 "embeddings": (List[List[torch.float32]])}

        target_features: .pth file of target features (to be used in loss
            function but not passed to network)

        train (bool): Determines how csv is partitioned (see partition_dataframe)
            True for training set, False validation set

        train_comment_sampling (string):
            One of ``['all', 'first', 'random']`` specifying how to sample comments
            at train time, since there are multiple comments per reddit post.

            For the different options the comment embeddings returned by __getitem__ are as follows:

            - 'all': [n_comments, embedding_size] tensor containg embeddings
                of all the captions for the video
            - 'first': [embedding_size] shape tensor of the first comment
            - 'random': [embedding_size] shape tensor of a random comment

        test_comment_sampling (string):
            As above but at test time

    """

    def __init__(
        self,
        csv_file,
        input_features=None,
        target_features=None,
        train=True,
        train_comment_sampling=None,
        test_comment_sampling=None,
    ):
        self.train = train

        self.feature_sampling = (
            train_comment_sampling if train else test_comment_sampling
        )

        df = pd.read_csv(csv_file)
        df = partition_dataframe(df, split="train" if train else "val")

        # allow string or list of string to load multiple input features
        if isinstance(input_features, str):
            input_features = [input_features]

        # Allow up to one level of nesting
        self.feats = [
            (
                [load_features(df, feats_inner) for feats_inner in feats]
                if isinstance(feats, list)
                else load_features(df, feats)
            )
            for feats in input_features
        ]

        self.targets = None
        if target_features:
            self.targets = load_features(df, target_features)

    def __len__(self):
        return len(self.feats[0])

    def __getitem__(self, idx):
        input = []
        for feat in self.feats:
            if isinstance(feat, list):
                # If feat is a list concatenate the features
                input.append(
                    torch.cat(
                        [sample_if_list(f[idx], self.feature_sampling) for f in feat]
                    )
                )
            else:
                input.append(sample_if_list(feat[idx], self.feature_sampling))

        meta = {}
        if self.targets is not None:
            meta["target"] = self.targets[idx]
        return (*input, meta)


class ImTextDataset(VisionTitleCommentDatasetBase):
    """Load thumbnail images, titles, and comments with CLIP preprocessing.
    TODO: add option for other preprocessing.
    Args:
        csv_file (str): A csv file of reddit posts
        root (str): root directory prefix for the images and videos data
        train (bool): determines how csv is partitioned (see partition_dataframe)
            True for training set, False validation set
        add_comments (str): ['always', 'train_only', 'never']
        num_comms (int): number of comments per post to add when adding comments
        comment_sampling (str or None): if set to "random" it will sample
            random comments per post
    """

    def __init__(
        self,
        csv_file,
        root,
        train=True,
        test=False,
        add_comments="train_only",
        num_comms=0,
        comment_sampling="random",
        cached_vision_features=None,
        test_on_over_k_comms=None,
        test_set_limit=None,
        use_augmentation=False,
        cached_audio_features=None,
        audio_with_comms=None,
        audio_instead_of_title=False,
    ):
        self.train = train
        self.root = root
        self.num_comms = int(num_comms)
        self.comment_sampling = comment_sampling if train else None
        self.cached_vision_features = cached_vision_features
        self.test_on_over_k_comms = test_on_over_k_comms
        self.test_set_limit = test_set_limit
        self.use_augmentation = use_augmentation
        self.cached_audio_features = cached_audio_features
        self.audio_with_comms = audio_with_comms
        self.audio_instead_of_title = audio_instead_of_title

        self.add_comments = self.should_add_comments(add_comments, train)

        self.ids = []
        self.filenames = []
        self.titles = []
        self.video_lengths = []
        self.comments = []

        df = pd.read_csv(csv_file)
        df = self.split_dataset(
            csv_file,
            df,
            train,
            test,
            test_on_over_k_comms=test_on_over_k_comms,
            test_set_limit=test_set_limit,
        )
        self._load_reddit(df, file_extension=".jpg")

        self.preprocess = CLIP_TRANSFORM
        self.tokenizer = SimpleTokenizer()
        self.rake = Rake()

        if cached_vision_features is not None:
            self.vision_feats = load_features(df, cached_vision_features)
        if cached_audio_features is not None:
            self.audio_feats = load_features(df, cached_audio_features)

        if self.train:
            self.img_tfm = IMG_AUG
        else:
            self.img_tfm = transforms.Compose([])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        im_path = self.filenames[idx]
        title = self.titles[idx]
        id = self.ids[idx]
        comments = self.comments[idx]

        if self.cached_vision_features is not None:
            im = self.vision_feats[idx]
        else:
            im = Image.open(im_path).convert("RGB")
            if self.use_augmentation:
                im = self.img_tfm(im)
            im = self.preprocess(im)

        title_tok = self._tokenise([title])[0]

        if self.add_comments:
            comments = self.preprocess_comments(
                comments, sampling=self.comment_sampling, num_comms=self.num_comms
            )
            comments_tok = self._tokenise(comments)
            if torch.rand([]) < 0.0001:
                print(
                    "Debug dataloader -- title:",
                    title,
                    "comms:",
                    comments,
                )
        else:
            comments_tok = self._tokenise([""])

        if self.cached_audio_features:
            audio_clips = self.audio_feats[idx]
            if self.audio_instead_of_title:
                input = (im, audio_clips)
            elif self.audio_with_comms:
                input = (im, title_tok, (comments_tok, audio_clips))
            else:
                input = (im, title_tok, audio_clips)
        else:
            input = (im, title_tok, comments_tok)
        meta = {"id": id}

        return (*input, meta)


class VideoDatasetReddit(VideoDatasetSegments):
    def __init__(
        self,
        root,
        reddit_csv,
        train=False,
        split="test",
        num_comms=5,
        test_on_over_k_comms=3,
        test_set_limit=5000,
        comment_sampling=None,
        first_frame_only=False,
    ):
        # This loader is only intended for testing
        assert train is False
        assert split == "test"
        self.preprocess = CLIP_TRANSFORM

        super().__init__(
            csv_file=reddit_csv,
            root=root,
            train=train,
            test=True,
            add_comments="always" if num_comms != 0 else "train_only",
            num_comms=num_comms,
            comment_sampling=comment_sampling,
            first_frame_only=first_frame_only,
            test_on_over_k_comms=test_on_over_k_comms,
            test_set_limit=test_set_limit,
        )

    def __getitem__(self, index):
        video_path = self.filenames[index]
        title = self.titles[index]
        comments = self.comments[index]
        vid_id = self.ids[index]

        vid, _, _ = torchvision.io._read_video_from_file(
            video_path, read_audio_stream=False
        )
        if vid.shape[0] == 0:
            print(f"Failed reading: {video_path}")
            vid = torch.zeros(8, 300, 300, 3, dtype=torch.uint8)

        images = []
        for frame in vid[:8]:
            images.append(
                self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
            )

        vid = torch.stack(images)
        title_tok = self._tokenise(title)
        pp_comments = self.preprocess_comments(
            comments, sampling=self.comment_sampling, num_comms=self.num_comms
        )
        comments_tok = self._tokenise(pp_comments)
        if vid.shape[0] != 8:
            vid_padding = torch.zeros(
                8 - vid.shape[0], *vid.shape[1:], dtype=torch.uint8
            )
            vid = torch.cat((vid, vid_padding), axis=0)
        return vid, title_tok, comments_tok, vid_id

    def __len__(self):
        return len(self.filenames)


class VideoDatasetLivebot(VideoDatasetSegments):
    def __init__(
        self,
        root,
        cvs_file,
        train=False,
        split="test",
        add_comments=True,
    ):
        # This loader is only intended for testing
        assert train is False
        assert split == "test"

        df = pd.read_csv(cvs_file)
        self.video_files = []
        self.titles = []
        self.comments = []
        self.preprocess = CLIP_TRANSFORM
        self.add_comments = add_comments

        for _, row in df.iterrows():
            self.video_files.append(os.path.join(root, row.video_path))
            self.titles.append(row.title)
            self.comments.append(ast.literal_eval(row.comments))

        print(len(self.video_files), "comments test files")

    def __getitem__(self, index):
        video_path = self.video_files[index]
        title = self.titles[index]
        comments = self.comments[index]
        vid_id = video_path.split("/")[-1].split(".")[0]
        try:
            vid, _, _ = torchvision.io._read_video_from_file(
                video_path, read_audio_stream=False
            )
        except Exception as e:
            print("failed video: ", video_path, e)
            vid = None
        if len(vid) == 0:
            vid = None
        if vid is not None:
            images = []
            for frame in vid:
                images.append(
                    self.preprocess(Image.fromarray(frame.numpy()).convert("RGB"))
                )
            vid = torch.stack(images)

        title_tok = _tokenize_max_len(title)
        if self.add_comments:
            comments_tok = _tokenize_max_len(comments)
        else:
            comments_tok = _tokenize_max_len([""])

        return vid, title_tok, comments_tok, vid_id

    def __len__(self):
        return len(self.video_files)
