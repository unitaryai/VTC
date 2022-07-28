import argparse
import logging
import sys

sys.path.append(".")

from typing import Optional

import clip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_loaders.dataset_loaders as module_data
import model.model as module_arch
from model.metric import RecallAtK

logging.getLogger().setLevel("INFO")


def compute_recall(
    tensor_v: torch.tensor,
    tensor_t: torch.tensor,
    split: str = "full-test",
    dataset_name: str = "MSRVTT",
) -> pd.DataFrame:
    recall_range = [1, 5, 10]
    recall_all_k_title_from_im = RecallAtK("videos", "titles", recall_range).compute(
        tensor_v.numpy(), tensor_t.numpy().squeeze()
    )
    recall_all_k_im_from_title = RecallAtK("titles", "videos", recall_range).compute(
        tensor_t.numpy().squeeze(), tensor_v.numpy()
    )
    vtr = np.array(recall_all_k_title_from_im)[:, 1] * 100.0
    tvr = np.array(recall_all_k_im_from_title)[:, 1] * 100.0
    recall_range_index = [f"R@{i}" for i in recall_range]
    df = pd.DataFrame(
        {
            f"{dataset_name} {split} split Video to Text": tvr,
            f"{dataset_name} {split} split Text to Video": vtr,
        },
        index=recall_range_index,
    )
    logging.info(df)
    return df


models_needing_comments = (
    module_arch.PretrainedCLIP_finaltf,
    module_arch.PretrainedCLIP_TimeSformer_finaltf,
)

image_models = (module_arch.PretrainedCLIP, module_arch.PretrainedCLIP_finaltf)

video_models = (
    module_arch.PretrainedCLIP,
    module_arch.PretrainedCLIP_finaltf,
    module_arch.PretrainedCLIP_TimeSformer,
    module_arch.PretrainedCLIP_TimeSformer_finaltf,
)


def load_model(checkpoint_path: str, device: str, model_type: str):
    # TODO switch to load from config

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        init_from_avg = checkpoint["config"]["arch"]["args"].get("init_from_avg", False)

    if model_type == "pretrained_clip":
        model = module_arch.PretrainedCLIP(
            model_type="ViT-B/32",
            freeze=False,
            residual_activation=args.residual_activation,
        )
    elif model_type == "clip_timesformer":
        model = module_arch.PretrainedCLIP_TimeSformer(
            residual_activation=args.residual_activation
        )
    elif model_type == "pretrained_clip_finaltf":
        print(
            f"PretrainedCLIP_finaltf: branch_to_adapt_val={args.branch_to_adapt} residual_activation={args.residual_activation} init_from_avg={init_from_avg}"
        )
        model = module_arch.PretrainedCLIP_finaltf(
            branch_to_adapt_val=args.branch_to_adapt,
            residual_activation=args.residual_activation,
            init_from_avg=init_from_avg,
        )
    elif model_type == "clip_timesformer_finaltf":
        print(
            f"PretrainedCLIP_TimeSformer_finaltf: branch_to_adapt_val={args.branch_to_adapt} residual_activation={args.residual_activation} init_from_avg={init_from_avg}"
        )
        model = module_arch.PretrainedCLIP_TimeSformer_finaltf(
            branch_to_adapt_val=args.branch_to_adapt,
            residual_activation=args.residual_activation,
            init_from_avg=init_from_avg,
        )

    if checkpoint_path is not None:
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def retrieval_evaluation(
    model: torch.nn.module,
    datasetname: str,
    split: str,
    device: str,
    out_csv: Optional[str] = None,
    frame_stride: int = 16,
    first_frame_only: bool = False,
    first_chunk_only: bool = False,
):
    n_comments = 5
    if datasetname == "MSRVTT_videos":
        dataset = module_data.VideoDatasetMSRVTT(train=False, split=split)
    elif datasetname == "MSVD_videos":
        dataset = module_data.VideoDatasetMSVD(train=False, split=split)
    elif datasetname == "K700_videos":
        dataset = module_data.VideoDatasetK700Comments(train=False, split=split)
    elif datasetname == "Reddit_videos":
        dataset = module_data.VideoDatasetReddit(train=False, split=split)
    elif datasetname == "livebot":
        dataset = module_data.VideoDatasetLivebot(
            train=False,
            split=split,
        )
    else:
        raise Exception("Unknown dataset")

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
    )

    video_joint_embeddings = []
    caption_joint_embeddings = []

    logging.info("Computing joint embeddings")

    for items in tqdm(data_loader):
        if len(items) == 3:
            frames, captions, _ = items
            comments = None
        else:
            frames, captions, comments, _ = items
            comments = comments.to(device)

        frames = frames.to(device)
        captions = captions.to(device)

        assert len(captions.shape) == 3 and captions.shape[0] == 1
        assert len(frames.shape) == 5 and frames.shape[0] == 1 and frames.shape[2] == 3

        # [1, ncaptions, 77] --> [ncaptions, 77]
        captions = captions[0]

        if (
            isinstance(model, image_models) and not isinstance(model, video_models)
        ) or first_frame_only:
            # [1, nframes, nchans, h, w] -> [nframes, nchans, h, w]
            frames = frames[0]
            if first_frame_only:
                frames = frames[0:1]
            assert not first_chunk_only

        elif isinstance(model, video_models):
            # Split into batches of 8 frames
            # [1, nframes, nchans, h, w] -> [nchunks, 8, nchans, h, w]
            nframes = 8

            frames = frames[:, ::frame_stride]
            splits = torch.split(frames, nframes, 1)
            splits_pad = [
                x
                if x.shape[1] == nframes
                else torch.index_select(
                    x,
                    dim=1,
                    index=torch.floor(
                        torch.linspace(0, x.shape[1] - 1, nframes, device=device)
                    ).to(torch.int64),
                )
                for x in splits
            ]
            chunks = torch.cat(splits_pad, dim=0)
            if first_chunk_only:
                chunks = chunks[0:1]
            frames = chunks
            assert not first_frame_only
        else:
            raise Exception("Unknown model_type")

        if isinstance(model, models_needing_comments):
            # Still not sure what is best to do here
            # (mask-only comment, repeat title, skip adapting)
            # NB empty string will be replaced with the model's mask token
            if model.branch_to_adapt_val == "image":
                ncomms = len(frames)
            else:
                ncomms = len(captions)

            if comments is None:
                dummy_comments = torch.stack(
                    [clip.tokenize([""] * n_comments) for _ in range(ncomms)]
                ).to(device)
                comments = dummy_comments
            else:
                comments = comments[0, :n_comments]
                n_real_comments = comments.shape[0]
                pad_comments = False
                # Append dummy comments to make it up to 5
                if pad_comments:
                    n_dummy_needed = n_comments - n_real_comments
                else:
                    n_dummy_needed = 0
                comments = torch.cat(
                    (comments, clip.tokenize([""] * n_dummy_needed).to(device))
                )
                comments = torch.stack([comments for _ in range(ncomms)])

            feats_a, feats_b, sim = model.forward(frames, captions, comments)
        else:
            feats_a, feats_b, sim = model.forward(frames, captions)

        video_joint_embeddings.append(feats_a.cpu().detach())
        caption_joint_embeddings.append(feats_b.cpu().detach())

    # pad captions tensor for when there's a different # of captions per video
    max_length = max([s.shape[0] for s in caption_joint_embeddings])
    padded_caption_joint_embeddings = [
        torch.cat(
            [
                k,
                torch.full(
                    (max_length - k.shape[0], k.shape[1]),
                    float("-inf"),
                    device=k.device,
                ),
            ]
        )
        for k in caption_joint_embeddings
    ]
    # take average of frame features
    video_joint_tensor = torch.cat(
        [
            torch.mean(torch.tensor(k), dim=0, keepdim=True)
            for k in video_joint_embeddings
        ]
    )
    caption_joint_tensor = torch.stack(padded_caption_joint_embeddings)

    outdf = compute_recall(
        video_joint_tensor, caption_joint_tensor, split=split, dataset_name=datasetname
    )

    if out_csv is not None:
        outdf.to_csv(out_csv)
    return outdf


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--dataset",
        default="MSRVTT_videos",
        choices=[
            "MSRVTT_videos",
            "MSVD_videos",
            "K700_videos",
            "Reddit_videos",
            "livebot",
        ],
        type=str,
        help="dataset to load",
    )
    args.add_argument(
        "-r",
        "--checkpoint",
        default=None,
        type=str,
        help="path to checkpoint (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_type",
        default=None,
        type=str,
        help="model arch to be loaded",
    )
    args.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="device to load model on",
    )
    args.add_argument(
        "-s",
        "--split",
        default="full-test",
        type=str,
        help="which test split to use",
    )
    args.add_argument(
        "--branch_to_adapt",
        default="text",
        choices=["text", "image", "random", "skip"],
        type=str,
        help="which branch to adapt for finaltf models",
    )
    args.add_argument(
        "--residual_activation",
        default="none",
        type=str,
        help="which activation fn to use on the residual",
    )
    args.add_argument(
        "--out_csv",
        default=None,
        type=str,
        help="File to save output csv",
    )
    args.add_argument(
        "--frame_stride",
        default=16,
        type=int,
        help="Video frame stride",
    )
    args.add_argument(
        "--first_frame_only",
        action="store_true",
        help="Use only the first frame of a video, as if it were an image",
    )
    args.add_argument(
        "--first_chunk_only",
        action="store_true",
        help="Use only the first 8-frame chunk of a video",
    )
    args = args.parse_args()

    model = load_model(args.checkpoint, args.device, model_type=args.model_type)

    retrieval_evaluation(
        model,
        args.dataset,
        args.split,
        args.device,
        out_csv=args.out_csv,
        frame_stride=args.frame_stride,
        first_frame_only=args.first_frame_only,
        first_chunk_only=args.first_chunk_only,
    )
