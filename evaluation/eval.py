import argparse
import collections
import sys

sys.path.append(".")

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_loaders.dataset_loaders as module_data
import model.model as module_arch
from model.metric import RecallAtK
from utils.parse_config import ConfigParser

logging.getLogger().setLevel(logging.INFO)


def add_irrelevant_comms(
    comments: torch.tensor, num_irrelevant_comments: int
) -> torch.tensor:
    """Adds irrelevant comments by randomly selecting other comments from the batch.
    All additional comments come from different elements in the batch.
    """
    bs = comments.shape[0]
    total_comm_len = comments.shape[1] + num_irrelevant_comments
    updated_comments = torch.zeros((bs, total_comm_len, 77))
    for i in range(len(comments) - 1):
        new_comm_list = []
        # select random indices for each irrelevant comment to be selected from the batch
        comm_indices = np.random.randint(
            low=0, high=comments.shape[1], size=num_irrelevant_comments
        )
        for comm_ind in comm_indices:
            # select random index from the batch
            batch_ind = np.random.randint(low=0, high=bs, size=[1])
            if batch_ind != i:
                new_comm_list.append(comments[batch_ind][:, comm_ind, :])
            else:
                batch_ind = np.random.randint(low=0, high=bs, size=[1])
                new_comm_list.append(comments[batch_ind][:, comm_ind, :])
        updated_comments[i] = torch.cat([comments[i], new_comm_list], 0)
        return updated_comments.long()


def main(
    config: ConfigParser,
    args: argparse.Namespace,
    checkpoint_path: str,
    device: str = "cuda",
):
    logger = config.get_logger("test")

    dataset = config.init_obj("dataset", module_data, train=False, test=True)

    branch_to_adapt = config["arch"]["args"].get("branch_to_adapt_val", None)
    comment_fusion = config["arch"]["args"].get("comment_fusion", None)
    num_comms = config["dataset"]["args"].get("num_comms", None)
    add_comments = config["dataset"]["args"]["add_comments"]
    num_irrelevant_comments = args.num_irrelevant_comments

    if branch_to_adapt is None:
        if add_comments != "always":
            exp_combo = "title_only"
        else:
            exp_combo = f"{comment_fusion}_{num_comms}_comms"
    else:
        exp_combo = f"adapted_{branch_to_adapt}_{num_comms}_comms"

    if checkpoint_path is not None:
        save_path = f"{checkpoint_path.absolute().as_posix()[:-4]}_res_{exp_combo}.json"
    else:
        save_path = f"zero_shot_res_{comment_fusion}.json"
    logging.info(f"Saving results to {save_path}")

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=10,
        shuffle=False,
    )

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logger.info(model)

    model = model.to(device)

    res_vis = []
    res_text = []
    ids = []
    for items in tqdm(data_loader):
        vis, title, comments, meta = items
        with torch.no_grad():
            if num_irrelevant_comments:
                assert (
                    num_irrelevant_comments <= config["batch_size"]
                ), "Number of irrelevant comments needs to be smaller than batch size."
                comments = add_irrelevant_comms(comments, num_irrelevant_comments)
            feats_vis, feats_text, sim = model.forward(
                torch.squeeze(vis).to(device),
                torch.squeeze(title).to(device),
                comments.to(device),
            )
        res_vis.extend(feats_vis.cpu().detach().numpy())
        res_text.extend(feats_text.cpu().detach().numpy())
        ids.extend(meta["id"].cpu().detach().numpy())

    res_vis = np.stack(res_vis)
    res_text = np.stack(res_text)

    recall_all_k_title_from_im = RecallAtK("images", "titles", [1, 5, 10]).compute(
        res_vis, res_text
    )
    recall_all_k_im_from_title = RecallAtK("titles", "images", [1, 5, 10]).compute(
        res_text, res_vis
    )

    logging.info("Recall im from title: ", recall_all_k_im_from_title)
    logging.info("Recall title from im: ", recall_all_k_title_from_im)

    out = {
        "R1_title_from_im": recall_all_k_title_from_im[0][1],
        "R5_title_from_im": recall_all_k_title_from_im[1][1],
        "R10_title_from_im": recall_all_k_title_from_im[2][1],
        "R1_im_from_title": recall_all_k_im_from_title[0][1],
        "R5_im_from_title": recall_all_k_im_from_title[1][1],
        "R10_im_from_title": recall_all_k_im_from_title[2][1],
    }

    with open(save_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="configs/pretrained_clip.jsonc",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default="3",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "--num_irrelevant_comments",
        default=0,
        type=int,
    )
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="batch_size",
        ),
        CustomArgs(
            ["--bv", "--branch_to_adapt_val"],
            type=str,
            target="arch;args;branch_to_adapt_val",
        ),
        CustomArgs(["--nc", "--num_comms"], type=str, target="dataset;args;num_comms"),
        CustomArgs(
            ["--am", "--comment_fusion"], type=str, target="arch;args;comment_fusion"
        ),
        CustomArgs(
            ["--ac", "--add_comments"], type=str, target="dataset;args;add_comments"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()

    main(config, args, config.resume, device="cuda:" + args.device)
