import argparse
import collections
import os
import random

import torch

# Limit the number of threads used in CPU ops
# (mainly for tensor ops in data loaders which already have multiproc
# parallelism and then try to use #ncores threads too)
torch.set_num_threads(35)

# Make sure we don't run out of file descriptors
# (possibly the cause of some deadlocks)
# https://github.com/pytorch/pytorch/issues/1355#issuecomment-819203114
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import numpy as np
import wandb
from torch.utils.data import DataLoader

import dataset_loaders.dataset_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import prepare_device
from utils.parse_config import ConfigParser


def main(config: ConfigParser):
    seed_value = int(config.get("random_seed_value", 1023))
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = config.get_logger("train")

    # setup data_loader instances
    dataset = config.init_obj("dataset", module_data)
    valid_dataset = config.init_obj("dataset", module_data, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    wandb.watch(model, log_freq=100)

    logger.info(model)

    # prepare for (multi-device) GPU training
    gpu_ids = (
        [int(s) for s in _args.device.split(",")] if _args.device is not None else []
    )
    device, device_ids = prepare_device(config["n_gpu"], gpu_ids=gpu_ids)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_ids, output_device=device_ids[1]
        )
    else:
        model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [
        getattr(module_metric, met["type"])(**dict(met["args"]))
        for met in config["metrics"]
    ]

    # Allow setting different learning rates for different parts of the network,
    # based on parameter names matching certain strings

    fc_lr = config.get("fc_lr", None)
    time_lr = config.get("time_lr", None)
    adapter_lr = config.get("adapter_lr", None)

    # The final linear layers of CLIP (exact name match)
    clip_final_linear = ["model.text_projection", "model.visual.proj"]

    # Temporal parameters for TimeSformer
    time_layers = ["time", "temporal"]

    # The parameters related to our final transformer
    final_adapter_layers = ["final_transformer.", "final_linear.", "mask_embedding"]

    # Params containing these strings should not have weight decay
    nodecay = ["bias", ".ln", "embedding", "temporal_embed"]

    npara = len(list(filter(lambda p: p.requires_grad, model.parameters())))

    trainable_params_clip_final_linear = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and n in clip_final_linear
    ]

    trainable_params_time = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and any(t in n for t in time_layers)
    ]

    trainable_params_final_adapter = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and any(t in n for t in final_adapter_layers)
    ]

    trainable_params_rest = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad
        and p
        not in set(
            x[1]
            for x in (
                trainable_params_clip_final_linear
                + trainable_params_time
                + trainable_params_final_adapter
            )
        )
    ]

    assert (
        len(trainable_params_clip_final_linear)
        + len(trainable_params_time)
        + len(trainable_params_rest)
        + len(trainable_params_final_adapter)
    ) == npara

    def makeparamdicts(named_params, lr, no_decay_matches):
        """
        Convert a list of (name,parameter) tuples into a list containing a pair
        of dictionaries in the format expected by torch optimizers. The first dictionary
        contains the parameters that may have weight decay, the second those that should
        not have weight decay (setting "weight_decay": 0), which is determined
        if a parameter name contains any substring given in no_decay_matches.
        If lr is not None, it will also be added to the dictionaries.
        """
        paras_decay = [
            p for n, p in named_params if all(t not in n for t in no_decay_matches)
        ]
        paras_nodecay = [
            p for n, p in named_params if any(t in n for t in no_decay_matches)
        ]
        assert len(paras_decay) + len(paras_nodecay) == len(named_params)

        dicts = []

        d1 = {"params": paras_decay}
        if lr is not None:
            d1["lr"] = lr
        if len(paras_decay):
            dicts.append(d1)

        d2 = {"params": paras_nodecay, "weight_decay": 0.0}
        if lr is not None:
            d2["lr"] = lr
        if len(paras_nodecay):
            dicts.append(d2)

        return dicts

    optimizer = config.init_obj(
        "optimizer",
        torch.optim,
        makeparamdicts(trainable_params_rest, None, nodecay)
        + makeparamdicts(trainable_params_final_adapter, adapter_lr, nodecay)
        + makeparamdicts(trainable_params_clip_final_linear, fc_lr, nodecay)
        + makeparamdicts(trainable_params_time, time_lr, nodecay),
    )

    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--fc_lr"], type=float, target="fc_lr"),
        CustomArgs(["--time_lr"], type=float, target="time_lr"),
        CustomArgs(["--adapter_lr"], type=float, target="adapter_lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="batch_size"),
        CustomArgs(["--n_gpu"], type=int, target="n_gpu"),
        CustomArgs(
            ["--b", "--branch_to_adapt"], type=str, target="arch;args;branch_to_adapt"
        ),
        CustomArgs(
            ["--bv", "--branch_to_adapt_val"],
            type=str,
            target="arch;args;branch_to_adapt_val",
        ),
        CustomArgs(["--nc", "--num_comms"], type=int, target="dataset;args;num_comms"),
        CustomArgs(
            ["--nl", "--num_imlabels"], type=int, target="dataset;args;num_imlabels"
        ),
        CustomArgs(
            ["--cached_vision_features"],
            type=str,
            target="dataset;args;cached_vision_features",
        ),
        CustomArgs(["--add_comments"], type=str, target="dataset;args;add_comments"),
        CustomArgs(["--e", "--exp_name"], type=str, target="name"),
        CustomArgs(["--freeze"], type=str, target="arch;args;freeze"),
        CustomArgs(
            ["--residual_activation"], type=str, target="arch;args;residual_activation"
        ),
        CustomArgs(["--comment_fusion"], type=str, target="arch;args;comment_fusion"),
        CustomArgs(["--save_dir"], type=str, target="trainer;save_dir"),
        CustomArgs(["--epochs"], type=int, target="trainer;epochs"),
        CustomArgs(["--visual_device"], type=str, target="arch;args;visual_device"),
        CustomArgs(["--random_seed_value"], type=int, target="random_seed_value"),
    ]
    config = ConfigParser.from_args(args, options)
    _args = args.parse_args()

    wandb.init(config=_args)
    wandb.run.name = config["name"]
    wandb.run.save()

    main(config)
