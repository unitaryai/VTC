import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import pyjson5
import torch


def is_image_like_batch(obj):
    """ Return True if it seems to be a batch of RGB images """
    return torch.is_tensor(obj) and len(obj.shape) == 4 and obj.shape[1] == 3


def move_to(obj, device):
    """Transfer to gpu for nested objects
    from https://discuss.pytorch.org/t/66283/2"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def extract_tensors(obj):
    """ Get list of tensor objects from nested data structure """
    tensors = []

    def visit(obj):
        if torch.is_tensor(obj):
            tensors.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
        else:
            raise TypeError("Invalid type for move_to")

    visit(obj)
    return tensors


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return pyjson5.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use, gpu_ids=[0]):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu >= n_gpu_use:
        assert isinstance(gpu_ids, list) and len(gpu_ids) == n_gpu_use
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device(f"cuda:{gpu_ids[0]}" if n_gpu_use > 0 else "cpu")
    return device, gpu_ids
