import torch
import torch.nn.functional as F


def binary_cross_entropy(input, meta):
    target = meta["target"].reshape(input.shape)
    return F.binary_cross_entropy_with_logits(input, target)


def cross_entropy(input, meta):
    return F.cross_entropy(input, meta["target"])


def mse_loss(input, meta, reduction="mean"):
    return F.mse_loss(input, meta["target"], reduction=reduction)


def clip_loss(input, meta):
    sim = input[2]
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
    return loss
