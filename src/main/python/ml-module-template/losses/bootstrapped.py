import functools
import operator
from torch import nn
import torch


def prod(iterator):
    return functools.reduce(operator.mul, iterator, 1)


def get_topk_percent(tensor, top_k_percent_pixels):
    """
    Returns the top_k pixels of a tensor.
    Similar to
    https://github.com/tensorflow/models/blob/master/research/deeplab/utils/train_utils.py
    Args:
        tensor: At least 2D.
        top_k_percent_pixels (float): percent of pixels we want to return (between 0 and 1)
    """
    assert len(tensor.shape) >= 2
    num_pixels = prod(tensor[0].shape)
    top_k_pixels = int(top_k_percent_pixels * num_pixels)
    tensor = tensor.view(tensor.shape[0], -1)
    return torch.topk(tensor, top_k_pixels)


class BootstrappedLoss(nn.Module):
    """
    Makes a bootstrapped loss from any loss.

    Bootstrapping that only the top x percent that contribute to the loss
    are used.

    This is a form of hard mining.
    """
    def __init__(self, loss_fn, top_k_percent_pixels):
        super().__init__()
        self.loss_fn = loss_fn
        # this is usually used for an optimizer that can have multiple parameter groups
        self.group = {'top_k_percent_pixels': top_k_percent_pixels}
        self.param_groups = [self.group]

    def forward(self, *args):
        loss = self.loss_fn(*args)
        top_k_percent = self.group['top_k_percent_pixels']
        if top_k_percent < 1.0:
            loss, indices = get_topk_percent(loss, top_k_percent)
        # TODO mirror reduction
        return loss.mean()


def build(cfg, loss_fn):
    return BootstrappedLoss(loss_fn, cfg.get('top_k_percent_pixels', 0.25))
