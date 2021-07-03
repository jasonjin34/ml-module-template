from ml_module_template.common.utils import register
from torch import nn


@register('loss', 'BCE')
def build_BCE(cfg):
    reduction = cfg.get('reduction', 'none')
    return nn.BCELoss(reduction=reduction)

@register('loss', 'BCEWithLogits')
def build_BCE(cfg):
    reduction = cfg.get('reduction', 'none')
    return nn.BCEWithLogitsLoss(reduction=reduction)
