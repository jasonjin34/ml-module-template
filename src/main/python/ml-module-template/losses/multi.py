from torch import nn
from ml_module_template.common.utils import register
from ml_module_template.builders import loss_builder


class MultiLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses


class WeightedMultiLoss(MultiLoss):
    def __init__(self, losses):
        super().__init__(losses)

    def forward(self, y_pred, y):
        loss = 0.0
        for ll in self.losses:
            loss += ll(y_pred, y)
        return loss


@register('loss', 'multi_loss')
def build_weighted_multi_loss(cfg):
    losses = []
    for loss_cfg in cfg['losses']:
        loss = loss_builder.build(loss_cfg)
        losses.append(loss)
    return WeightedMultiLoss(losses)
