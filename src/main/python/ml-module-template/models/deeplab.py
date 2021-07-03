from ml_module_template.common.utils import register
from torchvision import models
from torch import nn

class Wrapper(nn.Module):
    """
    Wrapper around torchvision model.

    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output['out']


@register('model', 'deeplabv3')
def build(cfg):
    backbone = cfg.get('backbone', 'resnet50')
    num_classes = cfg.get('num_classes', 1)
    progress = cfg.get('progress', False)
    pretrained = cfg.get('pretrained', True)

    if backbone == 'resnet50':
        build_fn = models.segmentation.deeplabv3_resnet50
    elif backbone == 'resnet101':
        build_fn = models.segmentation.deeplabv3_resnet101
    else:
        raise ValueError("Unknown backbone %s" % backbone)

    model = build_fn(pretrained=pretrained, progress=progress)
    # init from scratch
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    return Wrapper(model)
