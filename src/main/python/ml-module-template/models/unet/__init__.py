from .unet_model import UNet
from ml_module_template.common.utils import register


@register('model', 'unet')
def build(cfg):
    n_channels = cfg.get('n_channels', 3)

    # backwards compability
    if 'num_classes' in cfg:
        num_classes = cfg.num_classes
    elif 'n_classes' in cfg:
        num_classes = cfg.n_classes
    else:
        raise ValueError("Needs n_classes or num_classes")
    return UNet(n_channels, num_classes)
