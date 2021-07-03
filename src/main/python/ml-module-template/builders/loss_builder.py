from ml_module_template.common.utils import get_registered
from ml_module_template.losses import bootstrapped


def build(cfg):
    name = cfg['name']
    loss_fn = get_registered('loss', name)(cfg)
    if 'bootstrapped' in cfg:
        return bootstrapped.build(cfg, loss_fn)
    return loss_fn
