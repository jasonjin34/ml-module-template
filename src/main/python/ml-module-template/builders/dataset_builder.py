from ml_module_template.common.utils import get_registered
from ml_module_template.builders import transform_builder


def build(cfg):
    name = cfg['name']
    dataset_fn = get_registered('dataset', name)

    if 'transform' in cfg:
        transform = transform_builder.build(cfg.transform)
    else:
        transform = None

    return dataset_fn(cfg, transform)
