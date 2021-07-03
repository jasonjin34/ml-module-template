from torch import optim


def build(params, cfg):
    cfg = cfg.copy()
    name = cfg.pop('name').lower()

    if name == "adam":
        optimizer = optim.Adam
    elif name == "SGD":
        optimizer = optim.SGD

    return optimizer(params, **cfg)
