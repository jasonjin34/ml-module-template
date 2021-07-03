from torch import optim
from ignite.contrib.handlers import param_scheduler


def build(optimizer, cfg):
    parameter = cfg.get('parameter', 'lr')
    if cfg.name.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    elif cfg.name.lower() == 'PiecewiseLinear'.lower():
        milestones = cfg.milestones
        tmilestones = []
        for m in milestones:
            tmilestones.append(tuple(m))
        # requires tuple
        scheduler = param_scheduler.PiecewiseLinear(optimizer, parameter, tmilestones)
    else:
        raise ValueError()

    # Set initial learning rate
    v = scheduler.get_param()
    for param_group in optimizer.param_groups:
        param_group[parameter] = v
    return scheduler
