from ml_module_template.common.utils import get_registered
import torch
import logging

logger = logging.getLogger(__name__)


def build(cfg, device=None):
    name = cfg['name']
    model_fn = get_registered('model', name)
    model = model_fn(cfg)
    if 'weights' in cfg and cfg['weights'] is not None:
        # loads to cpu always
        weights = torch.load(cfg.weights, map_location=lambda storage, loc: storage)
        if 'model' in weights:
            weights = weights['model']
        model.load_state_dict(weights)
        logger.info("Restored weights from %s", cfg.weights)
    model = model.to(device)
    return model
