import segmentation_models_pytorch as smp
from ml_module_template.common.utils import register


@register('model', 'FPN')
def build(cfg):
    encoder_name = cfg.get('encoder_name', 'resnet34')
    encoder_weights = cfg.get('encoder_weights', 'imagenet')
    in_channels = cfg.get('in_channels', 3)
    classes = cfg.get('num_classes', 1)
    activation = cfg.get('activation', 'sigmoid')

    return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
    )


@register('model', 'UnetV2')
def build(cfg):
    encoder_name = cfg.get('encoder_name', 'resnet34')
    encoder_weights = cfg.get('encoder_weights', 'imagenet')
    in_channels = cfg.get('in_channels', 3)
    classes = cfg.get('num_classes', 1)
    activation = cfg.get('activation', 'sigmoid')

    return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
    )
