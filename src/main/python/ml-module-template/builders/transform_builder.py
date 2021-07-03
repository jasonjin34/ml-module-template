from imgaug import augmenters as iaa

def build(cfg):
    # test does not need size
    size = cfg.get('size')
    if cfg.name == "test":
        seq = iaa.Sequential([
            iaa.CenterCropToFixedSize(size, size)
        ])
    elif cfg.name == "pretrain":
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of the image
            # only shifting because we are using smaller image than available
            iaa.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            ),
            iaa.CenterCropToFixedSize(size, size),
            iaa.PadToFixedSize(size, size),

        ])
    elif cfg.name == "train":
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of the image
            iaa.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            ),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.CenterCropToFixedSize(size, size),
            iaa.PadToFixedSize(size, size),
        ])
    elif cfg.name == "trainv2":
        clouds = iaa.CloudLayer(
            intensity_mean=(180, 190),
            intensity_freq_exponent=(-2.5, -2.0),
            intensity_coarse_scale=10,
            alpha_min=0,
            alpha_multiplier=(0.25, 0.75),
            alpha_size_px_max=(2, 8),
            alpha_freq_exponent=(-2.5, -2.0),
            sparsity=(0.8, 1.0),
            density_multiplier=(0.5, 1.0)
        )
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5), # horizontally flip 50% of the image
                iaa.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                ),
                iaa.CenterCropToFixedSize(size, size),
                iaa.PadToFixedSize(size, size),
                iaa.AddToSaturation(value=15),
                iaa.MultiplyHueAndSaturation(mul_saturation=(0.9, 1.1), per_channel=False),
                iaa.MultiplyBrightness((0.90, 1.1)),
                iaa.Sometimes(0.1, iaa.MotionBlur()),
                iaa.Sometimes(0.05, clouds),
            ],
        )
    elif cfg.name == "testv2":
        augs = [
            iaa.AddToSaturation(value=15)
        ]

        if size is not None:
            augs.append(iaa.CenterCropToFixedSize(size, size))
        seq = iaa.Sequential(augs)
    else:
        raise ValueError("Unknown transform %s" % cfg.name)
    return seq
