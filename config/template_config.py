PATH = "{Dataset path}"
MULTIPLIER = 400

cfg = {
    "model": {
        "name": "fpn",
        "num_classes": 1,
        "encoder_name": "resnet-34"
        "pretrained": "True",
        "activation": None,
        #"weights": "Pretain weight"
    },
    "training": {
        "dataset": {
            "transform": {
                "name": "trainv2",
                "size": 320
            },
            "path": PATH,
            "span": [0.0, 0.8]
        },
        "optimizer": {
            "name": "adam"
        },
        "scheduler": {
            "name": "PiecewiseLinear",
            "parameter": "lr",
            "milestones": [(0, 0.0),
                           (4 * MULTIPLIER, 0.002),
                           (8 * MULTIPLIER, 0.001),
                           (15 * MULTIPLIER, 0.0003),
                           (20 * MULTIPLIER, 0.0001)]
        },
        "loss": {
            "name": "soft_dice",
            "do_bg": True,
            "batch_dice": True
        },
        "batch_size": 16,
        "max_steps": 30 * MULTIPLIER,
        "validation_interval": 2,
    },
    "validation": {
        "dataset": {
            "transform": {
                "name": "testv2",
                "size": 320
            },
            "path": PATH,
            "span": [0.8, 1.0]
        },
        "batch_size": 32
    },
    "test": {
        "dataset": {
            "transform": {
                "name": "testv2",
                "size": 320
            },
            "path": PATH,
            "clips": []
        },
        "batch_size": 32
    },
}