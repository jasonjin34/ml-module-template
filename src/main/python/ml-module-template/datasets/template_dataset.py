"""
Example dataset buildup for the Segementation 
The dataset is based on COCO dataset
"""


import glob
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import logging
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from ml_module_template.common.utils import register


logger = logging.getLogger(__name__)

def get_patches(data_path, field):
    field_dir = os.path.join(data_path, field)
    if not os.path.isdir(field_dir):
        logger.warning("Directory not found %s: Skipping", field_dir)
        return [], []
    # warning, not always same sorting
    imgs = glob.glob(os.path.join(field_dir, '**_img.npy'))
    masks = glob.glob(os.path.join(field_dir, '**_mask.npy'))

    img_ids = set([img[:-len('_img.npy')] for img in imgs])
    mask_ids = set([mask[:-len('_mask.npy')] for mask in masks])

    ids = img_ids.intersection(mask_ids)


    missing_imgs = mask_ids - img_ids
    missing_masks = img_ids - mask_ids

    if len(missing_imgs) > 0:
        logger.warning("Missing imgs for the following masks %s", missing_imgs)
    if len(missing_masks) > 0:
        logger.warning("Missing masks for the following imgs %s", missing_masks)

    ids = sorted(ids)
    imgs = [id + '_img.npy' for id in ids]
    masks = [id + '_mask.npy' for id in ids]
    return imgs, masks


def get_all_fields_from_dir(data_path):
    return [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]


def get_idxs_by_span(idxs, span):
    import math
    f, t = span
    assert 0.0 <= f <= 1.0, "Invalid from %f" % f
    assert 0.0 <= t <= 1.0, "Invalid to %f" % t
    assert f < t, "from must be smaller than to %f %f" % (f, t)
    f = math.ceil(f * len(idxs))
    t = math.floor(t * len(idxs))
    idxs = idxs[f:t]
    return idxs


def shuffle_deterministic(array, identifier):
    # same seed for same field
    np.random.seed(hash(identifier) % 2**32)
    np.random.shuffle(array)


def make_data(data_path, fields, span=None, num_samples=-1):
    """
    Args:
        num_samples (int): -1 means all
        seed (int): Used to initialize the seed, this will produce the same training dataset
        **kwargs: Keyword arguements passed to class
    """
    all_samples = []
    all_labels = []
    all_clips = []

    # set seed
    for field in fields:
        imgs, masks = get_patches(data_path, field)
        if span is not None:
            idxs = np.arange(len(imgs))
            shuffle_deterministic(idxs, field)
            idxs = get_idxs_by_span(idxs, span)
            logger.debug("Using idxs %s for %s", idxs, field)
            imgs = np.array(imgs)[idxs]
            masks = np.array(masks)[idxs]

        all_samples.extend(imgs)
        all_labels.extend(masks)
        all_clips.extend([field] * len(imgs))

    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)
    all_clips = np.array(all_clips)

    if num_samples > -1:
        idxs = np.random.choice(len(all_samples), num_samples, replace=False)
        all_samples = all_samples[idxs]
        all_labels = all_labels[idxs]
        all_clips = all_clips[idxs]

    return all_samples, all_labels, all_clips


class TemplateDataset(Dataset):
    def __init__(self, data, transform_imgaug=None, transform=None, cotton_as_weed=False):
        """
        """
        img_files, mask_files, clips = data
        assert len(img_files) == len(mask_files), "must be same size"
        self.img_files = img_files
        self.mask_files = mask_files
        self._clips = clips
        self.clips = np.unique(clips)
        self.transform = transform
        self.transform_imgaug = transform_imgaug

    @staticmethod
    def make_binary_mask(mask):
        """
        Converts a mask with multiple catgories to a binary spray not spray mask
        """
        if len(mask.shape) != 3:
            raise ValueError(f"Expected 3D input. Got {mask.shape}")
        binary = np.zeros(mask.shape[-2:], dtype=np.bool)

        for m in mask:
            binary = np.logical_or(binary, m)
        return binary

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]
        # TODO what to do with alpha channel
        img = np.load(img_file)[:, :, :3]
        mask = np.load(mask_file)
        # TODO
        if len(mask.shape) == 3:
            mask = self.make_binary_mask(mask)

        clip = self._clips[idx]
        if self.transform_imgaug is not None:
            segmap = SegmentationMapsOnImage(mask, shape=img.shape)
            img, mask = self.transform_imgaug(image=img, segmentation_maps=segmap)
            img = img.astype(np.uint8)
            mask = mask.get_arr()
            # add channel dimension
        if self.transform is not None:
            img = self.transform(img)
            mask = mask[None, :, :].astype(np.bool)
            mask = torch.from_numpy(mask)
        d = {'img': img, 'mask': mask, 'img_file': img_file,
             'mask_file': mask_file, 'clip': clip}
        return d

    def __len__(self):
        return len(self.img_files)


@register('dataset', 'template_dataset')
def build(cfg, transform_imgaug=None):
    debug = cfg.get('debug', False)
    num_samples = cfg.get('num_samples', -1)
    clips = cfg.get('clips', [])
    span = cfg.get('span', None)
    logger.info(cfg)
    data_dir = cfg.path

    fields = []
    fields.extend(clips)

    # use all folders in data
    fields = get_all_fields_from_dir(data_dir)

    fields = set(fields)

    if debug:
        transform_torch = None
    else:
        from torchvision import transforms

        transform_torch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    data = make_data(data_dir, fields, span,
                     num_samples=num_samples),

    return TemplateDataset(
            *data,
            transform_imgaug=transform_imgaug,
            transform=transform_torch,
            )
