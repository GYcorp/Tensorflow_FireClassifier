import cv2
import random

import numpy as np

from tensorpack import imgaug, dataset
from tensorpack.dataflow import AugmentImageComponent, PrefetchData, BatchData, MultiThreadMapData

import core.randaugment_ops.policies as found_policies
import core.randaugment_ops.augmentation_transforms as transform

def pad(x, border = 4):
    pad_x = np.pad(x, [[border, border], [border, border], [0, 0]], mode = 'reflect')
    return pad_x

def RandomPadandCrop(x):
    new_h, new_w = x.shape[:2]
    x = pad(x, new_w // 8)
    
    h, w = x.shape[:2]
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    x = x[top: top + new_h, left: left + new_w, :]
    return x

def RandomFlip(x):
    if np.random.rand() < 0.5:
        x = x[:, ::-1, :]
    return x

class RandAugment(imgaug.ImageAugmentor):
    def __init__(self):
        mean, std = transform.get_mean_and_std()
        policies = found_policies.randaug_policies()

        self._init(locals())

    def _augment(self, image, _):
        h, w, c = image.shape
        
        image = image.astype(np.float32)

        image /= 255.
        image = (image - self.mean) / self.std

        chosen_policy = random.choice(self.policies)
        image = transform.apply_policy(chosen_policy, image)
        # image = transform.cutout_numpy(image)
        
        image = (image * self.std) + self.mean
        image *= 255.    
        
        return image.astype(np.float32)

class Weakly_Augment(imgaug.ImageAugmentor):
    def __init__(self):
        self._init(dict())

    def _augment(self, image, label):
        image = RandomFlip(image)
        image = RandomPadandCrop(image)
        return image

