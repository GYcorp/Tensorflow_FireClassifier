# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np

def random_bbox(size, lamb):
    w, h = size

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    cut_ratio = np.sqrt(1. - lamb)
    cut_w = np.int(w * cut_ratio)
    cut_h = np.int(h * cut_ratio)

    xmin = np.clip(cx - cut_w // 2, 0, w - 1)
    ymin = np.clip(cy - cut_h // 2, 0, h - 1)
    xmax = np.clip(cx + cut_w // 2, 0, w - 1)
    ymax = np.clip(cy + cut_h // 2, 0, h - 1)
    
    return xmin, ymin, xmax, ymax

def CutMix(images, labels, beta = 1.0):
    batch_size, h, w, c = images.shape

    indexs = np.random.permutation(batch_size)
    lamb = np.random.beta(beta, beta)
    
    xmin, ymin, xmax, ymax = random_bbox([w, h], lamb)
    lamb = 1 - ((xmax - xmin) * (ymax - ymin) / (w * h))

    images[:, ymin:ymax, xmin:xmax, :] = images[indexs, ymin:ymax, xmin:xmax, :]
    labels = lamb * labels + (1 - lamb) * labels[indexs, :]

    return images, labels

