# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import random
import argparse

import numpy as np
import tensorflow as tf

import multiprocessing as mp

import tracemalloc

from core.Config import *
from core.Classifier import *
from core.efficientnet.utils import *

from utils.Utils import *
from utils.Dataflow import *
from utils.Generator import *
from utils.Tensorflow_Utils import *

if __name__ == '__main__':

    #######################################################################################
    # 0. Config
    #######################################################################################
    flags = get_config()

    #######################################################################################
    # 1. Dataset
    #######################################################################################
    augmentors = []
    use_cores = mp.cpu_count() // 8

    if flags.augment == 'randaugment':
        use_cores = mp.cpu_count() // 2
        augmentors.append(RandAugment())

    elif flags.augment == 'weakaugment':
        augmentors.append(Weakly_Augment())

    train_dic = {
        'positive' : [],
        'negative' : [],
    }

    delete_dir = 'D:/_ImageDataset/Recon_FireClassifier_DB_20200120/'
    load_dataset = lambda npy_path: np.load(npy_path, allow_pickle = True)

    train_dic['positive'] += [[image_path.replace('\\', '/').replace(delete_dir, flags.root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_pos.npy')]
    train_dic['negative'] += [[image_path.replace('\\', '/').replace(delete_dir, flags.root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_neg.npy')]

    #######################################################################################
    # 2. Generator & Queue
    #######################################################################################
    dataflow_option = {
        'augmentors' : augmentors,

        'shuffle' : True,
        'remainder' : False,
        
        'batch_size' : flags.batch_size,
        'image_size' : (flags.image_size, flags.image_size),
        
        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : use_cores,
    }

    train_dataset = generate_dataflow(train_dic['positive'] + train_dic['negative'], dataflow_option)
    
    train_iterator = train_dataset.get_data()

    for iter in range(1, flags.max_iteration + 1):
        batch_image_data, batch_label_data = next(train_iterator)
        del batch_image_data, batch_label_data

        time.sleep(0.01)

        if iter % 100 == 0:
            print(iter)
