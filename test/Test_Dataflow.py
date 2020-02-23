import cv2
import warnings
import multiprocessing

import numpy as np
import tensorflow as tf

from core.DataAugmentation import *

from utils.Timer import *
from utils.Dataflow import *

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    root_dir = 'C:/DB/Recon_FireClassifier_DB_20200219/'

    delete_dir = 'D:/_ImageDataset/Recon_FireClassifier_DB_20200120/'
    load_dataset = lambda npy_path: np.load(npy_path, allow_pickle = True)

    train_dic = {}
    train_dic['positive'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_pos.npy')]
    train_dic['negative'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_neg.npy')]

    train_dataset = generate_dataflow(train_dic['positive'] + train_dic['negative'], {
        'augmentors' : [
            RandAugment(),
            # Weakly_Augment(),
        ],

        'shuffle' : False,
        'remainder' : False,

        'batch_size' : 64,
        'image_size' : (224, 224),

        'num_prefetch_for_dataset' : 10,
        'num_prefetch_for_batch' : 2,
    })
    train_dataset.reset_state()
    
    count = 0

    for (batch_image_data, batch_label_data) in train_dataset.get_data():
        print(batch_image_data.shape, batch_image_data[0].dtype)
        print(batch_label_data.shape)
        print()
        
        count += 1
        if count % 5 == 0:
            print('reset')
            train_dataset.reset_state()

        cv2.imshow('show', batch_image_data[0].astype(np.uint8))
        cv2.waitKey(0)

        # input()

    # sess = tf.Session()
    # sess.run(init_op)

    # timer = Timer()

    # while True:
    #     timer.tik()
    #     images, labels = sess.run([images_op, labels_op])
    #     print('{}, {} = {}ms'.format(images.shape, labels.shape, timer.tok()))

    #     for (image, label) in zip(images, labels):
    #         print(label)
    #         cv2.imshow('show', image[..., ::-1].astype(np.uint8))
    #         cv2.waitKey(0)

