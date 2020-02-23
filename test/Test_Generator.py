import cv2
import warnings
import multiprocessing

import numpy as np
import tensorflow as tf

from core.DataAugmentation import *

from utils.Timer import *
from utils.Dataflow import *
from utils.Generator import *

if __name__ == '__main__':
    root_dir = 'C:/DB/Recon_FireClassifier_DB_20200219/'

    delete_dir = 'D:/_ImageDataset/Recon_FireClassifier_DB_20200120/'
    load_dataset = lambda npy_path: np.load(npy_path, allow_pickle = True)

    train_dic = {}
    train_dic['positive'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_pos.npy')]
    train_dic['negative'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_neg.npy')]
    
    # train_dataset = generate_dataflow(train_dic['positive'][:32] + train_dic['negative'][:32], {
    train_dataset = generate_dataflow(train_dic['positive'] + train_dic['negative'], {
        'augmentors' : [
            # RandAugment(),
            # Weakly_Augment(),
        ],

        'shuffle' : False,
        'remainder' : False,

        'batch_size' : 64,
        'image_size' : (224, 224),

        'num_prefetch_for_dataset' : 10,
        'num_prefetch_for_batch' : 2,
    })
    
    image_var = tf.placeholder(tf.float32, [64, 224, 224, 3])
    label_var = tf.placeholder(tf.float32, [64, 2])

    generator = Generator({
        'dataset' : train_dataset,
        'placeholders' : [image_var, label_var],

        'queue_size' : 10,
    })
    
    images_op, labels_op = generator.dequeue()
    
    ###############################################
    # Run
    ###############################################
    sess = tf.Session()
    coord = tf.train.Coordinator()
    
    generator.set_session(sess)
    generator.set_coordinator(coord)
    generator.start()

    timer = Timer()

    while True:
        timer.tik()
        images, labels = sess.run([images_op, labels_op])

        print(images.shape, labels.shape, generator.size(), '{}ms'.format(timer.tok()))
        
        # for (image, label) in zip(images, labels):
        #     print(label)
        #     cv2.imshow('show', image.astype(np.uint8))
        #     cv2.waitKey(0)
        #     break

