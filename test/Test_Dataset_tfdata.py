import cv2
import multiprocessing

import numpy as np
import tensorflow as tf

from utils.Timer import *

root_dir = '//gynetworks/Data/DATA/Image/Recon_FireClassifier_DB_20200219/'

delete_dir = 'D:/_ImageDataset/Recon_FireClassifier_DB_20200120/'
load_dataset = lambda npy_path: np.load(npy_path, allow_pickle = True)

train_dic = {}
train_dic['positive'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_pos.npy')]
train_dic['negative'] = [[image_path.replace('\\', '/').replace(delete_dir, root_dir), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_neg.npy')]

def generate_dataset(dataset, batch_size, is_training):
    # def custom_map(image_path):
    #     JPEG_OPT = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}

    #     image_data = tf.read_file(image_path)
    #     image = tf.image.decode_jpeg(tf.reshape(image_data, shape=[]), 3, **JPEG_OPT)
    #     image = tf.image.resize_bicubic([image], [224, 224])
    #     image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)[0]

    #     return image
    def custom_map(image_path):
        image_data = tf.read_file(image_path)
        image = tf.image.decode_jpeg(tf.reshape(image_data, shape=[]), 3)
        image = tf.image.resize_bicubic([image], [224, 224])
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)[0]
        return image

    image_paths = tf.constant([data[0] for data in dataset], name = 'image_paths')
    labels = tf.constant([data[1:] for data in dataset], dtype = tf.float32, name = 'labels')

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if is_training:
        ds = ds.shuffle(len(dataset), reshuffle_each_iteration = True).repeat()

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            lambda image_path, label: (custom_map(image_path), label),
            batch_size = batch_size,
            # drop_remainder = is_training,
            num_parallel_batches = multiprocessing.cpu_count() // 2,
        )
    )
    ds = ds.prefetch(100)
    return ds

dataset = generate_dataset(train_dic['positive'], 64, False)

iterator = dataset.make_initializable_iterator()
images_op, labels_op = iterator.get_next()

print(images_op, labels_op)

sess = tf.Session()
sess.run(iterator.initializer)

timer = Timer()

while True:
    timer.tik()
    images, labels = sess.run([images_op, labels_op])
    print('{}, {} = {}ms'.format(images.shape, labels.shape, timer.tok()))

    for (image, label) in zip(images, labels):
        print(label)
        cv2.imshow('show', image)
        cv2.waitKey(0)

