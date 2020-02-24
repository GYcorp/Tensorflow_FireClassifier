import os
import sys
import cv2
import glob
import time
import argparse

import numpy as np
import tensorflow as tf

from core.Classifier import *

from utils.Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='HelmetClassifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # preprocessing
    parser.add_argument('--max_image_size', dest='max_image_size', help='max_image_size', default=224, type=int)
    
    # update !!
    parser.add_argument('--experimenter', dest='experimenter', help='experimenter', default='JSH', type=str)
    parser.add_argument('--error_dir', dest='error_dir', help='error_dir', default='D:/_ImageDataset/', type=str)
    parser.add_argument('--root_dir', dest='root_dir', help='root_dir', default='D:/_ImageDataset/Recon_HelmetClassifier_DB_20191206/', type=str)
    
    # gpu option
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', default=32, type=int)
    
    # model option
    parser.add_argument('--option', dest='option', default='b0', type=str)
    parser.add_argument('--ckpt_path', dest='ckpt_path', help='ckpt_path', type=str)

    return parser.parse_args()

args = vars(parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

model_dir = os.path.dirname(args['ckpt_path'])
model_name = os.path.basename(args['ckpt_path'])

log_txt_path = model_dir + '{}_accuracy.txt'.format(model_name)

image_var = tf.placeholder(tf.float32, [None] + [args['max_image_size'], args['max_image_size'], 3], name = 'images')
label_var = tf.placeholder(tf.float32, [None, 3])
is_training = tf.placeholder(tf.bool)

output_dic = EfficientNet(image_var, is_training, option)
predictions_op = output_dic['predictions']

correct_op = tf.equal(tf.argmax(predictions_op, axis = 1), tf.argmax(label_var, axis = 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, args['ckpt_path'])

test_dic = {
    'positive' : [],
    'negative' : [],
}

dataset = np.load('./dataset/test_crop_dataset.npy', allow_pickle = True)
test_dic['positive'] += [[args['root_dir'] + image_name, bbox, size] for (image_name, bbox, size) in dataset.item().get('positive')]
test_dic['negative'] += [[args['root_dir'] + image_name, bbox, size] for (image_name, bbox, size) in dataset.item().get('negative')]

dataset = np.load('./dataset/test.npy', allow_pickle = True)
test_dic['positive'] += [args['root_dir'] + image_name for image_name in dataset.item().get('positive')]
test_dic['negative'] += [args['root_dir'] + image_name for image_name in dataset.item().get('negative')]

test_accuracy_dic = {}
test_time = time.time()

log_print('### Test', log_txt_path)
for key in ['positive', 'negative']:
    log_print('=> {:10s} = {}'.format(key, len(test_dic[key])), log_txt_path)

for key in ['positive', 'negative']:
    test_accuracy_list = []
    test_dataset = test_dic[key]
    
    if key == 'positive':
        label = [0, 0, 1]
    else:
        label = [0, 1, 0]
    
    for i in range(len(test_dataset) // args['batch_size']):
        batch_dataset = test_dataset[i * args['batch_size'] : (i + 1) * args['batch_size']]

        batch_image_data = []
        batch_label_data = []
        
        for batch_data in batch_dataset:
            if type(batch_data) == list:
                image_path, bbox, size = batch_data

                image = cv2.imread(image_path)    
                if image is None:
                    if os.path.isfile(image_path):
                        print('[!] delete : {}'.format(image_path))
                        # os.remove(image_path)
                    continue
                else:
                    xmin, ymin, xmax, ymax = bbox

                    image = image[ymin : ymax, xmin : xmax]
            else:
                image_path = batch_data

                image = cv2.imread(image_path)
                if image is None:
                    print(image_path)
                    continue
            
            image = cv2.resize(image, (args['max_image_size'], args['max_image_size']), interpolation = cv2.INTER_CUBIC)
                    
            batch_image_data.append(image.astype(np.float32))
            batch_label_data.append(label)

        _feed_dict = {
            input_var : batch_image_data,
            label_var : batch_label_data,
            is_training : False
        }
        accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
        test_accuracy_list.append(accuracy)
    
    test_accuracy = np.mean(test_accuracy_list)
    test_accuracy_dic[key] = test_accuracy

total_test_accuracy = [test_accuracy_dic[key] for key in ['positive', 'negative']]
total_test_accuracy = np.mean(total_test_accuracy)

test_time = int(time.time() - test_time)

log_print('# Test = {}sec'.format(test_time), log_txt_path)
log_print('- Positive Accuracy : {}'.format(test_accuracy_dic['positive']), log_txt_path)
log_print('- Negative Accuracy : {}'.format(test_accuracy_dic['negative']), log_txt_path)
log_print('- Mean Accuracy : {}'.format(total_test_accuracy), log_txt_path)
