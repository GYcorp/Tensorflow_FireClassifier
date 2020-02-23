

import os
import cv2
import time
import sys
sys.path.insert(1, './')

import numpy as np
import tensorflow as tf


# from core.Define import *
from core.Classifier import *

from tensorflow.python.platform import app
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_util

##################################################################################################
# Define
##################################################################################################
model_path = './experiments/model/EfficientNet-b0_#mixup_#random_crop_#min@64_#max@224/end.ckpt'
pb_dir = './test/'

pb_name = 'GY_HelmetClassifier_b0_error_woo.pb'
##################################################################################################

input_var = tf.placeholder(tf.float32, [None, 224, 224, 3], name = 'images')
logits, predictions, feature_maps = EfficientNet(input_var, False, {
        'name' : 'b0',
        'custom_getter' : None,

        'final_activation' : tf.nn.softmax,
        'final_name' : 'softmax',

        'classes' : 3
})
# logits, predictions, feature_maps = Inception_ResNet_v2(input_var, False)

vars = tf.trainable_variables()
for var in vars:
    if 'logits' in var.name:
        fc_w = var

print(feature_maps, fc_w)
heatmaps = Visualize(feature_maps, fc_w)
print(heatmaps)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, model_path)
    
gd = sess.graph.as_graph_def()
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['Classifier/softmax', 'heatmaps'])
tf.train.write_graph(converted_graph_def, pb_dir, pb_name, as_text=False)
print('freeze graph save complete')

