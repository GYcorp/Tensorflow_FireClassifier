# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import core.efficientnet.efficientnet_builder as efficientnet

from utils.Utils import *

def Visualize(feature_maps, fc_w, classes):
    batch_size, h, w, c = feature_maps.get_shape().as_list()
    
    feature_maps = tf.reshape(feature_maps, [-1, h * w, c])
    fc_w = tf.reshape(fc_w, [-1, c, classes])
    
    flatted_heatmaps = tf.matmul(feature_maps, fc_w)
    heatmaps = tf.reshape(flatted_heatmaps, [-1, h, w, classes])
    
    min_value = tf.math.reduce_min(heatmaps, axis = [0, 1, 2])
    max_value = tf.math.reduce_max(heatmaps, axis = [0, 1, 2])
    heatmaps = (heatmaps - min_value) / (max_value - min_value) * 255.
    
    return tf.identity(heatmaps, name = 'heatmaps')

def Visualize_Conv(feature_maps):
    feature_maps = tf.nn.relu(feature_maps)
    
    min_value = tf.math.reduce_min(feature_maps, axis = [0, 1, 2])
    max_value = tf.math.reduce_max(feature_maps, axis = [0, 1, 2])
    heatmaps = (heatmaps - min_value) / (max_value - min_value) * 255.
    
    return tf.identity(heatmaps, name = 'heatmaps')

def EfficientNet(x, is_training, option):
    model_name = 'efficientnet-{}'.format(option['name'])
    
    log_print('# {}'.format(model_name), option['log_txt_path'])
    log_print('- mean = {}, std = {}'.format(efficientnet.MEAN_RGB, efficientnet.STDDEV_RGB), option['log_txt_path'])
    
    x = (x[..., ::-1] - efficientnet.MEAN_RGB) / efficientnet.STDDEV_RGB
    _, end_points = efficientnet.build_model_base(x, model_name, is_training)
    
    for i in range(1, 5 + 1):
        log_print('- reduction_{} : {}'.format(i, end_points['reduction_{}'.format(i)]), option['log_txt_path'])
    
    with tf.variable_scope('Classifier', reuse = tf.AUTO_REUSE):
        if option['mode'] == 'Conv1x1':
            feature_maps = tf.layers.conv2d(end_points[option['feature_extractor']], option['classes'], [1, 1], 1, name = 'feature_maps')
            logits = tf.reduce_mean(feature_maps, axis = [1, 2], name = 'GAP')
        else:
            feature_maps = end_points[option['feature_extractor']]
            x = tf.reduce_mean(feature_maps, axis = [1, 2], name = 'GAP')
            logits = tf.layers.dense(x, option['classes'], use_bias = False, name = 'logits')
        
        predictions = tf.nn.sigmoid(logits, name = 'outputs')
    
    return {
        'logits' : logits,
        'predictions' : predictions,
        'feature_maps' : feature_maps
    }

