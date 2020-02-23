# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

from core.Classifier import *

input_var = tf.placeholder(tf.float32, [32, 224, 224, 3], name = 'images')
label_var = tf.placeholder(tf.float32, [32, 10], name = 'labels')
is_training = tf.placeholder(tf.bool)

logits_op, predictions_op, _ = EfficientNet(input_var, is_training, {
    'name' : 'b0',
    'custom_getter' : None,
    
    'final_activation' : tf.nn.softmax,
    'final_name' : 'outputs',

    'classes' : 10,
})

