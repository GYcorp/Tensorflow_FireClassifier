# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

# tensorflow_gpu 1.12.0

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

from core.Config import *
from core.Classifier import *
from core.efficientnet.utils import *

from utils.Utils import *
from utils.Dataflow import *
from utils.Generator import *
from utils.NCCL_Utils import *
from utils.Tensorflow_Utils import *

#######################################################################################
# 0. Config
#######################################################################################
flags = get_config()

flags['warmup_iteration'] = int(flags['max_iteration'] * 0.05) # warmup iteration = 5%

width_coeff, depth_coeff, resolution, dropout_rate = efficientnet.efficientnet_params('efficientnet-{}'.format(flags['option']))
flags['image_size'] = resolution

num_gpu = len(flags['use_gpu'].split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = flags['use_gpu']

flags['init_learning_rate'] = 0.016 
flags['alpha_learning_rate'] = 0.002 

flags['batch_size'] = flags['batch_size_per_gpu'] * num_gpu
if flags['batch_size'] > 256:
    flags['init_learning_rate'] *= flags['batch_size'] / 256
    flags['alpha_learning_rate'] *= flags['batch_size'] / 256

model_name = '{}-{}-EfficientNet-{}'.format(flags['experimenter'], get_today(), flags['option'])
model_dir = './experiments/model/{}/'.format(model_name)
tensorboard_dir = './experiments/tensorboard/{}/'.format(model_name)

ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

if os.path.isfile(log_txt_path):
    open(log_txt_path, 'w').close()

#######################################################################################
# 1. Dataset
#######################################################################################
log_print('# {}'.format(model_name), log_txt_path)
log_print('{}'.format(json.dumps(flags, indent='\t')), log_txt_path)

augmentors = []
if flags['augment'] == 'randaugment':
    augmentors.append(RandAugment())
elif flags['augment'] == 'weakaugment':
    augmentors.append(Weakly_Augment())

train_dic = {
    'positive' : [],
    'negative' : [],
}

# valid_dic = {
#     'positive' : [],
#     'negative' : [],
# }

delete_dir = 'D:/_ImageDataset/Recon_FireClassifier_DB_20200120/'
load_dataset = lambda npy_path: np.load(npy_path, allow_pickle = True)

train_dic['positive'] += [[image_path.replace('\\', '/').replace(delete_dir, flags['root_dir']), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_pos.npy')]
train_dic['negative'] += [[image_path.replace('\\', '/').replace(delete_dir, flags['root_dir']), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/train_neg.npy')]

# valid_dic['positive'] += [[image_path.replace('\\', '/').replace(delete_dir, flags['root_dir']), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/valid_pos.npy')]
# valid_dic['negative'] += [[image_path.replace('\\', '/').replace(delete_dir, flags['root_dir']), int(flame), int(smoke)] for (image_path, flame, smoke) in load_dataset('./dataset/valid_neg.npy')]

#######################################################################################
# 1.1. Info (Dataset)
#######################################################################################
log_print('\n', log_txt_path)
log_print('### Train', log_txt_path)
for key in ['positive', 'negative']:
    log_print('=> {:10s} = {}'.format(key, len(train_dic[key])), log_txt_path)

# log_print('### Validation', log_txt_path)
# for key in ['positive', 'negative']:
#     log_print('=> {:10s} = {}'.format(key, len(valid_dic[key])), log_txt_path)

#######################################################################################
# 2. Generator & Queue
#######################################################################################
dataflow_option = {
    'augmentors' : augmentors,

    'shuffle' : True,
    'remainder' : False,

    'batch_size' : flags['batch_size'] // num_gpu,
    'image_size' : (flags['image_size'], flags['image_size']),
    
    'num_prefetch_for_dataset' : 1000,
    'num_prefetch_for_batch' : 100,

    'number_of_cores' : mp.cpu_count() // num_gpu,
}

train_image_var = tf.placeholder(tf.float32, [flags['batch_size'], flags['image_size'], flags['image_size'], 3])
train_label_var = tf.placeholder(tf.float32, [flags['batch_size'], 2])
is_training = tf.placeholder(tf.bool)

train_dataset_list = [generate_dataflow(train_dic['positive'] + train_dic['negative'], dataflow_option) for _ in range(num_gpu)]
train_generators = [Generator({'dataset' : train_dataset_list[i], 'placeholders' : [train_image_var, train_label_var], 'queue_size' : 100}) for i in range(num_gpu)]

log_print('[i] generate dataset and generators', log_txt_path)

#######################################################################################
# 3. Optimizer
#######################################################################################
global_step = tf.placeholder(dtype = tf.int32)

warmup_lr_op = tf.to_float(global_step) / tf.to_float(flags['warmup_iteration']) * flags['init_learning_rate']
decay_lr_op = tf.train.cosine_decay(
    flags['init_learning_rate'],
    global_step = global_step - flags['warmup_iteration'],
    decay_steps = flags['max_iteration'] - flags['warmup_iteration'],
    alpha = flags['alpha_learning_rate']
)

learning_rate = tf.where(global_step < flags['warmup_iteration'], warmup_lr_op, decay_lr_op)
optimizers = [tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True) for _ in range(num_gpu)]

#######################################################################################
# 4. Model
#######################################################################################
logits_list = []
predictions_list = []

class_loss_list = []
l2_loss_list = []
loss_list = []

grad_list = []

train_image_ops = []
train_label_ops = []

model_option = {
    'name' : flags['option'],
    'classes' : 2,
    'log_txt_path' : log_txt_path,
}

for gpu_id in range(num_gpu):
    train_image_op, train_label_op = train_generators[gpu_id].dequeue()
    train_image_ops.append(train_image_op)
    train_label_ops.append(train_label_op)

    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)), tf.variable_scope('tower%d'%gpu_id):
        output_dic = EfficientNet(train_image_op, is_training, model_option)
        logits_list.append(output_dic['logits'])
        predictions_list.append(output_dic['predictions'])

        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = output_dic['logits'], labels = train_label_op)
        class_loss = tf.reduce_mean(class_loss)
        class_loss_list.append(class_loss)

        train_vars = tf.trainable_variables(scope = 'tower%d'%gpu_id)
        l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * flags['weight_decay']
        l2_loss_list.append(l2_reg_loss)

        loss = class_loss + l2_reg_loss
        loss_list.append(loss)

        grad_list.append([x for x in optimizers[gpu_id].compute_gradients(loss) if x[0] is not None])

    log_print('[i] calculate gradients (tower%d)'%gpu_id, log_txt_path)

loss_op = tf.concat(loss_list, axis = 0)
class_loss_op = tf.concat(class_loss_list, axis = 0)
l2_reg_loss_op = tf.concat(l2_loss_list, axis = 0)

logits_op = tf.concat(logits_list, axis = 0)
predictions_op = tf.concat(predictions_list, axis = 0)

label_op = tf.concat(train_label_ops, axis = 0)

log_print('[i] finish concatenation', log_txt_path)

# calculate gradients using NCCL
grads, all_vars = split_grad_list(grad_list)
reduced_grad = allreduce_grads(grads, average=True)
grads = merge_grad_list(reduced_grad, all_vars)

# optimize model using NCCL
train_ops = []
for index, grad_and_vars in enumerate(grads):
    # apply_gradients may create variables. 
    with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=index)):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='tower%d' % index)
        with tf.control_dependencies(update_ops):
            train_ops.append(optimizers[index].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(index)))

optimize_op = tf.group(*train_ops, name='train_op')

log_print('[i] finish optimizer', log_txt_path)

#######################################################################################
# 5. Metrics
#######################################################################################
correct_op = tf.equal(tf.greater_equal(predictions_op, 0.5), tf.greater_equal(label_op, 0.5))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

#######################################################################################
# 6. tensorboard
#######################################################################################
train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Clasification_Loss' : class_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op, 
    'Accuracy/Train_Accuracy' : accuracy_op,
    'Learning_rate' : learning_rate,
}
train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

valid_summary_dic = {
    'Accuracy/Validation_Accuracy' : tf.placeholder(tf.float32),
    'Accuracy/Validation_Positive_Accuracy' : tf.placeholder(tf.float32),
    'Accuracy/Validation_Negative_Accuracy' : tf.placeholder(tf.float32),
}
valid_summary_op = tf.summary.merge([tf.summary.scalar(name, valid_summary_dic[name]) for name in valid_summary_dic.keys()])

train_writer = tf.summary.FileWriter(tensorboard_dir)
log_print('[i] tensorboard directory is {}'.format(tensorboard_dir), log_txt_path)

#######################################################################################
# 7. create Session and Saver.
#######################################################################################
sess = tf.Session()
coord = tf.train.Coordinator()

saver = tf.train.Saver(
    var_list = tf.trainable_variables(scope = 'tower0'),
    max_to_keep = 20
)

# pretrained model
pretrained_model_name = 'efficientnet-{}'.format(flags['option'])
pretrained_model_path = './pretrained_model/{}/model.ckpt'.format(pretrained_model_name)

imagenet_saver = tf.train.Saver(var_list = [var for var in train_vars if pretrained_model_name in var.name])
imagenet_saver.restore(sess, pretrained_model_path)

log_print('[i] restore pretrained model ({}) -> {}'.format(pretrained_model_name, pretrained_model_path), log_txt_path)

#######################################################################################
# 8. initialize
#######################################################################################
sync_op = get_post_init_ops()

sess.run(sync_op)
sess.run(tf.global_variables_initializer())

for train_generator in train_generators:
    train_generator.set_session(sess)
    train_generator.set_coordinator(coord)
    train_generator.start()

    log_print('[i] start train generator ({})'.format(train_generator), log_txt_path)

#######################################################################################
# 7. Train
#######################################################################################
loss_list = []
class_loss_list = []
l2_reg_loss_list = []
accuracy_list = []
train_time = time.time()

best_valid_accuracy = 0
train_ops = [optimize_op, loss_op, class_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

for iter in range(1, flags['max_iteration'] + 1):
    _feed_dict = {
        is_training : True,
        global_step : iter,
    }
    _, loss, class_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)

    loss_list.append(loss)
    class_loss_list.append(class_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)
    train_writer.add_summary(summary, iter)
    
    if iter % flags['log_iteration'] == 0:
        loss = np.mean(loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        accuracy = np.mean(accuracy_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter = {}, loss = {:.4f}, class_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, class_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
        
        loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []
        accuracy_list = []
        train_time = time.time()
    
    #######################################################################################
    # 8. Validation
    #######################################################################################
    if iter % flags['valid_iteration'] == 0:
        saver.save(sess, ckpt_format.format(iter))   

        # valid_time = time.time()
        # valid_accuracy_dic = {}

        # for key in ['positive', 'negative']:
        #     valid_accuracy_list = []
        #     valid_dataset = valid_dic[key]
            
        #     valid_iteration = len(valid_dataset) // flags['batch_size']
        #     for i in range(valid_iteration):
        #         sys.stdout.write('\rValidation = [{}/{}]'.format(i + 1, valid_iteration))
        #         sys.stdout.flush()

        #         batch_dataset = valid_dataset[i * flags['batch_size'] : (i + 1) * flags['batch_size']]

        #         batch_image_data = []
        #         batch_label_data = []

        #         for image_path, flame, smoke in batch_dataset:
        #             image = imread(image_path)    
        #             if image is None:
        #                 print('[!] validation (imread_list) : {}'.format(image_path))
        #                 continue
                        
        #             image = cv2.resize(image, (flags['max_image_size'], flags['max_image_size']), interpolation = cv2.INTER_CUBIC)
                            
        #             batch_image_data.append(image.astype(np.float32))
        #             batch_label_data.append([int(flame), int(smoke)])

        #         _feed_dict = {
        #             test_image_var : batch_image_data,
        #             test_label_var : batch_label_data,
        #             is_training : False
        #         }
        #         accuracy = sess.run(test_accuracy_op, feed_dict = _feed_dict)
        #         valid_accuracy_list.append(accuracy)
            
        #     valid_accuracy = np.mean(valid_accuracy_list)
        #     valid_accuracy_dic[key] = valid_accuracy

        # total_valid_accuracy = np.mean([valid_accuracy_dic[key] for key in ['positive', 'negative']])
        
        # summary = sess.run(valid_summary_op, feed_dict = {
        #     valid_summary_dic['Accuracy/Validation_Accuracy'] : total_valid_accuracy,
        #     valid_summary_dic['Accuracy/Validation_Positive_Accuracy'] : valid_accuracy_dic['positive'],
        #     valid_summary_dic['Accuracy/Validation_Negative_Accuracy'] : valid_accuracy_dic['negative'],
        # })
        # train_writer.add_summary(summary, iter)
        
        # if best_valid_accuracy <= total_valid_accuracy:
        #     best_valid_accuracy = total_valid_accuracy
        #     saver.save(sess, ckpt_format.format(iter))            
        
        # train_time = time.time()
        # valid_time = int(time.time() - valid_time)

        # print()
        # log_print('[i] iter = {}, total_valid_accuracy = {:.2f}, best_valid_accuracy = {:.2f}, valid_time = {}sec'.format(iter, total_valid_accuracy, best_valid_accuracy, valid_time), log_txt_path)

# saver.save(sess, ckpt_format.format('end'))    

