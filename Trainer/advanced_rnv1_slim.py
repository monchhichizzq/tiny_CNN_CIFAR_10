# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 17:17
# @Author  : Zeqi@@
# @FileName: mbv1_slim.py.py
# @Software: PyCharm


'''
reference: https://www.twblogs.net/a/5c3630d3bd9eee35b21d3fa9
我們把前三個 down sampling 取消，也即前三個stride=2 改爲 stride =1，保證 feature map 的 resolution.
'''

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
from os.path import dirname, realpath, sep, pardir
abs_path = dirname(realpath(__file__)) + sep + pardir
sys.path.append(abs_path)

import datetime
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from Utils.utils import color_preprocessing, scheduler
# from Models.MobileNet import MobileNet_V1_slim
from Models.ResNet18 import build_ResNet, ResNet_slim
from Loss.loss import Total_loss
from ads.Callbacks.model_history import ModelHistory
from ads.Callbacks.model_l1_coeff_update import set_l1
from ads.Callbacks.Checkpoint_callbacks import ModelCheckpoint
from ads.Callbacks.model_macs_callbacks import MACs_Counter_Callback



training           = True
l1_coef            = 8e-7
num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
pretrained_path = '../Checkpoints/resnet_slim/ep_00054-valloss_0.54-valacc_94.06.h5'
# pretrained_path = '../Checkpoints/resnet_slim_l1_1e-06_from_scratch/ep_00103-valloss_0.47-closs_0.30-sloss_0.18-valacc_93.46-spikes_0.13.h5'

tb_dir = '../Checkpoints'
log_filepath  = os.path.join(tb_dir, 'resnet_slim_l1_{}_from_scratch'.format(l1_coef))
os.makedirs(log_filepath, exist_ok=True)

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test  = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

# Load model
img_input=Input(shape=(32, 32, 3))

ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                 'ResNet34': [3, 4, 6, 3]}

model = ResNet_slim(layers_dims=ResNet_Config['ResNet18'],
                    nb_classes=num_classes,
                    use_bn=True,
                    use_bias=False,
                    use_avg=True,
                    add_spike_layer=True,
                    count_spikes=True,
                    add_l1_loss=True,
                    assign_inputs=True).build(img_input)

model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)

set_l1(model, l1_coef, ignore_layers=[], verbose=True)

# Set loss
total_loss = Total_loss(model)

# set optimizer
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1)
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy',
                       'categorical_crossentropy',
                       total_loss.sparse_loss,
                       total_loss.nb_spikes,
                       total_loss.nb_total_spikes])


save_path=log_filepath
save_name='history.npy'
target_metrics=['loss', 'sparse_loss', 'accuracy', 'nb_spikes', 'nb_total_spikes',
                'val_loss', 'val_sparse_loss', 'val_accuracy', 'val_nb_spikes']
check_sparsity=False
history_callback = ModelHistory(log_filepath, save_name,
                                target_metrics, check_sparsity, verbose=False)

checkpoint_callback = ModelCheckpoint(save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      save_weights_only=False,
                                      verbose=1,
                                      add_spikes=True,
                                      model_name='ep_%05d-valloss_%0.2f-closs_%0.2f-sloss_%0.2f-valacc_%0.2f-spikes_%0.2f.h5')

macs_callbacks = MACs_Counter_Callback(sequence_inputs=False, verbose=True)

cbks = [reduce_lr, checkpoint_callback, history_callback]
val_callbacks = [macs_callbacks]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)

# start training
if training:
    metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2, callbacks=cbks)
    loss, accuracy, cat_loss, sparse_loss, nb_spikes, nb_total_spikes = metrics
    print('loss:  %0.2f, accuracy: %0.2f, cat_loss: %0.2f, sparse_loss: %0.2f, nb_spikes: %0.2f M,\
    nb_total_spikes: %0.2f M '%(loss, accuracy*100, cat_loss, sparse_loss, nb_spikes, nb_total_spikes))

    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test),
                        verbose=2)

else:
    metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1, callbacks=val_callbacks)
    loss, accuracy, cat_loss, sparse_loss, nb_spikes, nb_total_spikes = metrics
    print('loss:  %0.2f, accuracy: %0.2f, cat_loss: %0.2f, sparse_loss: %0.2f, nb_spikes: %0.2f M,\
    nb_total_spikes: %0.2f M '%(loss, accuracy*100, cat_loss, sparse_loss, nb_spikes, nb_total_spikes))

# save_dir = 'model_save/mbv1_slim'
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, 'mobilenet_slim_spikes.h5')
# model.save(save_path)