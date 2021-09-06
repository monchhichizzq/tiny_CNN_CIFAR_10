# -*- coding: utf-8 -*-
# @Time    : 2021/9/2 22:31
# @Author  : Zeqi@@
# @FileName: mbv1_from_scratch.py
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
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

from Models.MobileNet import MobileNet_V1_slim
from Utils.utils import color_preprocessing, scheduler


num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test  = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

# Load model
img_input=Input(shape=(32, 32, 3))
mbv1 = MobileNet_V1_slim(add_spike_layer=False, count_spikes=True, add_l1_loss=True, assign_inputs=True)
model = mbv1.build(img_input, shallow=False, classes=10)


# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_dir = '../tensorboard'
log_filepath  = os.path.join(tb_dir, 'mobilenet_slim')
log_dir=os.path.join(log_filepath, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)

# start training
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test),
                    verbose=2)

# save model
save_dir = 'checkpoint/mbv1_slim'
os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'mobilenet_slim.h5')
model.save(save_path)