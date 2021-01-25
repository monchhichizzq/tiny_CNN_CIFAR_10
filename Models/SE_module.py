# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 0:48
# @Author  : Zeqi@@
# @FileName: SE_module.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.activations import sigmoid

def Squeeze_excitation_layer(input_x, out_dim, ratio):

    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=out_dim / ratio)(squeeze)
    excitation = ReLU()(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation

    return scale

def Squeeze_excitation_layer_simple(input_x, out_dim, ratio, layer_name):

    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = sigmoid(squeeze)
    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation

    return scale