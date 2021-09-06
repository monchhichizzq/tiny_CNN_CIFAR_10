# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 17:24
# @Author  : Zeqi@@
# @FileName: MobileNet.py
# @Software: PyCharm

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from Models.custom_layer import spike_counter

def depthwise_separable(x, params):
    # f1/f2 filter size, s1 stride of conv
    (s1,f2) = params
    x = DepthwiseConv2D((3,3),strides=(s1[0],s1[0]), padding='same',depthwise_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(f2[0]), (1,1), strides=(1,1), padding='same',
               kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def MobileNet_V1(img_input, shallow=False, classes=10):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        # Arguments
            alpha: optional parameter of the network to change the
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
    """
    # 32x32x3 -> 16x16x3
    x = Conv2D(int(32), (3, 3), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16x3 -> 16x16x64
    x = depthwise_separable(x, params=[(1,), (64,)])
    # 16x16x64 -> 8x8x128
    x = depthwise_separable(x, params=[(2,), (128,)])
    # 8x8x128 -> 8x8x128
    x = depthwise_separable(x, params=[(1,), (128,)])
    # 8x8x256 -> 4x4x256
    x = depthwise_separable(x, params=[(2,), (256,)])
    # 4x4x256 -> 4x4x256
    x = depthwise_separable(x, params=[(1,), (256,)])
    # 4x4x256 -> 2x2x512
    x = depthwise_separable(x, params=[(2,), (512,)])
    if not shallow:
        for _ in range(5):
            x = depthwise_separable(x, params=[(1,), (512,)])

    # 2x2x512 -> 1x1x1024
    x = depthwise_separable(x, params=[(2,), (1024,)])
    # 1x1x1024 -> 1x1x1024
    x = depthwise_separable(x, params=[(1,), (1024,)])

    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)
    return out

class MobileNet_V1_slim:
    def __init__(self, **kwargs):
        """Instantiates the MobileNet.Network has two hyper-parameters
            which are the width of network (controlled by alpha)
            and input size.
            # Arguments
                alpha: optional parameter of the network to change the
                    width of model.
                shallow: optional parameter for making network smaller.
                classes: optional number of classes to classify images
                    into.
        """
        self.l2_re = kwargs.get('l2_regularization', 1e-4)
        self.add_spike_layer = kwargs.get('add_spike_layer', False)
        self.count_spikes = kwargs.get('count_spikes', False)
        self.add_loss = kwargs.get('add_l1_loss', False)
        self.assign_inputs = kwargs.get('assign_inputs', False)

    def depthwise_separable(self, x, params, block_id):
        # f1/f2 filter size, s1 stride of conv
        (s1, f2) = params

        depthwise_name = 'depthwise_conv_%s'%(block_id)
        depthwise_spike = depthwise_name + '_spikes'
        depthwise_bn = depthwise_name + '_bn'
        depthwise_act = depthwise_name + '_relu'

        conv_name = 'conv_%s'%(block_id)
        conv_spike = conv_name + '_spikes'
        conv_bn = conv_name + '_bn'
        conv_act = conv_name + '_relu'

        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                            assign_inputs=self.assign_inputs, name=depthwise_spike)(x)

        x = DepthwiseConv2D((3, 3), strides=(s1[0], s1[0]), padding='same',
                            depthwise_initializer="he_normal", name=depthwise_name)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=depthwise_bn)(x)
        x = ReLU(name=depthwise_act)(x) # not relu6

        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                              assign_inputs=self.assign_inputs, name=conv_spike)(x)

        x = Conv2D(int(f2[0]), (1, 1), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(self.l2_re), name=conv_name)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=conv_bn)(x)
        x = ReLU(name=conv_act)(x)
        return x

    def build(self, img_input, shallow=False, classes=10):
        # change stride
        # 32x32x3 -> 32x32x3
        x = img_input
        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                            assign_inputs=self.assign_inputs, name='conv_spikes')(x)

        x = Conv2D(int(32), (3, 3), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(self.l2_re), name='conv')(x)
        x = BatchNormalization(name='conv_bn')(x)
        x = ReLU(name='conv_relu')(x)

        # 32x32x3 -> 32x32x64
        x = self.depthwise_separable(x, params=[(1,), (64,)], block_id=1)
        # 32x32x64 -> 32x32x128
        x = self.depthwise_separable(x, params=[(1,), (128,)], block_id=2)  # change stride
        # 32x32x128 -> 32x32x128
        x = self.depthwise_separable(x, params=[(1,), (128,)], block_id=3)
        # 32x32x128 -> 32x32x256
        x = self.depthwise_separable(x, params=[(1,), (256,)], block_id=4)   # change stride
        # 32x32x256 -> 32x32x256
        x = self.depthwise_separable(x, params=[(1,), (256,)], block_id=5)
        # 32x32x256 -> 16x16x512
        x = self.depthwise_separable(x, params=[(2,), (512,)], block_id=6)
        block_id = 7
        if not shallow:
            for i in range(5):
                x = self.depthwise_separable(x, params=[(1,), (512,)], block_id=block_id)
                block_id+=1
        # 16x16x512 -> 8x8x1024
        x = self.depthwise_separable(x, params=[(2,), (1024,)], block_id=block_id+1)
        # 8x8x1024 -> 8x8x1024
        x = self.depthwise_separable(x, params=[(1,), (1024,)], block_id=block_id+2)
        # 8x8x1024 -> 1x1x1024
        x = GlobalAveragePooling2D()(x)
        # 1x1x1024 -> 10
        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                              assign_inputs=self.assign_inputs, name='prediction_spikes')(x)
        out = Dense(classes, name='prediction')(x)
        out = Softmax()(out)
        model = Model(img_input, out, name='MobileNet_V1_slim')
        return model