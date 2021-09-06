# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 20:52
# @Author  : Zeqi@@
# @FileName: ResNet.py
# @Software: PyCharm

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras import regularizers
from ads.Custom_layers.spike_counting_layer import spike_counter
from ads.Custom_layers.custom_regularizer import BN_Regularizer


class ResNet_slim():
    def __init__(self, layers_dims, nb_classes,
                 use_bn=True, use_bias=False,
                 use_avg=False, **kwargs):
        super(ResNet_slim, self).__init__()
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.use_avg = use_avg
        self.layers_dims = layers_dims
        self.nb_classes = nb_classes

        self.add_spike_layer = kwargs.get('add_spike_layer', False)
        self.count_spikes = kwargs.get('count_spikes', False)
        self.add_loss = kwargs.get('add_l1_loss', False)
        self.assign_inputs = kwargs.get('assign_inputs', False)
        self.l2_re = kwargs.get('l2_regularization', 1e-4)
        self.sparse_factor = kwargs.get('sparse_factor', 1e-4)

    def build_basic_block(self, inputs, filter_num, blocks, stride, module_name):
        # The first block stride of each layer may be non-1
        x = self.Basic_Block(inputs, filter_num, stride, block_name='{}_{}'.format(module_name, 0))

        for i in range(1, blocks):
            x = self.Basic_Block(x, filter_num, stride=1, block_name='{}_{}'.format(module_name, i))

        return x

    def Basic_Block(self, inputs, filter_num, stride=1, block_name=None):
        conv_name_1 = 'block_' + block_name + '_conv_1'
        conv_name_2 = 'block_' + block_name + '_conv_2'
        skip_connection = 'block_' + block_name + '_skip_connection'

        # Part 1
        b_in = inputs
        if self.add_spike_layer:
            b_in = spike_counter(self.count_spikes, self.add_loss,
                                assign_inputs=self.assign_inputs,
                                 name=conv_name_1+'_spikes')(b_in)

        x = Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                   kernel_regularizer=regularizers.l2(self.l2_re),
                   use_bias=self.use_bias, name=conv_name_1)(b_in)
        if self.use_bn:
            x = BatchNormalization(gamma_regularizer=BN_Regularizer(sparse_factor=self.sparse_factor),
                                   name=conv_name_1 + '_bn')(x)
        x = ReLU(name=conv_name_1 + '_relu')(x)

        # Part 2
        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                                assign_inputs=self.assign_inputs,
                                 name=conv_name_2+'_spikes')(x)
        x = Conv2D(filter_num, (3, 3), strides=1,
                   padding='same', kernel_regularizer=regularizers.l2(self.l2_re),
                   use_bias=self.use_bias, name=conv_name_2)(x)
        if self.use_bn:
            x = BatchNormalization(gamma_regularizer=BN_Regularizer(sparse_factor=self.sparse_factor),
                                   name=conv_name_2 + '_bn')(x)

        # skip
        if stride != 1 or filter_num==128:
            if self.add_spike_layer:
                b_in = spike_counter(self.count_spikes, self.add_loss,
                                  assign_inputs=self.assign_inputs,
                                  name=skip_connection + '_spikes')(b_in)
            residual = Conv2D(filter_num, (1, 1), strides=stride,
                              kernel_regularizer=regularizers.l2(self.l2_re),
                              use_bias=self.use_bias, name=skip_connection)(b_in)
        else:
            residual = b_in

        # Add
        x = Add(name='block_' + block_name + '_residual_add')([x, residual])
        out = ReLU(name='block_' + block_name + '_residual_add_relu')(x)

        return out

    def ConvBn_Block(self, x, num_filters, kernel_size, strides, block_name):
        conv_name = 'block_convbn' + block_name + '_conv'
        conv_spike = 'block_convbn' + block_name + '_conv_spikes'

        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                              assign_inputs=self.assign_inputs, name=conv_spike)(x)

        x = Conv2D(num_filters, kernel_size,
                   kernel_regularizer=regularizers.l2(self.l2_re),
                   strides=strides, padding='same', name=conv_name)(x)
        x = BatchNormalization(gamma_regularizer=BN_Regularizer(sparse_factor=self.sparse_factor),
                               name=conv_name + '_bn')(x)
        out = ReLU(name=conv_name + '_relu')(x)
        return out

    def build(self, inputs):

        # Initial
        # 32, 32, 3 -> 32, 32, 64
        x = self.ConvBn_Block(inputs, num_filters=64, kernel_size=(7, 7), strides=(1, 1), block_name='0') # change strides

        # if self.use_avg:
        #     x = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x)
        # else:
        #     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Basic blocks
        # 32, 32, 64 -> 32, 32, 64
        x = self.build_basic_block(x, filter_num=64, blocks=self.layers_dims[0], stride=1, module_name='module_0')
        # 32, 32, 64 -> 32, 32, 128
        x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=1, module_name='module_1')
        # 32, 32, 128 -> 16, 16, 256
        x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=2, module_name='module_2')
        # 16, 16, 256 -> 8, 8, 512
        x = self.build_basic_block(x, filter_num=512, blocks=self.layers_dims[3], stride=2, module_name='module_3')

        # Top
        # 8, 8, 512 -> 1, 1, 512
        x = GlobalAveragePooling2D()(x)

        if self.add_spike_layer:
            x = spike_counter(self.count_spikes, self.add_loss,
                              assign_inputs=self.assign_inputs,
                              name='prediction_spikes')(x)
        out = Dense(self.nb_classes, name='prediction')(x)
        out = Softmax()(out)
        model = Model(inputs=inputs, outputs=out, name='ResNet')

        return model

def build_ResNet(NetName, nb_classes=10,
                 is_slim=True, input_shape=(32, 32, 3),
                 use_bn=True, use_bias=False, use_avg=False):

    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3]}

    inputs = Input(shape=input_shape)
    if is_slim:
        ResNet_model_c = ResNet_slim(layers_dims=ResNet_Config[NetName], nb_classes=nb_classes,
                                    use_bn=use_bn, use_bias=use_bias, use_avg=use_avg)
    else:
        ResNet_model_c = ResNet(layers_dims=ResNet_Config[NetName], nb_classes=nb_classes,
                                use_bn=use_bn, use_bias=use_bias, use_avg=use_avg)
    ResNet_model = ResNet_model_c(inputs)
    return ResNet_model


def main():
    model = build_ResNet('ResNet18', 1000)
    model.summary()


if __name__ == '__main__':
    main()