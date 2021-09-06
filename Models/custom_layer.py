# -*- coding: utf-8 -*-
# @Time    : 2021/9/2 22:26
# @Author  : Zeqi@@
# @FileName: custom_layer.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Layer

class spike_counter(Layer):
    def __init__(self, count_spikes, add_loss, assign_inputs=True, name='spike', **kwargs):
        super(spike_counter, self).__init__()
        self._name = name
        self.count_spikes = count_spikes
        self.add_l1_loss = add_loss
        self.assign_inputs = assign_inputs
        self.loss_mode = kwargs.get('loss_mode', 'l1')  # hoyer or l1
        self.verbose = kwargs.get('verbose', False)

    def build(self, input_shape):
        self._alpha = tf.Variable(name='alpha',
                                  initial_value=0.0,
                                  trainable=False,
                                  dtype=tf.float32)

        self._spikes = tf.Variable(name='spikes',
                                   initial_value=0.0,
                                   trainable=False,
                                   dtype=tf.float32)

        self._spikes_total = tf.Variable(name='spikes_all',
                                         initial_value=0.0,
                                         trainable=False,
                                         dtype=tf.float32)

        self._value = tf.Variable(name='value',
                                  shape=tf.TensorShape(None),
                                  initial_value=0.0,
                                  trainable=False,
                                  dtype=tf.float32)

        super(spike_counter, self).build(input_shape)

    def get_spikes(self):
        return self._spikes, self._spikes_total

    @property
    def layer_inputs(self):
        return self._value

    def call(self, inputs, training=None, *args, **kwargs):
        out = inputs

        if self.verbose:
            print(' ')
            print('in spikes: ', out.shape)
        if self.assign_inputs:
            self._value.assign(out)

        spikes_batch_fm = tf.reduce_mean(tf.cast(out != 0, tf.float32), axis=0)
        total_spikes_batch_fm = tf.reduce_mean(tf.add(tf.cast(out == 0, tf.float32), tf.cast(out != 0, tf.float32)), axis=0)

        if self.verbose:
            print('spikes_batch_fm: ', spikes_batch_fm.shape)
            print('total_spikes_batch_fm: ', total_spikes_batch_fm.shape)

        if self.count_spikes:
            self._spikes.assign(tf.reduce_sum(spikes_batch_fm))
            self._spikes_total.assign(tf.reduce_sum(total_spikes_batch_fm))

        if self.loss_mode == 'l1':
            # layer_loss = tf.reduce_sum(tf.abs(out)) * self._alpha
            if len(out.shape) < 4:
                layer_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(out), 1)) * self._alpha
            else:
                layer_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(out), [1, 2, 3])) * self._alpha
            # layer_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(out), axis=0)) * self._alpha
        elif self.loss_mode == 'hoyer':
            # sum(abs(v))^2/sum(v^2)
            # mean_activation = tf.reduce_mean(tf.abs(out), axis=0)
            if len(out.shape) < 4:  # dense layer
                top = tf.pow(tf.reduce_sum(tf.abs(out), 1), 2)
                bot = tf.reduce_sum(tf.pow(out, 2), 1)
                layer_loss = tf.reduce_mean(top / bot) * self._alpha
            else:
                top = tf.pow(tf.reduce_sum(tf.abs(out), [1, 2, 3]), 2)
                bot = tf.reduce_sum(tf.pow(out, 2), [1, 2, 3])
                layer_loss = tf.reduce_mean(top / bot) * self._alpha

        if self.add_l1_loss:
            self.add_loss(layer_loss)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(spike_counter, self).get_config()
        config.update({'count_spikes': self.count_spikes})
        config.update({'add_l1_loss': self.add_l1_loss})
        config.update({'assign_inputs': self.assign_inputs})
        return config
