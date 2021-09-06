# -*- coding: utf-8 -*-
# @Time    : 2021/7/3 1:00
# @Author  : Zeqi@@
# @FileName: loss.py.py
# @Software: PyCharm

import tensorflow as tf

class Total_loss(object):
    def __init__(self, model, **kwargs):
        self.model = model

    def count_spikes(self):

        sparse_spikes = 0
        total_spikes = 0

        for i, layer in enumerate(self.model.layers):

            if hasattr(layer, 'get_spikes'):
                sparse_spikes += layer.get_spikes()[0]/10**6
                total_spikes += layer.get_spikes()[1]/10**6

        return sparse_spikes, total_spikes

    def sparse_loss(self, y_true, y_pred):
        return tf.reduce_sum(self.model.losses)

    def nb_spikes(self, y_true, y_pred):
        self.spikes, self.total_spikes = self.count_spikes()
        return self.spikes

    def nb_total_spikes(self, y_true, y_pred):
        return self.total_spikes