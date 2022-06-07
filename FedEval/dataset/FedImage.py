import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .FedDataBase import FedData, shuffle


class mnist(FedData):
    def load_data(self):
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mnist')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
            os.path.join(dir_path, 'mnist.npz'))
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]
        return x.astype(np.float64), y.astype(np.int64)


class cifar10(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]
        return x.astype(np.float64), y.astype(np.int64)


class cifar100(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]
        return x.astype(np.float64), y.astype(np.int64)
