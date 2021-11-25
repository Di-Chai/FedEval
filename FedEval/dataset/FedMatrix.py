import os
import numpy as np
import tensorflow as tf
from .FedDataBase import FedData
from ..config import ConfigurationManager


class wine(FedData):
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(self.local_path), 'data', 'wine')
        with open(os.path.join(data_dir, 'winequality-red.csv')) as f:
            wine_red = f.readlines()[1:]
            wine_red = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_red]
        with open(os.path.join(data_dir, 'winequality-white.csv')) as f:
            wine_white = f.readlines()[1:]
            wine_white = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_white]
        x = np.array(wine_red + wine_white)
        y = np.concatenate([np.zeros(len(wine_red), dtype=np.int32),
                            np.ones(len(wine_white), dtype=np.int32)])
        y = np.eye(np.max(y)+1)[y]
        self.num_class = 2
        return x, y


class mnist_matrix(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = np.reshape(x, [-1, np.prod(x.shape[1:])])
        self.num_class = np.max(y) + 1
        y = tf.keras.utils.to_categorical(y, self.num_class)
        return x, y


def load_synthetic(m, n, alpha):
    # Reference: https://github.com/andylamp/federated_pca/blob/master/synthetic_data_gen.m
    k = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    Sigma = np.array(list(range(1, k + 1))).astype(np.float32) ** -alpha
    V = np.random.randn(k, n)
    Y = (U @ np.diag(Sigma) @ V) / np.sqrt(n - 1)
    yn = np.max(np.sqrt(np.sum(Y ** 2, axis=1)))
    Y /= yn
    return Y, None


class synthetic_matrix_horizontal(FedData):
    def load_data(self):
        m = ConfigurationManager().data_config.synthetic_features
        n = int(ConfigurationManager().data_config.sample_size) * \
            int(ConfigurationManager().runtime_config.client_num)
        alpha = 1.0
        x, y = load_synthetic(m, n, alpha)
        # feature * sample
        x = x.T
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class synthetic_matrix_vertical(FedData):
    def load_data(self):
        m = ConfigurationManager().data_config.synthetic_features
        n = int(ConfigurationManager().data_config.sample_size) * \
            int(ConfigurationManager().runtime_config.client_num)
        alpha = 1.0
        # sample * feature
        x, y = load_synthetic(m, n, alpha)
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class ml25m_matrix(FedData):
    def load_data(self):
        ranking=np.load(os.path.join(os.path.dirname(__file__), 'ranking.npy')).T
        x=np.zeros((1001,59047)) #162542 users 59047 movies
        for item in ranking:
            #print(item[0],item[1],item[2])^M
            x[int(item[0])][int(item[1])]=item[2]
            if item[0]>=1000:
                break
        y = np.zeros((x.shape[0],1), dtype=int)
        self.num_class = np.max(y) + 1
        y = tf.keras.utils.to_categorical(y, self.num_class)
        print("finish loading ")
        return x,y
