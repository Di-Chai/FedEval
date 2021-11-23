import os
import numpy as np
import tensorflow as tf
from .FedDataBase import FedData


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


class ml25m_matrix(FedData):
    def load_data(self):
        ranking=np.load('ranking.npy').T
        x=np.zeros((1001,59047)) #162542 users 59047 movies
        for item in ranking:
            #print(item[0],item[1],item[2])
            x[int(item[0])][int(item[1])]=item[2]
            if item[0]>=1000:
                break
        y = np.zeros((x.shape[0],1), dtype=int)
        self.num_class = np.max(y) + 1
        y = tf.keras.utils.to_categorical(y, self.num_class)
        print("finish loading ")
        return x,y


