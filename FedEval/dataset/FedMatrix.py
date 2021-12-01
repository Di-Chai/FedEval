import os
import hickle
from abc import ABC

import numpy as np
import tensorflow as tf
from .FedDataBase import FedData, shuffle
from ..config import ConfigurationManager
from sklearn.datasets import make_regression


class FedVerticalMatrix(FedData, ABC):

    def non_iid_data(self, *args):
        raise ModuleNotFoundError('FedVerticalMatrix has no non-iid data')

    """Generate datasets for vertical federated learning"""
    def iid_data(self, save_file=True):
        self.x, self.y = shuffle(self.x, self.y)
        # Assume the features are uniformly distributed
        num_samples, num_features = self.x.shape
        num_clients = ConfigurationManager().runtime_config.client_num
        # Recompute the number of features hold by each client
        local_num_features = np.array([int(num_features / num_clients)] * num_clients)
        if num_features % num_clients != 0:
            local_num_features[:num_features % num_clients] += 1

        train_size = int(num_samples * self.train_val_test[0])
        val_size = int(num_samples * self.train_val_test[1])
        test_size = int(num_samples * self.train_val_test[2])

        local_dataset = []
        for i in range(num_clients):
            local_dataset.append({
                'x_train': self.x[:train_size, sum(local_num_features[:i]):sum(local_num_features[:i+1])].T,
                'x_val': [] if val_size == 0 else
                self.x[train_size:train_size+val_size, sum(local_num_features[:i]):sum(local_num_features[:i+1])].T,
                'x_test': [] if test_size == 0 else
                self.x[train_size+val_size:, sum(local_num_features[:i]):sum(local_num_features[:i+1])].T
            })
            if (i+1) == num_clients:
                local_dataset[-1].update({
                    'y_train': self.y[:train_size],
                    'y_val': [] if val_size == 0 else self.y[train_size:train_size+val_size],
                    'y_test': [] if test_size == 0 else self.y[train_size+val_size:],
                })
        if save_file:
            for i in range(len(local_dataset)):
                with open(os.path.join(self.output_dir, f'client_{i}.pkl'), 'wb') as f:
                    hickle.dump(local_dataset[i], f)
        return local_dataset


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
        m = ConfigurationManager().data_config.feature_size
        n = int(ConfigurationManager().data_config.sample_size)
        alpha = 1.0
        x, y = load_synthetic(m, n, alpha)
        # feature * sample
        x = x.T
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class synthetic_matrix_vertical(FedData):
    def load_data(self):
        m = ConfigurationManager().data_config.feature_size
        n = int(ConfigurationManager().data_config.sample_size)
        alpha = 1.0
        # sample * feature
        x, y = load_synthetic(m, n, alpha)
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class ml25m_matrix(FedData):
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(self.local_path), 'data', 'ml-25m')
        num_users = 162541
        num_movies = 62423
        x = np.zeros([num_users, num_movies])
        with open(os.path.join(data_dir, 'movies.csv'), 'r') as f:
            movie_ids = [e.split(',')[0] for e in f.readlines()[1:]]
            movie_ids = {e: movie_ids.index(e) for e in movie_ids}
        with open(os.path.join(data_dir, 'ratings.csv'), 'r') as f:
            ratings = f.readlines()[1:]
            ratings = [e.strip('\n').split(',')[:3] for e in ratings]
        for record in ratings:
            x[int(record[0])-1][movie_ids[record[1]]] = record[2]
        # Split by movies
        x = x.T
        # Dummy infos
        y = np.zeros((x.shape[0], 1), dtype=int)
        self.num_class = 1
        return x, y


class vertical_linear_regression(FedVerticalMatrix):
    def load_data(self):
        n_samples = int(ConfigurationManager().data_config.sample_size)
        n_features = ConfigurationManager().data_config.feature_size
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features*0.9), shuffle=True, n_targets=1
        )
        return x, y
