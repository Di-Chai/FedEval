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
        return x.astype(np.float64), y


class mnist_matrix(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = np.reshape(x, [-1, np.prod(x.shape[1:])])
        self.num_class = np.max(y) + 1
        y = tf.keras.utils.to_categorical(y, self.num_class)
        return x.astype(np.float64), y


def load_synthetic(m, n, alpha):
    # Reference: https://github.com/andylamp/federated_pca/blob/master/synthetic_data_gen.m
    k = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    Sigma = np.array(list(range(1, k + 1))).astype(np.float64) ** -alpha
    V = np.random.randn(k, n)
    Y = (U @ np.diag(Sigma) @ V) / np.sqrt(n - 1)
    yn = np.max(np.sqrt(np.sum(Y ** 2, axis=1)))
    Y /= yn
    return Y, None


def load_synthetic_large_scale(m, n, alpha):
    # num_feature: m
    # num_sample: n
    result_memmap = np.memmap(
        filename=os.path.join(ConfigurationManager().data_config.dir_name, f'synthetic_large_scale_{m}_{n}.npy'),
        dtype=np.float64, shape=(m, n), mode='write'
    )
    k = min(m, n)
    u, _ = np.linalg.qr(np.random.randn(m, m))
    sigma = np.array(list(range(1, k + 1))).astype(np.float64) ** -alpha
    u_sigma = u @ np.diag(sigma)
    step_size = 100000
    d = np.zeros(m)
    for i in range(0, n, step_size):
        print('Generating large scale data step', i)
        tmp_n_size = min(step_size, n-i)
        tmp = u_sigma @ np.random.randn(k, tmp_n_size) / np.sqrt(n - 1)
        result_memmap[:, i:i+tmp_n_size] = tmp
        d += np.sum(tmp ** 2, axis=1)
        result_memmap.flush()
        del tmp
    result_memmap /= np.max(np.sqrt(d))
    result_memmap.flush()
    return result_memmap, None


class synthetic_matrix_horizontal(FedData):
    def load_data(self):
        m = ConfigurationManager().data_config.feature_size
        n = ConfigurationManager().data_config.sample_size * ConfigurationManager().runtime_config.client_num
        alpha = 1.0
        x, y = load_synthetic(m, n, alpha)
        # feature * sample
        x = x.T
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class synthetic_matrix_vertical(FedData):
    def load_data(self):
        m = ConfigurationManager().data_config.feature_size * ConfigurationManager().runtime_config.client_num
        n = int(ConfigurationManager().data_config.sample_size)
        alpha = 1.0
        x, y = load_synthetic(m, n, alpha)
        # sample * feature
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y


class synthetic_matrix_horizontal_memmap(FedData):

    def load_data(self):
        m = ConfigurationManager().data_config.feature_size
        n = ConfigurationManager().data_config.sample_size * ConfigurationManager().runtime_config.client_num
        alpha = 0.1
        x, y = load_synthetic_large_scale(m, n, alpha)
        x = x.T
        y = np.zeros([len(x), 1])
        self.num_class = 1
        return x, y

    def iid_data(self, save_file=True):
        local_dataset_index = super(synthetic_matrix_horizontal_memmap, self).iid_data(save_file=False)
        local_dataset = []
        for i in range(len(local_dataset_index)):
            np.save(
                os.path.join(ConfigurationManager().data_config.dir_name, f'client_{i}_train_x.npy'),
                self.x[local_dataset_index[i][0]]
            )
            local_dataset.append(
                {'x_train': os.path.join(ConfigurationManager().data_config.dir_name, f'client_{i}_train_x.npy')}
            )
        if save_file:
            for i in range(len(local_dataset)):
                with open(os.path.join(self.output_dir, f'client_{i}.pkl'), 'wb') as f:
                    hickle.dump(local_dataset[i], f)
        return local_dataset


class ml25m_matrix(FedData):
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(self.local_path), 'data', 'ml-25m')
        num_users = 162541
        num_movies = 62423
        x = np.zeros([num_users, num_movies], dtype=np.float64)
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
        n_samples = ConfigurationManager().data_config.sample_size
        n_features = ConfigurationManager().data_config.feature_size * ConfigurationManager().runtime_config.client_num
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features*0.9), shuffle=True, n_targets=1,
            noise=1.0, bias=1
        )
        return x.astype(np.float64), y


class wine_lr(FedVerticalMatrix):
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(self.local_path), 'data', 'wine')
        with open(os.path.join(data_dir, 'winequality-red.csv')) as f:
            wine_red = f.readlines()[1:]
            wine_red = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_red]
        with open(os.path.join(data_dir, 'winequality-white.csv')) as f:
            wine_white = f.readlines()[1:]
            wine_white = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_white]
        data = np.array(wine_red + wine_white)
        x = data[:, :-1]
        y = data[:, -1]
        return x.astype(np.float64), y


class ml100k_lr(FedVerticalMatrix):
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(self.local_path), 'data', 'ml-100k')
        item_size = 1682
        user_size = 943
        num_item_selected = 500
        with open(os.path.join(data_dir, 'u.user'), 'r') as f:
            user_attr = f.readlines()
            y = np.array([int(e.strip('\n').split('|')[1]) for e in user_attr])
        with open(os.path.join(data_dir, 'u.data'), 'r') as f:
            ratings = f.readlines()
            ratings = [e.strip('\n').split('\t')[:3] for e in ratings]
        item_acc = np.zeros([item_size])
        for _, i_id, _ in ratings:
            item_acc[int(i_id)-1] += 1
        selected_items = [e[0] for e in sorted(
                [[i+1, item_acc[i]] for i in range(len(item_acc))], key=lambda x: x[1], reverse=True
            )[:num_item_selected]]
        selected_items = sorted(selected_items)
        item_set = set(selected_items)
        x = np.zeros([user_size, num_item_selected])
        for u_id, i_id, rate in ratings:
            if int(i_id) in item_set:
                x[int(u_id)-1][selected_items.index(int(i_id))] = float(rate)
        return x.astype(np.float64), y
