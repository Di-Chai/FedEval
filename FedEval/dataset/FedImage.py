import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .FedDataBase import FedData, shuffle


class mnist(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]
        return x, y


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
        return x, y


class cifar100(FedData):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]
        return x, y


class femnist(FedData):
    def load_data(self):
        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'femnist')
        data_files = [e for e in os.listdir(data_path) if e.endswith('.json')]
        data_files = sorted(data_files, key=lambda x: int(x.strip('.json').split('_')[-1]))

        num_required_files = int(np.ceil(self.num_clients / 100))
        required_file = data_files[:num_required_files]

        data = []
        for file in required_file:
            with open(os.path.join(data_path, file), 'r') as f:
                data.append(json.load(f))

        x = []
        y = []
        identity = []
        for d in data:
            for u_id, xy in d['user_data'].items():
                tmp_x, tmp_y = shuffle(xy['x'], xy['y'])
                x.append(tmp_x)
                y.append(tmp_y)
                identity.append(len(xy['x']))
                if len(identity) >= self.num_clients:
                    break
        x = np.concatenate(x, axis=0).astype(np.float32).reshape([-1, 28, 28, 1])
        y = np.concatenate(y, axis=0).astype(np.int32)
        self.identity = identity
        # [1000, 1500, 2000, ...]
        # sum(self.identity) = x.shape[0]

        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]

        return x, y


class celeba(FedData):
    def load_data(self):
        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'celeba')

        image_path = os.path.join(data_path, 'images-small')

        with open(os.path.join(data_path, 'list_attr_celeba.csv'), 'r') as f:
            y = [e.strip(' \n').split(',') for e in f.readlines()]
            header, y = y[0], y[1:]

        # Here we only user Smiling as label
        label_index = header.index('Smiling')
        y_dict = dict([(e[0], int((int(e[label_index]) + 1) / 2)) for e in y])

        with open(os.path.join(data_path, 'identity_CelebA.txt'), 'r') as f:
            identity = [e.strip(' \n').split(' ')[::-1] for e in f.readlines()]

        all_identity = list(set([e[0] for e in identity]))
        all_identity_dict = {}
        for e in identity:
            all_identity_dict[e[0]] = all_identity_dict.get(e[0], []) + [e[1]]
        all_identity = sorted(all_identity, key=lambda x: int(x))
        selected_identity = []
        for e in all_identity:
            if len(all_identity_dict[e]) >= 5:
                selected_identity.append(e)

        x = []
        y = []
        self.identity = []
        for si in selected_identity[:self.num_clients]:
            local_images = all_identity_dict[si]
            np.random.shuffle(local_images)
            for img in local_images:
                x.append(plt.imread(os.path.join(image_path, img)))
                y.append(y_dict[img])
            self.identity.append(len(local_images))

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if len(y.shape) == 1 or y.shape[-1] == 1:
            self.num_class = np.max(y) + 1
            y = tf.keras.utils.to_categorical(y, self.num_class)
        else:
            self.num_class = y.shape[-1]

        return x, y