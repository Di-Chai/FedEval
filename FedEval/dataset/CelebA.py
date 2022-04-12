import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .FedDataBase import FedData, shuffle


class celeba(FedData):
    """
    CelebA Median: Top 1000 clients
    """
    def load_data(self):
        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'celeba')

        image_path = os.path.join(data_path, 'images-small')

        assert 0 < self.num_clients <= 9343, \
            f"CelebA has maximum 9343 clients, received parameter num_clients={self.num_clients}"

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
        selected_identity = sorted(all_identity, key=lambda e: len(all_identity_dict[e]), reverse=True)
        selected_identity = selected_identity[:self.num_clients]

        total_num_samples = sum([len(all_identity_dict[e]) for e in all_identity])

        x = []
        y = []
        self.identity = []
        for si in selected_identity:
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

        print('#' * 40)
        print(f'# Data info, total samples {total_num_samples}, selected clients {self.num_clients}, '
              f'selected samples {sum(self.identity)} Ratio {sum(self.identity) / total_num_samples}')
        print('#' * 40)

        return x, y
