import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .FedDataBase import FedData, shuffle
from functools import reduce


class femnist(FedData):
    """
    FEMnistLarge, 3500 Clients
    FEMnistMedian, 1000 Clients
    FEMnistSmall, 100 Clients
    """

    def load_data(self):
        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'femnist')
        data_files = [e for e in os.listdir(data_path) if e.endswith('.json')]
        data_files = sorted(data_files, key=lambda e: int(e.strip('.json').split('_')[-1]))

        assert 0 < self.num_clients <= 3500, \
            f"FEMnist has maximum 3500 clients, received parameter num_clients={self.num_clients}"

        data = []
        for file in data_files:
            with open(os.path.join(data_path, file), 'r') as f:
                data.append(json.load(f))
        # Get the clients with the highest training samples
        client_sample_number = reduce(
            lambda a, b: a+b, [[[e, len(tmp_d['user_data'][e]['x'])] for e in tmp_d['user_data']] for tmp_d in data])
        client_sample_number = sorted(client_sample_number, key=lambda e: e[1], reverse=True)
        # top_selected_clients = set([e[0] for e in client_sample_number][:self.num_clients])
        # Random Choice
        top_selected_clients = np.random.choice(
            [e[0] for e in client_sample_number], replace=False, size=self.num_clients)
        total_num_samples = sum([e[1] for e in client_sample_number])

        x = []
        y = []
        identity = []
        for d in data:
            for u_id, xy in d['user_data'].items():
                if u_id not in top_selected_clients:
                    continue
                tmp_x, tmp_y = shuffle(xy['x'], xy['y'])
                x.append(tmp_x)
                y.append(tmp_y)
                identity.append(len(xy['x']))
        x = np.concatenate(x, axis=0).astype(np.float32).reshape([-1, 28, 28, 1])
        y = np.concatenate(y, axis=0).astype(np.int32)
        self.identity = identity

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
