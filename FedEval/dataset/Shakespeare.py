import os
import json
import numpy as np
import tensorflow as tf
from .FedDataBase import FedData, shuffle
from functools import reduce


class shakespeare(FedData):
    def load_data(self):
        with open(os.path.join(self.data_dir, 'shakespeare', 'all_data.json'), 'r') as f:
            data = json.load(f)

        self.chars = '\n,  , !, ", &, \', (, ), ,, -, ., 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, :, ;, >, ?, A, B, C, D,' \
                     ' E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, [, ], a, b, c, d,' \
                     ' e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, }'
        self.chars = self.chars.split(', ')
        self.chars = {self.chars[i]: i for i in range(len(self.chars))}

        def process_sentences(user_data):
            results = []
            for record in user_data:
                results.append([self.chars[e] for e in list(record)])
            return results

        assert 0 < self.num_clients <= 1121, \
            f"Shakespeare has maximum 1121 clients, received parameter num_clients={self.num_clients}"

        # selected_clients = set(
        #     sorted(data['users'], key=lambda e: data['num_samples'][data['users'].index(e)], reverse=True)
        #     [:self.num_clients]
        # )
        # Random Sample
        selected_clients = np.random.choice(data['users'], size=self.num_clients, replace=False)

        total_num_samples = sum(data['num_samples'])

        x, y = [], []
        self.identity = []
        for i in range(len(data['users'])):
            # set constraint to the number of data samples
            if data['users'][i] not in selected_clients:
                continue
            x += process_sentences(data['user_data'][data['users'][i]]['x'])
            y += process_sentences(data['user_data'][data['users'][i]]['y'])
            self.identity.append(data['num_samples'][i])

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Fix num class
        self.num_class = 80

        if len(y.shape) == 1 or y.shape[-1] == 1:
            y = tf.keras.utils.to_categorical(y, self.num_class)

        print('#' * 40)
        print(f'# Data info, total samples {total_num_samples}, selected clients {self.num_clients}, '
              f'selected samples {sum(self.identity)} Ratio {sum(self.identity) / total_num_samples}')
        print('#' * 40)

        return x, y

