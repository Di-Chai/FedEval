import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_wrapper.preprocess import SplitData
from tf_wrapper.train import MiniBatchTrain

np.random.seed(1)


class FedImage:

    @property
    def load_celeba(self):

        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'celeba')

        image_path = os.path.join(data_path, 'images-small')

        with open(os.path.join(data_path, 'list_attr_celeba.csv'), 'r') as f:
            y = [e.strip(' \n').split(',') for e in f.readlines()]
            header, y = y[0], y[1:]

        # Here we only user Smiling as label
        label_index = header.index('Smiling')
        y_dict = dict([(e[0], int((int(e[label_index])+1)/2)) for e in y])

        with open(os.path.join(data_path, 'identity_CelebA.txt'), 'r') as f:
            identity = [e.strip(' \n').split(' ')[::-1] for e in f.readlines()]

        all_identity = list(set([e[0] for e in identity]))
        all_identity_dict = {}
        for e in identity:
            all_identity_dict[e[0]] = all_identity_dict.get(e[0], []) + [e[1]]
        all_identity = sorted(all_identity, key=lambda x: int(x))
        selected_identity = []
        for e in all_identity:
            if len(all_identity_dict[e]) > 20:
                selected_identity.append(e)

        x = []
        y = []
        self.identity = []
        for si in selected_identity[:self.num_clients]:
            local_images = all_identity_dict[si]
            for img in local_images:
                x.append(plt.imread(os.path.join(image_path, img)))
                y.append(y_dict[img])
            self.identity.append(len(local_images))

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        return x, y

    def load_femnist(self):

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
                tmp_x, tmp_y = MiniBatchTrain.shuffle(xy['x'], xy['y'])
                x.append(tmp_x)
                y.append(tmp_y)
                identity.append(len(xy['x']))
                if len(identity) >= self.num_clients:
                    break
        x = np.concatenate(x, axis=0).astype(np.float32).reshape([-1, 28, 28, 1])
        y = np.concatenate(y, axis=0).astype(np.int32)
        self.identity = identity

        return x, y

    def __init__(self, dataset, num_clients, data_dir, custom_data_loader=None,
                 flatten=False, normalize=True, train_val_test=(0.8, 0.1, 0.1,)):

        self.dataset = dataset
        self.num_clients = num_clients
        self.custom_data_loader = custom_data_loader
        self.data_dir = data_dir
        self.train_val_test = train_val_test

        self.local_path = os.path.dirname(os.path.abspath(__file__))
        self.identity = None

        if hasattr(tf.keras.datasets, dataset):
            (x_train, y_train), (x_test, y_test) = eval('tf.keras.datasets.%s.load_data()' % dataset)
            self.x = np.concatenate((x_train, x_test), axis=0)
            self.y = np.concatenate((y_train, y_test), axis=0)
            if dataset == 'mnist':
                self.x = np.expand_dims(self.x, axis=-1)
        elif dataset == 'femnist':
            self.x, self.y = self.load_femnist()
        elif dataset == 'celeba':
            self.x, self.y = self.load_celeba
        else:
            assert custom_data_loader
            self.x, self.y = custom_data_loader()

        if normalize:
            self.x = self.x / np.max(self.x)

        if flatten:
            self.x = np.reshape(self.x, [-1, np.prod(self.x.shape[1:])])

        if len(self.y.shape) == 1 or self.y.shape[-1] == 1:
            self.num_class = np.max(self.y) + 1
            self.y = tf.keras.utils.to_categorical(self.y, self.num_class)
        else:
            self.num_class = self.y.shape[-1]

        if data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)

    def iid_data(self, sample_size=300, save_file=True):

        # Temporally use, to guarantee the test set in iid/non-iid setting are the same
        if self.identity is not None:
            local_dataset = self.non_iid_data(non_iid_class=1, strategy='natural', sample_size=sample_size,
                                              shared_data=0, save_file=False)
        else:
            local_dataset = self.non_iid_data(non_iid_class=self.num_class, strategy='average', sample_size=sample_size,
                                              shared_data=0, save_file=False)
        # Transfer non-iid to iid
        x_train_all = np.concatenate([e['x_train'] for e in local_dataset], axis=0)
        y_train_all = np.concatenate([e['y_train'] for e in local_dataset], axis=0)

        x_train_all, y_train_all = MiniBatchTrain.shuffle(x_train_all, y_train_all)
        train_size = int(len(x_train_all) / len(local_dataset))

        for i in range(len(local_dataset)):
            local_dataset[i]['x_train'] = x_train_all[i*train_size: (i+1)*train_size]
            local_dataset[i]['y_train'] = y_train_all[i*train_size: (i+1)*train_size]

        # x, y = MiniBatchTrain.shuffle(self.x, self.y)
        #
        # local_data_size = []
        # for i in range(self.num_clients):
        #     local_data_size.append(sample_size)
        # local_data_size = np.array(local_data_size, dtype=np.int32)
        #
        # local_dataset = []
        # for i in range(self.num_clients):
        #
        #     local_x = x[np.sum(local_data_size[0:i]):np.sum(local_data_size[0:i+1])]
        #     local_y = y[np.sum(local_data_size[0:i]):np.sum(local_data_size[0:i+1])]
        #
        #     local_data = self.generate_local_data(local_x, local_y)
        #     local_dataset.append(local_data)

        if save_file:
            for i in range(len(local_dataset)):
                with open(os.path.join(self.data_dir, 'client_%s.pkl' % i), 'wb') as f:
                    pickle.dump(local_dataset[i], f)

        return local_dataset

    def generate_local_data(self, local_x, local_y, additional_test=None):
        if additional_test is None:
            local_train_x, local_val_x, local_test_x = SplitData.split_data(local_x, self.train_val_test)
            local_train_y, local_val_y, local_test_y = SplitData.split_data(local_y, self.train_val_test)
            return {
                'x_train': local_train_x,
                'y_train': local_train_y,
                'x_val': local_val_x,
                'y_val': local_val_y,
                'x_test': local_test_x,
                'y_test': local_test_y,
            }
        else:
            local_train_x, local_val_x = SplitData.split_data(local_x, self.train_val_test[:2])
            local_train_y, local_val_y = SplitData.split_data(local_y, self.train_val_test[:2])
            result = {
                'x_train': local_train_x,
                'y_train': local_train_y,
                'x_val': local_val_x,
                'y_val': local_val_y,
            }
            result.update(additional_test)
            return result

    def non_iid_data(self, non_iid_class=2, strategy='average', sample_size=300, shared_data=0, save_file=True):

        local_dataset = []

        if strategy == 'natural':

            if self.identity is None:
                raise AttributeError('Selected dataset has no identity')

            for i in range(self.num_clients):
                index_start = sum(self.identity[:i])
                index_end = sum(self.identity[:i+1])
                local_x = self.x[index_start: index_end]
                local_y = self.y[index_start: index_end]
                # Temporally Remove
                # local_x, local_y = MiniBatchTrain.shuffle(local_x, local_y)
                local_dataset.append(self.generate_local_data(local_x=local_x, local_y=local_y))

        else:
            sample_size = min(sample_size, int(len(self.x) / self.num_clients))

            train_size = int(sample_size * self.train_val_test[0])
            val_size = int(sample_size * self.train_val_test[1])
            test_size = int(sample_size * self.train_val_test[2])

            manual_val_x = self.x[-(val_size+test_size)*self.num_clients:-test_size*self.num_clients]
            manual_val_y = self.y[-(val_size+test_size)*self.num_clients:-test_size*self.num_clients]
            manual_test_x = self.x[-test_size*self.num_clients:]
            manual_test_y = self.y[-test_size*self.num_clients:]

            xy = list(zip(self.x[:-(val_size+test_size)*self.num_clients],
                          self.y[:-(val_size+test_size)*self.num_clients]))
            sorted_xy = sorted(xy, key=lambda x: np.argmax(x[1]), reverse=False)
            x = np.array([e[0] for e in sorted_xy], dtype=np.float32)
            y = np.array([e[1] for e in sorted_xy], dtype=np.float32)

            num_of_each_class = y.sum(0)
            class_pointer = np.array([int(np.sum(num_of_each_class[0:i])) for i in range(self.num_class)])

            # manual set test set
            class_size = int(train_size / non_iid_class)

            for i in range(self.num_clients):
                choose_class = np.random.choice(range(self.num_class), non_iid_class, replace=False)
                if strategy == 'average':
                    local_class_size_mask = np.zeros([self.num_class], dtype=int)
                    local_class_size_mask[choose_class] = 1
                    local_class_size = class_size * local_class_size_mask
                elif strategy.startswith('gaussian'):
                    local_class_size = []
                    scale = float(strategy.split('-')[-1])
                    for c in choose_class:
                        gaussian_choose = np.random.normal(loc=c, scale=scale, size=class_size)
                        local_class_size.append(np.eye(self.num_class)[np.round(gaussian_choose).astype(int) %
                                                                       self.num_class].sum(0))
                    local_class_size = np.sum(local_class_size, axis=0, dtype=int)
                else:
                    raise ValueError('strategy name error')
                local_x, local_y = [], []
                for j in range(self.num_class):
                    local_x.append(x[class_pointer[j]:class_pointer[j]+local_class_size[j]])
                    local_y.append(y[class_pointer[j]:class_pointer[j]+local_class_size[j]])
                    class_pointer[j] += local_class_size[j]
                local_x = np.concatenate(local_x, axis=0)
                local_y = np.concatenate(local_y, axis=0)
                local_dataset.append({
                    'x_train': local_x, 'y_train': local_y,
                    'x_val': manual_val_x[i * val_size: (i+1) * val_size],
                    'y_val': manual_val_y[i * val_size: (i+1) * val_size],
                    'x_test': manual_test_x[i * test_size: (i + 1) * test_size],
                    'y_test': manual_test_y[i * test_size: (i + 1) * test_size],
                })

        if shared_data > 0:
            shared_train_x = []
            shared_train_y = []
            for i in range(len(local_dataset)):
                # already shuffled
                shared_train_x.append(local_dataset[i]['x_train'][:int(shared_data / self.num_clients)])
                shared_train_y.append(local_dataset[i]['y_train'][:int(shared_data / self.num_clients)])
            shared_train_x = np.concatenate(shared_train_x, axis=0)
            shared_train_y = np.concatenate(shared_train_y, axis=0)
            for i in range(len(local_dataset)):
                local_dataset[i]['x_train'] = np.concatenate((local_dataset[i]['x_train'], shared_train_x), axis=0)
                local_dataset[i]['y_train'] = np.concatenate((local_dataset[i]['y_train'], shared_train_y), axis=0)

        if save_file:
            for i in range(len(local_dataset)):
                with open(os.path.join(self.data_dir, 'client_%s.pkl' % i), 'wb') as f:
                    pickle.dump(local_dataset[i], f)

        return local_dataset
