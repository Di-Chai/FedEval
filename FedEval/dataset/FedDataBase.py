import os
import copy
import pickle
import numpy as np


def split_data(data, ratio_list):
    '''
    Divide the data based on the given parameter ratio_list.

    Args:
        data(ndarray):Data to be split.
        ratio_list(list):Split ratio, the `data` will be split according to the ratio.
    :return:The elements in the returned list are the divided data, and the
        dimensions of the list are the same as ratio_list.
    :type: list
    '''
    if np.sum(ratio_list) != 1:
        ratio_list = np.array(ratio_list)
        ratio_list = ratio_list / np.sum(ratio_list)
    return [data[int(sum(ratio_list[0:e])*len(data)):
                 int(sum(ratio_list[0:e+1])*len(data))] for e in range(len(ratio_list))]


def shuffle(X, Y):
    '''
    Input (X, Y) pairs, shuffle and return it.
    '''
    xy = list(zip(X, Y))
    np.random.shuffle(xy)
    return np.array([e[0] for e in xy], dtype=np.float32), np.array([e[1] for e in xy], dtype=np.float32)


class FedData:

    def __init__(self, num_clients, output_dir,
                 flatten=False, normalize=False, train_val_test=(0.8, 0.1, 0.1,)):

        self.num_clients = num_clients
        self.output_dir = output_dir
        self.train_val_test = train_val_test

        self.local_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.local_path), 'data')
        self.identity = None
        self.num_class = None

        self.x, self.y = self.load_data()

        assert self.num_class is not None, 'Non-classification tasks will be supported in the future'

        if normalize:
            self.x = self.x / np.max(self.x)

        if flatten:
            self.x = np.reshape(self.x, [-1, np.prod(self.x.shape[1:])])

    def load_data(self):
        raise NotImplementedError('Please implement the load_data function')

    def iid_data(self, sample_size=300, save_file=True):
        # Temporally use, to guarantee the test set in iid/non-iid setting are the same
        if self.identity is not None:
            local_dataset = self.non_iid_data(non_iid_class=1, strategy='natural', sample_size=sample_size,
                                              shared_data=0, save_file=False)
        else:
            local_dataset = self.non_iid_data(non_iid_class=self.num_class, strategy='average', sample_size=sample_size,
                                              shared_data=0, save_file=False)
        # Transfer non-iid to iid
        x_train_all = np.concatenate([copy.deepcopy(e['x_train']) for e in local_dataset], axis=0)
        y_train_all = np.concatenate([copy.deepcopy(e['y_train']) for e in local_dataset], axis=0)

        x_train_all, y_train_all = shuffle(x_train_all, y_train_all)
        train_size = int(len(x_train_all) / len(local_dataset))

        for i in range(len(local_dataset)):
            local_dataset[i]['x_train'] = x_train_all[i * train_size: (i + 1) * train_size]
            local_dataset[i]['y_train'] = y_train_all[i * train_size: (i + 1) * train_size]

        if save_file:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
            for i in range(len(local_dataset)):
                with open(os.path.join(self.output_dir, 'client_%s.pkl' % i), 'wb') as f:
                    pickle.dump(local_dataset[i], f)

        return local_dataset

    def generate_local_data(self, local_x, local_y, additional_test=None):
        if additional_test is None:
            local_train_x, local_val_x, local_test_x = split_data(local_x, self.train_val_test)
            local_train_y, local_val_y, local_test_y = split_data(local_y, self.train_val_test)
            return {
                'x_train': local_train_x,
                'y_train': local_train_y,
                'x_val': local_val_x,
                'y_val': local_val_y,
                'x_test': local_test_x,
                'y_test': local_test_y,
            }
        else:
            local_train_x, local_val_x = split_data(local_x, self.train_val_test[:2])
            local_train_y, local_val_y = split_data(local_y, self.train_val_test[:2])
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
                index_end = sum(self.identity[:i + 1])
                local_x = self.x[index_start: index_end]
                local_y = self.y[index_start: index_end]
                local_x, local_y = shuffle(local_x, local_y)
                local_dataset.append(self.generate_local_data(local_x=local_x, local_y=local_y))

        else:
            sample_size = min(sample_size, int(len(self.x) / self.num_clients))

            train_size = int(sample_size * self.train_val_test[0])
            val_size = int(sample_size * self.train_val_test[1])
            test_size = int(sample_size * self.train_val_test[2])

            manual_val_x = self.x[-(val_size + test_size) * self.num_clients:-test_size * self.num_clients]
            manual_val_y = self.y[-(val_size + test_size) * self.num_clients:-test_size * self.num_clients]
            manual_test_x = self.x[-test_size * self.num_clients:]
            manual_test_y = self.y[-test_size * self.num_clients:]

            xy = list(zip(self.x[:-(val_size + test_size) * self.num_clients],
                          self.y[:-(val_size + test_size) * self.num_clients]))
            sorted_xy = sorted(xy, key=lambda x: np.argmax(x[1]), reverse=False)
            x = np.array([e[0] for e in sorted_xy], dtype=np.float32)
            y = np.array([e[1] for e in sorted_xy], dtype=np.float32)

            num_of_each_class = y.sum(0)
            class_pointer = np.array([int(np.sum(num_of_each_class[0:i])) for i in range(self.num_class)])

            # manual set test set
            non_iid_class = int(non_iid_class)
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
                    local_x.append(x[class_pointer[j]:class_pointer[j] + local_class_size[j]])
                    local_y.append(y[class_pointer[j]:class_pointer[j] + local_class_size[j]])
                    class_pointer[j] += local_class_size[j]
                local_x = np.concatenate(local_x, axis=0)
                local_y = np.concatenate(local_y, axis=0)
                local_dataset.append({
                    'x_train': local_x, 'y_train': local_y,
                    'x_val': manual_val_x[i * val_size: (i + 1) * val_size],
                    'y_val': manual_val_y[i * val_size: (i + 1) * val_size],
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
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
            for i in range(len(local_dataset)):
                with open(os.path.join(self.output_dir, 'client_%s.pkl' % i), 'wb') as f:
                    pickle.dump(local_dataset[i], f)

        return local_dataset
