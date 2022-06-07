import copy
import logging
import os
import pdb
import pickle
import hickle
import psutil
from abc import ABCMeta, abstractmethod
from typing import List, Mapping
from functools import reduce
from ..utils import obj_to_pickle_string, pickle_string_to_obj

import numpy as np

from ..config.configuration import ConfigurationManager


def split_data(data, ratio_list) -> List[np.ndarray]:
    '''
    Divide the data based on the given parameter ratio_list.

    Args:
        data(ndarray):Data to be split.
        ratio_list(list):Split ratio, the `data` will be split according to the ratio.

    Returns:
        list: The elements in the returned list are the divided data, and the 
            dimensions of the list are the same as ratio_list.
    '''
    assert len(data) >= len(ratio_list)
    if np.sum(ratio_list) != 1:
        ratio_list = np.array(ratio_list)
        ratio_list = ratio_list / np.sum(ratio_list)
    val_len = max(1, int(ratio_list[1] * len(data)))
    test_len = max(1, int(ratio_list[2] * len(data)))
    return [data[:-(val_len+test_len)], data[-(val_len+test_len):-test_len], data[-test_len:]]


def shuffle(X, Y):
    '''
    Input (X, Y) pairs, shuffle and return it.
    '''
    xy = list(zip(X, Y))
    np.random.shuffle(xy)
    return np.array([e[0] for e in xy], dtype=np.float64), np.array([e[1] for e in xy], dtype=np.float64)


class FedData(metaclass=ABCMeta):

    """By default, FedData produces datasets for horizontal federated learning"""

    def __init__(self):
        cfg_mgr = ConfigurationManager()
        d_cfg, mdl_cfg, rt_cfg = cfg_mgr.data_config, cfg_mgr.model_config, cfg_mgr.runtime_config
        self.output_dir = cfg_mgr.data_dir_name
        self.num_clients = rt_cfg.client_num
        self.train_val_test = d_cfg.data_partition

        # Clear the data if it exists
        # if os.path.isdir(self.output_dir):
        #     import shutil
        #     shutil.rmtree(self.output_dir, ignore_errors=True)
        self._regenerate = True
        if os.path.isdir(self.output_dir):
            client_data_files = [
                e for e in os.listdir(self.output_dir) if e.startswith('client') and e.endswith('.pkl')
            ]
            if len(client_data_files) == rt_cfg.client_num:
                self._regenerate = False
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        self.local_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.local_path), 'data')
        self.identity = None
        self.num_class = None
        self.x, self.y = None, None

        if self._regenerate:
            self._load_and_process_data()

    def _load_and_process_data(self):
        self.x, self.y = self.load_data()
        if ConfigurationManager().data_config.normalized:
            self.x = self.x / np.max(self.x)
        if ConfigurationManager().model_config.ml_method_name == 'MLP':
            self.x = np.reshape(self.x, [-1, np.prod(self.x.shape[1:])])

    @abstractmethod
    def load_data(self):
        raise NotImplementedError('Please implement the load_data function')

    def iid_data(self, save_file=True):
        if self.x is None or self.y is None:
            self._load_and_process_data()
        # Temporally use, to guarantee the test set in iid/non-iid setting are the same
        local_dataset = self.non_iid_data(save_file=False, called_in_iid=True)
        # Transfer non-iid to iid
        train = reduce(lambda x, y: x+y, [e[0] for e in local_dataset])
        np.random.shuffle(train)

        train_size = int(len(train) / len(local_dataset))

        for i in range(len(local_dataset)):
            local_dataset[i][0] = train[i * train_size: (i + 1) * train_size]
            local_dataset[i][0] = train[i * train_size: (i + 1) * train_size]

        if save_file:
            self._save_dataset_files(local_dataset)

        return local_dataset

    @property
    def need_regenerate(self) -> bool:
        return self._regenerate

    def _save_dataset_files(self, dataset: List[Mapping[str, List[np.ndarray]]]) -> None:
        client_data_size = []
        for i in range(len(dataset)):
            train, val, test = dataset[i]
            target = {
                'x_train': self.x[train],
                'y_train': self.y[train],
                'x_val': self.x[val],
                'y_val': self.y[val],
                'x_test': self.x[test],
                'y_test': self.y[test],
            }
            client_data_size.append([len(train), len(val), len(test)])
            hickle.dump(target, os.path.join(self.output_dir, f'client_{i}.pkl'))
            del target
        client_data_size = np.mean(client_data_size, axis=0).astype(int)
        print(f'Average Client sample size: train {client_data_size[0]} val {client_data_size[1]} '
              f'test {client_data_size[2]}')

    # def _generate_local_data(self, local) -> Mapping[str, List[np.ndarray]]:
    #     train, val, test = split_data(local, self.train_val_test)
    #     print(train)
    #     return {
    #         'x_train': self.x[train],
    #         'y_train': self.y[train],
    #         'x_val': self.x[val],
    #         'y_val': self.y[val],
    #         'x_test': self.x[test],
    #         'y_test': self.y[test],
    #     }

    def non_iid_data(self, save_file=True, called_in_iid=False) -> List[Mapping[str, List[np.ndarray]]]:
        if self.x is None or self.y is None:
            self._load_and_process_data()
        d_cfg = ConfigurationManager().data_config
        if called_in_iid:
            strategy = 'average' if self.identity is None else 'natural'
            non_iid_class_num = None if strategy == 'natural' else self.num_class
        else:
            strategy = d_cfg.non_iid_strategy_name
            non_iid_class_num = None if strategy == 'natural' else d_cfg.non_iid_class_num

        local_dataset = []

        if strategy == 'natural':

            if d_cfg.sample_size is not None:
                logging.warning(
                    'Sample size is not working! '
                    'The actual number of data sample held by each client is inherently decided by the data itself. '
                )

            if self.identity is None:
                raise AttributeError('Selected dataset has no identity')

            for i in range(self.num_clients):
                index_start = sum(self.identity[:i])
                index_end = sum(self.identity[:i + 1])
                local = list(range(index_start, index_end))
                local_dataset.append(split_data(local, self.train_val_test))

        else:

            if self.identity is not None:
                raise AttributeError('Selected dataset has identity, Please set Non-IID strategy to "natural" !')

            # TODO(fgh) shared_data = d_cfg.xxx
            sample_size = d_cfg.sample_size
            sample_size = min(sample_size, int(len(self.x) / self.num_clients))

            total_index = list(range(len(self.x)))[:sample_size*self.num_clients]

            train_size = int(sample_size * self.train_val_test[0])
            val_size = int(sample_size * self.train_val_test[1])
            test_size = int(sample_size * self.train_val_test[2])

            train_index = total_index[:train_size*self.num_clients]
            val_index = total_index[train_size*self.num_clients: (train_size+val_size)*self.num_clients]
            test_index = total_index[(train_size+val_size)*self.num_clients:]

            if self.num_class > 1:
                xy = list(zip(self.x[:len(train_index)], self.y[:len(train_index)]))
                xy = sorted(xy, key=lambda x: np.argmax(x[1]), reverse=False)
                self.x[:len(train_index)] = np.array([e[0] for e in xy], dtype=np.float64)
                self.y[:len(train_index)] = np.array([e[1] for e in xy], dtype=np.float64)
                del xy
                # sorted by the label
                train_index = sorted(train_index, key=lambda x: np.argmax(self.y[x]))
                num_of_each_class = self.y[train_index].sum(0)
            else:
                num_of_each_class = [len(self.y[train_index])]

            class_pointer = np.array([int(np.sum(num_of_each_class[0:i])) for i in range(self.num_class)])

            # manual set test set
            class_size = int(train_size / non_iid_class_num)
            class_consumption = np.zeros([self.num_class], dtype=int)
            for i in range(self.num_clients):
                candidate_class = []
                for e in range(self.num_class):
                    if class_consumption[e] < num_of_each_class[e]:
                        candidate_class.append(e)
                choose_class = np.random.choice(
                    candidate_class, min(non_iid_class_num, len(candidate_class)), replace=False
                )
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
                for e in range(self.num_class):
                    if (local_class_size[e] + class_consumption[e]) > num_of_each_class[e]:
                        local_class_size[e] = num_of_each_class[e] - class_consumption[e]
                class_consumption += local_class_size
                assert np.any(class_consumption <= num_of_each_class),\
                    f"{class_consumption} {num_of_each_class} {local_class_size}"
                local = []
                for j in range(self.num_class):
                    local += train_index[class_pointer[j]:class_pointer[j] + local_class_size[j]]
                    class_pointer[j] += local_class_size[j]
                local_dataset.append([
                    local, val_index[i * val_size: (i + 1) * val_size],
                    test_index[i * test_size: (i + 1) * test_size]
                ])

        # if shared_data > 0:
        #     shared_train_x = []
        #     shared_train_y = []
        #     for i in range(len(local_dataset)):
        #         # already shuffled
        #         shared_train_x.append(local_dataset[i]['x_train'][:int(shared_data / self.num_clients)])
        #         shared_train_y.append(local_dataset[i]['y_train'][:int(shared_data / self.num_clients)])
        #     shared_train_x = np.concatenate(shared_train_x, axis=0)
        #     shared_train_y = np.concatenate(shared_train_y, axis=0)
        #     for i in range(len(local_dataset)):
        #         local_dataset[i]['x_train'] = np.concatenate((local_dataset[i]['x_train'], shared_train_x), axis=0)
        #         local_dataset[i]['y_train'] = np.concatenate((local_dataset[i]['y_train'], shared_train_y), axis=0)

        if save_file:
            self._save_dataset_files(local_dataset)

        return local_dataset

