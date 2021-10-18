import os
import unittest
from multiprocessing import Process, Manager

from ..dataset import mnist, cifar10, cifar100, celeba, femnist, semantic140, shakespeare
from copy import deepcopy
from ..config.configuration import ConfigurationManager, _DEFAULT_D_CFG, _DEFAULT_RT_CFG
from ..run_util import recursive_update_dict


def get_data(shared_queue, locker, dataset, non_iid, non_iid_class, non_iid_strategy, num_clients):
    test_meta_info = {
        'data_config': {
            'dataset': dataset,
            'non-iid': non_iid,
            'non-iid-class': non_iid_class,
            'non-iid-strategy': non_iid_strategy
        },
        'runtime_config': {
            'server': {'num_clients': num_clients}
        }
    }
    print('Process ID', os.getpid(), str([dataset, non_iid, non_iid_class, non_iid_strategy, num_clients]))
    data_config = recursive_update_dict(deepcopy(_DEFAULT_D_CFG), test_meta_info['data_config'])
    runtime_config = recursive_update_dict(deepcopy(_DEFAULT_RT_CFG), test_meta_info['runtime_config'])
    d_cfg = ConfigurationManager(
        data_config=data_config, runtime_config=runtime_config
    ).data_config
    data = eval(d_cfg.dataset_name)()
    if d_cfg.iid:
        result = data.iid_data(save_file=False)
    else:
        result = data.non_iid_data(save_file=False)

    locker.acquire()
    shared_queue.put(result)
    locker.release()


class DatasetTestCase(unittest.TestCase):

    """Tests for data generation"""

    def check_data_size(self, records, msg):
        print('Start Checking')
        for record in records:
            self.assertTrue(len(record) == 6)
            # Each part should have equal length
            self.assertIn('x_train', record, msg=msg)
            self.assertIn('y_train', record, msg=msg)
            self.assertIn('x_val', record, msg=msg)
            self.assertIn('y_val', record, msg=msg)
            self.assertIn('x_test', record, msg=msg)
            self.assertIn('y_test', record, msg=msg)
            self.assertTrue(len(record['x_train']) == len(record['y_train']) > 0, msg=msg)
            self.assertTrue(len(record['x_val']) == len(record['y_val']) > 0, msg=msg)
            self.assertTrue(len(record['x_test']) == len(record['y_test']) > 0, msg=msg)
        print('Passed')

    def test_data(self):
        manager = Manager()
        shared_queue = manager.Queue()
        locker = manager.Lock()

        data_check_list = [
            ['mnist', False, 1, 'average', 100],
            ['mnist', True, 1, 'average', 100],
            ['mnist', True, 2, 'average', 100],
            ['mnist', True, 3, 'average', 100],

            ['cifar10', False, 1, 'average', 100],
            ['cifar10', True, 1, 'average', 100],
            ['cifar10', True, 2, 'average', 100],
            ['cifar10', True, 3, 'average', 100],

            ['cifar100', False, 1, 'average', 100],
            ['cifar100', True, 1, 'average', 100],
            ['cifar100', True, 2, 'average', 100],
            ['cifar100', True, 3, 'average', 100],

            ['femnist', False, 1, 'average', 1989],
            ['femnist', True, 1, 'natural', 1989],

            ['celeba', False, 1, 'average', 5304],
            ['celeba', True, 1, 'natural', 5304],

            ['semantic140', False, 1, 'average', 161],
            ['semantic140', True, 1, 'natural', 161],

            ['shakespeare', False, 1, 'average', 1121],
            ['shakespeare', True, 1, 'natural', 1121],
        ]

        for check_task in data_check_list:
            p = Process(target=get_data, args=[shared_queue, locker] + check_task)
            p.start()
            p.join()
            self.check_data_size(shared_queue.get_nowait(), msg=str(check_task))

