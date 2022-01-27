import os
import argparse
import socket

from FedEval.run_util import run
from multiprocessing import Process

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--dataset', '-d', type=str)
args_parser.add_argument('--configs', '-c', type=str)
args_parser.add_argument('--mode', '-m', type=str)
args_parser.add_argument('--tune', '-t', type=str)
args_parser.add_argument('--repeat', '-r', type=int, default=1)
args_parser.add_argument('--log_dir', '-l', type=str, default='log/debug')
args_parser.add_argument('--exec', '-e', type=str)
args = args_parser.parse_args()

execution = args.exec
config = args.configs
mode = args.mode
repeat = args.repeat

tune_params = {
    'block_size': [1, 10, 100, 1000]
}

data_config = {
    'dataset': args.dataset, 'non-iid': False,
    'sample_size': 600, 'feature_size': 100,
    'normalize': False,
    'non-iid-strategy': None,
    'non-iid-class': None
}
model_config = {
    'FedModel': {
        'name': 'FedSVD',
        'B': None, 'C': 1.0, 'E': None, 'max_rounds': 300000, 'num_tolerance': 10000,
        'block_size': 1000,
        'fedsvd_mode': 'svd',
        'fedsvd_top_k': -1,
        'fedsvd_lr_l2': 0
    }
}
runtime_config = {
    'server': {'num_clients': 10},
    'log': {'log_dir': args.log_dir},
    'docker': {'num_containers': 10, 'enable_gpu': False, 'num_gpu': 0},
    'communication': {
        'limit_network_resource': False,
        'bandwidth_upload': '10Mbit',
        'bandwidth_download': '30Mbit',
        'latency': '50ms'
    }
}

##################################################
# Dataset Config
if args.dataset == 'wine':
    pass

if args.dataset == 'femnist':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 1989

if args.dataset == 'celeba':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 5304

if args.dataset == 'semantic140':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 161
    model_config['MLModel']['embedding_dim'] = 0
    model_config['MLModel']['loss'] = 'binary_crossentropy'
    model_config['MLModel']['metrics'] = ['binary_accuracy']

if args.dataset == 'shakespeare':
    runtime_config['server']['num_clients'] = 1121
    data_config['normalize'] = False
    model_config['MLModel']['embedding_dim'] = 8

params = {
    'data_config': data_config,
    'model_config': model_config,
    'runtime_config': runtime_config
}

if __name__ == '__main__':

    for _ in range(repeat):
        if args.tune is None:
            p = Process(target=run, args=(execution, mode, config, config + '_tmp'), kwargs=params)
            p.start()
            p.join()
        else:
            print('Tuning', args.tune)
            if args.tune == 'lr':
                for lr in tune_params['lr']:
                    params['model_config']['MLModel']['optimizer']['lr'] = lr
                    p = Process(target=run, args=(execution, mode, config, config + '_tmp'), kwargs=params)
                    p.start()
                    p.join()
            else:
                raise ValueError('Unknown tuning params', args.tune)
