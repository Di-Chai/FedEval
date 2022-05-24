import os
import argparse
import socket

from FedEval.run_util import run_util, ConfigurationManager
from multiprocessing import Process

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--dataset', '-d', type=str)
args_parser.add_argument('--strategy', '-s', type=str)
args_parser.add_argument('--configs', '-c', type=str)
args_parser.add_argument('--mode', '-m', type=str)
args_parser.add_argument('--non_iid', '-i', type=str)
args_parser.add_argument('--non_iid_class', '-n', type=int)
args_parser.add_argument('--tune', '-t', type=str)
args_parser.add_argument('--repeat', '-r', type=int, default=1)
args_parser.add_argument('--log_dir', '-l', type=str, default='log/debug')
args_parser.add_argument('--exec', '-e', type=str)
args = args_parser.parse_args()


fine_tuned_params = {
    'mnist': {
        'Local': {'B': 8192, 'C': None, 'E': None, 'lr': 0.005},
        'Central': {'B': 8192, 'C': None, 'E': None, 'lr': 0.5},
        'FedSGD': {'B': 100000000, 'C': 1.0, 'E': 1, 'lr': 0.5},
        'FedSTC': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},  # Debug
        'FedAvg': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.5},
        'FedProx': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.5},
        'FedOpt': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 3},
        'model': 'MLP'
    },
    'femnist': {
        'Local': {'B': 8192, 'C': None, 'E': None, 'lr': 0.1},
        'Central': {'B': 8192, 'C': None, 'E': None, 'lr': 0.05},
        'FedSGD': {'B': 100000000, 'C': 1.0, 'E': 1, 'lr': 0.1},
        'FedSTC': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedAvg': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedProx': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedOpt': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'model': 'LeNet'
    },
    'celeba': {
        'Local': {'B': 8192, 'C': None, 'E': None, 'lr': 0.05},
        'Central': {'B': 8192, 'C': None, 'E': None, 'lr': 0.05},
        'FedSGD': {'B': 100000000, 'C': 1.0, 'E': 1, 'lr': 0.01},
        'FedSTC': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedAvg': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedProx': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedOpt': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'model': 'LeNet'
    },
    "semantic140": {
        'Local': {'B': 8192, 'C': None, 'E': None, 'lr': 0.1},
        'Central': {'B': 8192, 'C': None, 'E': None, 'lr': 2.5},
        'FedSGD': {'B': 100000000, 'C': 1.0, 'E': 1, 'lr': 0.01},
        'FedSTC': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedAvg': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.005},
        'FedProx': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedOpt': {'B': 128, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'model': 'StackedLSTM'
    },
    "shakespeare": {
        'Local': {'B': 8192 * 2, 'C': None, 'E': None, 'lr': 3},
        'Central': {'B': 8192 * 2, 'C': None, 'E': None, 'lr': 2},
        'FedSGD': {'B': 100000000, 'C': 1.0, 'E': 1, 'lr': 2},  # re
        'FedSTC': {'B': 1024, 'C': 0.1, 'E': 10, 'lr': None},  # re
        'FedAvg': {'B': 1024, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedProx': {'B': 1024, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedOpt': {'B': 1024, 'C': 0.1, 'E': 10, 'lr': 0.5},
        'model': 'StackedLSTM'
    }
}

# Inherit
parsed_strategy = {
    'MFedSGD': 'FedSGD',
    'MFedAvg': 'FedAvg',
}

try:
    if args.strategy in parsed_strategy:
        p: dict = fine_tuned_params[args.dataset][parsed_strategy[args.strategy]]
    else:
        p: dict = fine_tuned_params[args.dataset][args.strategy]
except KeyError:
    p = {}
    print('No params found')
    exit(1)

execution = args.exec
config = args.configs
mode = args.mode
repeat = args.repeat

tune_params = {
    'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
}

data_config = {
    'dataset': args.dataset, 'data_dir': 'data',
    'non-iid': True if args.non_iid.lower() == 'true' else False,
    'sample_size': 0,
    'non-iid-strategy': 'average' if args.dataset == 'mnist' else 'natural', 
    'non-iid-class': args.non_iid_class,
    'random_seed': 0,
}
model_config = {
    'MLModel': {
        'name': fine_tuned_params[args.dataset]['model'],
        'optimizer': {'name': 'sgd', 'lr': p['lr'], 'momentum': 0},
        'loss': 'categorical_crossentropy', 'metrics': ['accuracy'],
    },
    'FedModel': {
        'name': args.strategy,
        'B': p['B'], 'C': p['C'], 'E': p['E'],
        'distributed_evaluate': False,
        'max_rounds': 5000, 'num_tolerance': 100,
        'max_train_clients': 1000, 'max_eval_clients': 1000,
        'rounds_between_val': 1, 'evaluate_ratio': 1.0
    }
}
runtime_config = {
    'server': {'num_clients': 100},
    'log': {'log_dir': args.log_dir}, 
    'docker': {'num_containers': 100, 'enable_gpu': False, 'num_gpu': 0},
    'communication': {
        'limit_network_resource': False,
        'bandwidth_upload': '100Mbit',
        'bandwidth_download': '100Mbit',
        'latency': '50ms',
        'fast_mode': False
        }
}

##################################################
# Dataset Config
# All set to small dataset!

"""
MNIST: 100
FEMnist: 100, 1000, 3500
CelebA: 100, 1000, 9343
Sent140: 100, 1000, 10000, 50579
Shakespeare: 100, 1121
"""

if args.dataset == 'mnist':
    data_config['sample_size'] = 700
    runtime_config['server']['num_clients'] = 100

if args.dataset == 'femnist':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 1000

if args.dataset == 'celeba':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 1000

if args.dataset == 'semantic140':
    data_config['sample_size'] = None
    runtime_config['server']['num_clients'] = 1000
    model_config['MLModel']['embedding_dim'] = 0
    model_config['MLModel']['loss'] = 'binary_crossentropy'
    model_config['MLModel']['metrics'] = ['binary_accuracy']

if args.dataset == 'shakespeare':
    runtime_config['server']['num_clients'] = 100
    data_config['normalize'] = False
    model_config['MLModel']['embedding_dim'] = 8

##################################################
# Strategy Config
if args.strategy == 'MFedSGD' or args.strategy == 'MFedAvg':
    model_config['FedModel']['momentum'] = 0.9

if args.strategy == 'FedProx':
    model_config['FedModel']['mu'] = 0.01

if args.strategy == 'FedOpt':
    model_config['FedModel']['tau'] = 1
    model_config['FedModel']['beta1'] = 0.9
    model_config['FedModel']['beta2'] = 0.99
    model_config['FedModel']['eta'] = 1
    model_config['FedModel']['opt_name'] = 'fedadam'

if args.strategy == 'FedSTC':
    model_config['FedModel']['sparsity'] = 0.01

if args.strategy == 'Central' or args.strategy == 'Local':
    model_config['FedModel']['C'] = None
    model_config['FedModel']['E'] = None
    if execution == 'simulate_central':
        model_config['FedModel']['max_rounds'] = 10000
    else:
        model_config['FedModel']['max_rounds'] = 1000
    # Simulation & Docker
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_gpu'] = 1

if fine_tuned_params[args.dataset]['model'] == 'StackedLSTM':
    model_config['MLModel']['hidden_units'] = 64

if args.strategy == 'FedSGD':
    if execution == 'simulate_fedsgd':
        runtime_config['docker']['enable_gpu'] = True
        runtime_config['docker']['num_gpu'] = 1
        model_config['FedModel']['max_rounds'] = 10000
        # Change the batch size
        if args.dataset == 'mnist':
            model_config['FedModel']['B'] = 8192 * 4
        elif args.dataset == 'femnist':
            model_config['FedModel']['B'] = 8192 * 4
        elif args.dataset == 'celeba':
            model_config['FedModel']['B'] = 8192
        elif args.dataset == 'semantic140':
            model_config['FedModel']['B'] = 8192 * 2
        elif args.dataset == 'shakespeare':
            model_config['FedModel']['B'] = 8192 * 2

##################################################
# Limit the max_epoch to 100 if doing LR tuning
if args.tune == 'lr':
    runtime_config['communication']['fast_mode'] = True
    runtime_config['communication']['limit_network_resource'] = False

    if args.strategy == 'FedSGD' or execution == 'simulate_central':
        if args.dataset == 'shakespeare':
            tune_params['lr'] = [1.0, 1.5, 2, 2.5, 3]
        if args.dataset == 'semantic140':
            tune_params['lr'] = [1.0, 1.5, 2, 2.5, 3]
    
    if args.strategy == 'FedAvg':
        if args.dataset == 'semantic140':
            tune_params['lr'] = [0.005, 0.01, 0.05, 0.1]

    if execution == 'simulate_local':
        if args.dataset == 'shakespeare':
            tune_params['lr'] = [3.5, 4, 4.5, 5, 5.5, 6]
        if args.dataset == 'semantic140':
            tune_params['lr'] = [5e-2, 1e-1, 5e-1, 1.0]

    if args.strategy == 'FedOpt':
        if args.dataset == 'mnist':
            tune_params['lr'] = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    if args.strategy == 'FedSTC':
        tune_params['lr'] = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]

##################################################
# Hardware Config

# def get_gpu_info_and_determin_container_nums():
#     import pynvml
#     pynvml.nvmlInit()
#     num_gpus = pynvml.nvmlDeviceGetCount()
#     num_containers = []
#     for i in range(num_gpus):
#         num_containers.append(
#             int(pynvml.nvmlDeviceGetMemoryInfo(
#             pynvml.nvmlDeviceGetHandleByIndex(i)).free 
#             / 2**30)
#         )
#     num_containers = min(num_containers) * len(num_containers)
#     return num_containers, num_gpus

host_name = socket.gethostname()

if host_name == "workstation":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 8
    runtime_config['docker']['num_gpu'] = 2

if host_name == "gpu06":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 20
    runtime_config['docker']['num_gpu'] = 8

if host_name == "gpu05":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 20
    runtime_config['docker']['num_gpu'] = 2

if host_name == "gpu01":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 20
    runtime_config['docker']['num_gpu'] = 2

if host_name == "gpu02":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 20
    runtime_config['docker']['num_gpu'] = 2

if host_name == "ministation":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 10
    runtime_config['docker']['num_gpu'] = 1

if host_name == "mac":
    runtime_config['docker']['enable_gpu'] = False
    runtime_config['docker']['num_containers'] = 8
    runtime_config['docker']['num_gpu'] = 0

params = {
    'data_config': data_config,
    'model_config': model_config,
    'runtime_config': runtime_config
}

if __name__ == '__main__':

    for seed in range(repeat):
        params['data_config']['random_seed'] = seed
        if args.tune is None:
            pro = Process(target=run_util, args=(execution, mode, config), kwargs=params)
            pro.start()
            pro.join()
        else:
            print('Tuning', args.tune)
            if args.tune == 'lr':
                for lr in tune_params['lr']:
                    params['model_config']['MLModel']['optimizer']['lr'] = lr
                    pro = Process(target=run_util, args=(execution, mode, config), kwargs=params)
                    pro.start()
                    pro.join()
            else:
                raise ValueError('Unknown tuning params', args.tune)
    