import os
import argparse
import socket


from FedEval.run_util import run
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


"""
Tuned learning rate:
CelebA: 
    FedSGD: 0.03
    LocalCentral: 0.001
FEMNIST:
    FedAvg: 0.3
    FedSGD: 0.009
    FedSTC: 0.005
MNIST:
    FedAvg: 0.7
    FedOpt: 0.5
    FedSGD: 0.07
    FedSTC: 0.3
    FedProx: 0.5
    LocalCentral: 0.5

"""

fine_tuned_params = {
    'mnist': {
        'FedAvg': {'B': 32, 'C': 0.1, 'E': 10, 'lr': 0.7},
        'FedOpt': {'B': 32, 'C': 0.1, 'E': 10, 'lr': 0.5},
        'FedSTC': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.3},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.07},
        'LocalCentral': {'B': 64, 'C': None, 'E': None, 'lr': 0.5},
        'model': 'MLP'
    },
    'femnist': {
        'FedAvg': {'B': 32, 'C': 0.1, 'E': 10, 'lr': 0.3},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.009},
        'FedSTC': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.005},
        'LocalCentral': {'B': 64, 'C': None, 'E': None, 'lr': 0.5},
        'model': 'LeNet'
    },
    'celeba': {
        'FedAvg': {'B': 32, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.03},
        'LocalCentral': {'B': 64, 'C': None, 'E': None, 'lr': 0.5},
        'model': 'LeNet'
    },
    "semantic140": {
        'FedAvg': {'B': 32, 'C': 0.1, 'E': 10, 'lr': 0.0001},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.05},
        'LocalCentral': {'B': 64, 'C': None, 'E': None, 'lr': 0.5},
        'model': 'StackedLSTM'
    },
    "shakespeare": {
        'FedAvg': {'B': 32, 'C': 0.1, 'E': 10, 'lr': None},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': None},
        'LocalCentral': {'B': 64, 'C': None, 'E': None, 'lr': 0.5},
        'model': 'StackedLSTM'
    }
}

# Inherit
parsed_strategy = {
    'MFedSGD': 'FedSGD',
    'FedSTC': 'FedSGD',
    'MFedAvg': 'FedAvg',
    'FedProx': 'FedAvg',
    'FedOpt': 'FedAvg',
    'FedSCA': 'FedAvg',
}

try:
    if args.strategy in parsed_strategy:
        p = fine_tuned_params[args.dataset][parsed_strategy[args.strategy]]
    else:
        p = fine_tuned_params[args.dataset][args.strategy]
except KeyError:
    print('No params found')
    exit(1)

execution = args.exec
config = args.configs
mode = args.mode
repeat = args.repeat

tune_params = {
    'lr': [1e-4, 3e-4, 5e-4, 7e-4, 9e-4,
           1e-3, 3e-3, 5e-3, 7e-3, 9e-3,
           1e-2, 3e-2, 5e-2, 7e-2, 9e-2,
           1e-1, 3e-1, 5e-1, 7e-1, 9e-1, 1.0]
}

data_config = {
    'dataset': args.dataset, 
    'non-iid': True if args.non_iid.lower() == 'true' else False,
    'sample_size': 600,
    'non-iid-strategy': 'average' if args.dataset == 'mnist' else 'natural', 
    'non-iid-class': args.non_iid_class
}
model_config = {
    'MLModel': {
        'name': fine_tuned_params[args.dataset]['model'],
        'optimizer': {'name': 'sgd', 'lr': p['lr'], 'momentum': 0},
        'loss': 'categorical_crossentropy', 'metrics': ['accuracy'],
    },
    'FedModel': {
        'name': args.strategy,
        'B': p['B'], 'C': p['C'], 'E': p['E'], 'max_rounds': 3000, 'num_tolerance': 100,
        'rounds_between_val': 1,
    }
}
runtime_config = {
    'server': {'num_clients': None}, 
    'log': {'log_dir': args.log_dir}, 
    'docker': {'num_containers': 100, 'enable_gpu': False, 'num_gpu': 0},
    'communication': {
        'limit_network_resource': True, 
        'bandwidth_upload': '10Mbit', 
        'bandwidth_download': '30Mbit',
        'latency': '50ms'
        }
}

##################################################
# Dataset Config
if args.dataset == 'mnist':
    data_config['sample_size'] = 600
    runtime_config['server']['num_clients'] = 100

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

if args.strategy == 'LocalCentral':
    model_config['FedModel']['C'] = 1.0
    model_config['FedModel']['E'] = 3000
    model_config['FedModel']['max_rounds'] = 1

if fine_tuned_params[args.dataset]['model'] == 'StackedLSTM':
    model_config['MLModel']['hidden_units'] = 64

##################################################
# Limit the max_epoch to 100 if doing LR tuning
if args.tune == 'lr':
    if args.strategy != 'LocalCentral':
        model_config['FedModel']['max_rounds'] = 100
    else:
        model_config['FedModel']['E'] = 100
    runtime_config['communication']['limit_network_resource'] = False

# TMP Remove
if args.tune == 'lr' and args.strategy == 'LocalCentral':
    if args.dataset == 'femnist':
        tune_params['lr'] = [7e-3, 9e-3,
           1e-2, 3e-2, 5e-2, 7e-2, 9e-2,
           1e-1, 3e-1, 5e-1, 7e-1, 9e-1, 1.0]

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
    runtime_config['docker']['num_containers'] = 10
    runtime_config['docker']['num_gpu'] = 2

if host_name == "gpu06":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 80
    runtime_config['docker']['num_gpu'] = 8

if host_name == "gpu05":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 40
    runtime_config['docker']['num_gpu'] = 2

if host_name == "ministation":
    runtime_config['docker']['enable_gpu'] = True
    runtime_config['docker']['num_containers'] = 10
    runtime_config['docker']['num_gpu'] = 1

params = {
    'data_config': data_config,
    'model_config': model_config,
    'runtime_config': runtime_config
}

for _ in range(repeat):
    # print('*' * 40)
    # print('Running %s, %s with' % (args.dataset, args.strategy), p)
    # print(args)
    # print('*' * 40)
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
