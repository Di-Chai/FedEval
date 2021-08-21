import argparse
import os

from FedEval.run_util import run

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--dataset', '-d', type=str)
args_parser.add_argument('--strategy', '-s', type=str)
args_parser.add_argument('--configs', '-c', type=str)
args_parser.add_argument('--mode', '-m', type=str)
args_parser.add_argument('--non_iid', '-i', type=str)
args_parser.add_argument('--non_iid_class', '-n', type=int)
args_parser.add_argument('--tune', '-t', type=str)
args_parser.add_argument('--repeat', '-r', type=int, default=1)
args_parser.add_argument('--exec', '-e', type=str)
args = args_parser.parse_args()


fine_tuned_params = {
    'mnist': {
        'FedAvg': {'B': 16, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.5},
        'model': 'MLP'
    },
    'femnist': {
        'FedAvg': {'B': 8, 'C': 0.1, 'E': 10, 'lr': 0.1},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.1},
        'model': 'LeNet'
    },
    'celeba': {
        'FedAvg': {'B': 4, 'C': 0.1, 'E': 10, 'lr': 0.05},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.1},
        'model': 'LeNet'
    },
    "semantic140": {
        'FedAvg': {'B': 4, 'C': 0.1, 'E': 10, 'lr': 0.0001},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': 0.05},
        'model': 'StackedLSTM'
    },
    "shakespeare": {
        'FedAvg': {'B': 4, 'C': 0.1, 'E': 10, 'lr': None},
        'FedSGD': {'B': 1000, 'C': 1.0, 'E': 1, 'lr': None},
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
    print('*' * 40)
    print('Running %s, %s with' % (args.dataset, args.strategy), p)
    print(args)
    print('*' * 40)
except KeyError:
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
    'dataset': args.dataset, 'non-iid': True if args.non_iid.lower() == 'true' else False,
    # INF sample size / client
    'sample_size': 300,
    'non-iid-strategy': 'average' if args.dataset == 'mnist' else 'natural', 'non-iid-class': args.non_iid_class
}
model_config = {
    'MLModel': {
        'name': fine_tuned_params[args.dataset]['model'],
        'optimizer': {'name': 'sgd', 'lr': p['lr'], 'momentum': 0},
        'loss': 'categorical_crossentropy', 'metrics': ['accuracy'],
    },
    'FedModel': {
        'name': args.strategy,
        'B': p['B'], 'C': p['C'], 'E': p['E'], 'max_rounds': 3000, 'num_tolerance': 100
    }
}
runtime_config = {'server': {'num_clients': 100}, 'log_dir': 'log/nips', 'docker': {'num_containers': 100}}

if args.strategy == 'MFedSGD' or args.strategy == 'MFedAvg':
    model_config['FedModel']['momentum'] = 0.9

if args.strategy == 'FedProx':
    model_config['FedModel']['mu'] = 0.1

if args.strategy == 'FedOpt':
    model_config['FedModel']['tau'] = 1
    model_config['FedModel']['beta1'] = 0.9
    model_config['FedModel']['beta2'] = 0.99
    model_config['FedModel']['eta'] = 0.1
    model_config['FedModel']['opt_name'] = 'fedadam'

    # TODO: remove the client constraints
    # runtime_config['server']['num_clients'] = 10
    # runtime_config['docker']['num_containers'] = 10

if args.strategy == 'FedSTC':
    model_config['FedModel']['sparsity'] = 0.01

if fine_tuned_params[args.dataset]['model'] == 'StackedLSTM':
    model_config['MLModel']['hidden_units'] = 64

if args.dataset == 'semantic140':
    data_config['normalize'] = False
    model_config['MLModel']['embedding_dim'] = 0
    model_config['MLModel']['loss'] = 'binary_crossentropy'
    model_config['MLModel']['metrics'] = ['binary_accuracy']
    model_config['FedModel']['max_rounds'] = 10000
    model_config['FedModel']['num_tolerance'] = 500

if args.dataset == 'shakespeare':
    data_config['normalize'] = False
    model_config['MLModel']['embedding_dim'] = 8
    tune_params['lr'] = [1e-1, 5e-1, 1.0]

if args.dataset == 'femnist':
    model_config['FedModel']['num_tolerance'] = 500

params = {
    'data_config': data_config,
    'model_config': model_config,
    'runtime_config': runtime_config
}

for _ in range(repeat):
    if args.tune is None:
            run(execution=execution, mode=mode, config=config, new_config=config + '_tmp', **params)
    else:
        if args.tune == 'lr':
            for lr in tune_params['lr']:
                params['model_config']['MLModel']['optimizer']['lr'] = lr
                run(execution=execution, mode=mode, config=config, new_config=config + '_tmp', **params)
        else:
            raise ValueError('Unknown tuning params', args.tune)