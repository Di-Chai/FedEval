import os
from utils import get_default_params_dict
os.chdir('../')

execution = 'run'
config = 'configs/local'
mode = 'local'
output = 'NonIID.csv'
repeat = 10

trials = [
    # MNIST Non-IID
    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 1, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 2, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 3, 'non-iid-strategy': 'average'},

    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedAvg', 'lr': 5e-4, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 1, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedAvg', 'lr': 5e-4, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 2, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'MLP', 'fed_model': 'FedAvg', 'lr': 5e-4, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 3, 'non-iid-strategy': 'average'},

    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 1, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 2, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': 3, 'non-iid-strategy': 'average'},

    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedAvg', 'lr': 5e-3, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 1, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedAvg', 'lr': 5e-3, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 2, 'non-iid-strategy': 'average'},
    {'dataset': 'mnist', 'ml_model': 'LeNet', 'fed_model': 'FedAvg', 'lr': 5e-3, 'B': 8, 'C': 0.1, 'E': 16,
     'non-iid': True, 'non-iid-class': 3, 'non-iid-strategy': 'average'},

    # FEMNIST Non-IID
    {'dataset': 'femnist', 'ml_model': 'MLP', 'fed_model': 'FedSGD', 'lr': 1e-3, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},
    {'dataset': 'femnist', 'ml_model': 'MLP', 'fed_model': 'FedAvg', 'lr': 1e-4, 'B': 4, 'C': 0.1, 'E': 32,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},
    {'dataset': 'femnist', 'ml_model': 'LeNet', 'fed_model': 'FedSGD', 'lr': 1e-2, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},
    {'dataset': 'femnist', 'ml_model': 'LeNet', 'fed_model': 'FedAvg', 'lr': 5e-4, 'B': 4, 'C': 0.1, 'E': 32,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},

    # CelebA Non-IID
    {'dataset': 'celeba', 'ml_model': 'LeNet', 'fed_model': 'FedSGD', 'lr': 5e-4, 'B': 1000, 'C': 1.0, 'E': 1,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},
    {'dataset': 'celeba', 'ml_model': 'LeNet', 'fed_model': 'FedAvg', 'lr': 5e-4, 'B': 4, 'C': 1.0, 'E': 32,
     'non-iid': True, 'non-iid-class': None, 'non-iid-strategy': 'natural'},
]


params = get_default_params_dict()
params['config'] = config
params['mode'] = mode

if execution == 'run':
    for _ in range(repeat):
        for p in trials:
            params.update(p)
            os.system('python -m FedEval.run_util ' +
                      ' '.join(["--%s %s" % (key, value) for key, value in params.items()]))

if execution == 'stop':
    os.system('python -m FedEval.run_util --mode {} --config {} --exec stop'.format(mode, config))
