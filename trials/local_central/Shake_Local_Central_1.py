import sys
sys.path.append('../..')
from FedEval.run_util import local_central_trial

config = '../../configs/workstation'
repeat = 10
output_name = __file__.split('/')[-1].replace('py', 'csv')
output_name = 'tmp_' + output_name

params = {
    'data_config': {
        'dataset': 'shakespeare',
        'non-iid': False,
        'sample_size': 300,
        'normalize': False
    },
    'model_config': {
        'MLModel': {
            'name': 'StackedLSTM', 'embedding_dim': 8, 'hidden_units': 64,
            'optimizer': {
                'name': 'sgd', 'lr': None, 'momentum': 0.9
            }
        },
        'FedModel': {
            'name': 'FedAvg', 'B': 128, 'C': 0.1, 'E': 10,
            'max_rounds': 3000, 'num_tolerance': 100
        }
    },
    'runtime_config': {
        'server': {
            'num_clients': 100
        }
    }
}

for _ in range(repeat):
    # tune_client_lr = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    tune_client_lr = [1e-1, 5e-1, 1.0]
    for lr in tune_client_lr:
        params['model_config']['MLModel']['optimizer']['lr'] = lr
        local_central_trial(config=config, output_file=output_name, **params)

