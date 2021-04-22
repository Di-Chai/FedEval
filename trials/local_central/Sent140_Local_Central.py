import sys
sys.path.append('../..')
from FedEval.run_util import local_central_trial

config = '../../configs/workstation'
repeat = 10
output_name = __file__.split('/')[-1].replace('py', 'csv')
output_name = 'tmp_' + output_name

params = {
    'data_config': {
        'dataset': 'semantic140',
        'non-iid': False,
        'sample_size': 300,
        'normalize': False
    },
    'model_config': {
        'MLModel': {
            'name': 'StackedLSTM', 'embedding_dim': 0, 'hidden_units': 128,
            'loss': 'binary_crossentropy', 'metrics': ['binary_accuracy'],
            'optimizer': {
                'name': 'sgd', 'lr': None, 'momentum': 0
            }
        },
        'FedModel': {
            'name': 'FedAvg', 'B': 128, 'C': 1.0, 'E': 1,
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
    tune_client_lr = [5e-1]
    hidden_units = [64]
    for lr in tune_client_lr:
        for h_u in hidden_units:
            params['model_config']['MLModel']['optimizer']['lr'] = lr
            params['model_config']['MLModel']['hidden_units'] = h_u
            local_central_trial(config=config, output_file=output_name, **params)

