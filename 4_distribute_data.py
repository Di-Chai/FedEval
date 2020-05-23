import os
import yaml
import argparse
from FedEval.dataset import FedImage

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--path', default='./')

args = args_parser.parse_args()

with open(os.path.join(args.path, '1_data_config.yml'), 'r') as f:
    data_config = yaml.load(f)

with open(os.path.join(args.path, '2_model_config.yml'), 'r') as f:
    model_config = yaml.load(f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'r') as f:
    runtime_config = yaml.load(f)

data = FedImage(dataset=data_config['dataset'],
                data_dir=data_config['data_dir'],  # for saving
                flatten=True if model_config['Model'] == 'MLP' else False,
                normalize=data_config['normalize'],
                train_val_test=data_config['train_val_test'],
                num_clients=runtime_config['clients']['num_clients'])


if data_config['non-iid'] == 0:
    data.iid_data(sample_size=data_config['sample_size'])
else:
    data.non_iid_data(non_iid_class=data_config['non-iid'],
                      strategy=data_config['non-iid-strategy'],
                      shared_data=data_config['shared_data'], sample_size=data_config['sample_size'])