import os
import yaml
import pickle
import argparse

from FedEval.role import Client
from FedEval.model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 1 Load config
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--path', default='./')

args = args_parser.parse_args()

with open(os.path.join(args.path, '1_data_config.yml'), 'r') as f:
    data_config = yaml.load(f)

with open(os.path.join(args.path, '2_model_config.yml'), 'r') as f:
    model_config = yaml.load(f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'r') as f:
    runtime_config = yaml.load(f)

client_id = os.environ.get('CLIENT_ID', '0')

# 2 Config data
with open(os.path.join(data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
    data = pickle.load(f)

# 3 Config Model
input_params = {
    "inputs_shape": {'x': data['x_train'].shape[1:]},
    "targets_shape": {'y': data['y_train'].shape[1:]},
}
model_config[model_config['Model']].update(input_params)
model_config[model_config['Model']]['lr'] = float(runtime_config['clients']['lr'])
model_config[model_config['Model']]['gpu_device'] = '-1'
model = eval(model_config['Model'] + "(**model_config[model_config['Model']])")

# 4 Config clients
client = Client(server_host=runtime_config['server']['host'],
                server_port=runtime_config['server']['port'],
                model=model,
                train_data={'x': data['x_train'], 'y': data['y_train']},
                val_data={'x': data['x_val'], 'y': data['y_val']},
                test_data={'x': data['x_test'], 'y': data['y_test']},
                local_batch_size=runtime_config['clients']['local_batch_size'],
                local_num_rounds=runtime_config['clients']['local_rounds'],
                upload_name_filter=model_config['upload']['upload_name_filter'],
                upload_sparse=model_config['upload']['upload_sparse'],
                upload_strategy=model_config['upload']['upload_strategy'],
                client_name="Client_%s" % client_id)
