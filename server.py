import os
import yaml
import argparse
import numpy as np

from FedEval.model import *
from FedEval.role import Server

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

# 2 Config Model
if model_config['Model'] == "MLP":
    input_params = {
        "inputs_shape": {'x': [np.prod(data_config['input_shape'][data_config['dataset']]['image'])]},
        "targets_shape": {'y': data_config['input_shape'][data_config['dataset']]['label']},
    }
else:
    input_params = {
        "inputs_shape": {'x': data_config['input_shape'][data_config['dataset']]['image']},
        "targets_shape": {'y': data_config['input_shape'][data_config['dataset']]['label']},
    }

model_config[model_config['Model']].update(input_params)
model = eval(model_config['Model'] + "(**model_config[model_config['Model']])")

# 3 Config Server
server = Server(host=runtime_config['server']['listen'], port=runtime_config['server']['port'],
                model=model, server_config=runtime_config['server'])
server.start()

