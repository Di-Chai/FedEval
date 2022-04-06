import json
import os
import pdb
import hickle
import numpy as np
import tensorflow as tf

from FedEval.run_util import _load_config, _save_config
from FedEval.config import ConfigurationManager, Role
from FedEval.role import Client, ClientContextManager
from FedEval.model import LeNet
from FedEval.dataset import femnist
from FedEval.utils import ParamParser

config_path = 'configs/debug'

c1, c2, c3 = _load_config(config_path)

configs = ConfigurationManager(c1, c2, c3)

client_data_name = [os.path.join(c1['data_dir'], e) for e in os.listdir(c1['data_dir']) if e.startswith('client')]
client_data_name = sorted(client_data_name, key=lambda x: int(x.split('_')[-1].strip('.pkl')))
client_data = []
for data_name in client_data_name:
    with open(data_name, 'r') as f:
        client_data.append(hickle.load(f))

x_train = np.concatenate([e['x_train'] for e in client_data], axis=0)
y_train = np.concatenate([e['y_train'] for e in client_data], axis=0)
x_val = np.concatenate([e['x_val'] for e in client_data], axis=0)
y_val = np.concatenate([e['y_val'] for e in client_data], axis=0)
x_test = np.concatenate([e['x_test'] for e in client_data], axis=0)
y_test = np.concatenate([e['y_test'] for e in client_data], axis=0)
del client_data

parameter_parser = ParamParser()
ml_model = parameter_parser.parse_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
ml_model.fit(
    x=x_train, y=y_train, validation_data=(x_val, y_val),
    batch_size=len(x_train), epochs=3000,
    callbacks=[early_stop]
)
