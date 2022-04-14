import os
import pdb
import hickle
import argparse
import numpy as np
import tensorflow as tf

from FedEval.run_util import _load_config
from FedEval.config import ConfigurationManager
from FedEval.utils import ParamParser
from FedEval.run import generate_data


def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
        loss = loss_op(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dataset', '-d')

    config_path = 'configs/quickstart'

    c1, c2, c3 = _load_config(config_path)

    configs = ConfigurationManager(c1, c2, c3)

    generate_data(save_file=True)

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

    early_stopping_metric = np.inf
    best_test_metric = None
    patience = 0
    for epoch in range(configs.model_config.max_round_num):
        batch_size = 10000
        batched_gradients = []
        actual_size = []
        for i in range(0, len(x_train), batch_size):
            actual_size.append(min(batch_size, len(x_train) - i))
            batched_gradients.append(
                [e / float(actual_size[-1]) for e in compute_gradients(x_train[i:i+batch_size], y_train[i:i+batch_size])]
            )
        actual_size = np.array(actual_size) / np.sum(actual_size)
        aggregated_gradients = []
        for i in range(len(batched_gradients[0])):
            aggregated_gradients.append(np.average([e[i] for e in batched_gradients], axis=0, weights=actual_size))
        ml_model.optimizer.apply_gradients(zip(aggregated_gradients, ml_model.trainable_variables))
        # Evaluate
        val_log = ml_model.distribute_evaluate(x_val, y_val, verbose=0)
        test_log = ml_model.distribute_evaluate(x_test, y_test, verbose=0)
        print(f'Epoch {epoch} Val Loss {val_log[0]}, Val Acc {val_log[1]}, Test Loss {test_log[0]}, Test Acc {test_log[1]}')
        if val_log[0] < early_stopping_metric:
            early_stopping_metric = val_log[0]
            best_test_metric = test_log
            patience = 0
        else:
            patience += 1
        if patience > configs.model_config.tolerance_num:
            print('Train Finished')
            print(f'Best Test Metric {test_log}')
        del batched_gradients
        del actual_size
