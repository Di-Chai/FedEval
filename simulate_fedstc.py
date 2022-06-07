import os
import pdb
import time
import hickle
import multiprocessing

import numpy as np
import tensorflow as tf

from typing import List, Dict
from FedEval.run_util import ConfigurationManager, generate_data, compute_gradients
from FedEval.utils import ParamParser
from FedEval.strategy import FedSTC
from FedEval.aggregater import aggregate_weighted_average
from functools import reduce
from multiprocessing import Pool
from functools import partial


config_manager: ConfigurationManager = ConfigurationManager.from_files('configs/STC')
data_config = config_manager.data_config
model_config = config_manager.model_config
runtime_config = config_manager.runtime_config

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_devices = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)  # TODO(fgh) expose this exception

print('#' * 80)
print('->', 'Starting simulation on FedSTC for parameter tuning, Generating data...')
print('#' * 80)
generate_data(True)
print('#' * 80)
print('->', 'Data Generated')
print('#' * 80)
client_data_name = [
    os.path.join(config_manager.data_dir_name, e) for e in os.listdir(config_manager.data_dir_name) if
    e.startswith('client')
]
client_data_name = sorted(client_data_name, key=lambda x: int(x.split('_')[-1].strip('.pkl')))
client_data_size = []
client_data: List[Dict] = []
for data_name in client_data_name:
    with open(data_name, 'r') as f:
        client_data.append(hickle.load(f))
        client_data_size.append(len(client_data[-1]['x_train']))

client_data_size = np.array(client_data_size, dtype=np.int)
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
test_metric_each_round = []
patience = 0
batch_size = 512

params_shape = [e.shape for e in ml_model.get_weights()]
params_shape_flatten = np.sum([np.prod(e) for e in params_shape])

server_residual = np.zeros([params_shape_flatten])
client_residual = np.zeros([config_manager.runtime_config.client_num, params_shape_flatten])


def stc(input_tensor, sparsity):
    results = np.zeros(input_tensor.shape)
    sparse_size = int(len(input_tensor) * sparsity)
    index = np.argpartition(np.abs(input_tensor), -sparse_size)[-sparse_size:]
    results[index] = input_tensor[index]
    mu = np.mean(results[index])
    results[index] = mu * np.sign(results[index])
    return results


# def compute_gradients(args):
#     x, y = args
#     x = tf.expand_dims(x, 0)
#     y = tf.expand_dims(y, 0)
#     with tf.GradientTape() as tape:
#         y_hat = ml_model(x)
#         loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
#         loss = loss_op(y, y_hat)
#         gradients = tape.gradient(loss, ml_model.trainable_variables)
#     # for i in range(len(gradients)):
#     #     try:
#     #         gradients[i] = gradients[i].numpy()
#     #     except AttributeError:
#     #         gradients[i] = tf.convert_to_tensor(gradients[i]).numpy()
#     return gradients


for epoch in range(model_config.max_round_num):

    # st = time.time()
    # results = []
    # actual_size = []
    # pointer = 0
    # for i in range(0, len(x_train), batch_size):
    #     actual_size.append(min(batch_size, len(x_train) - pointer))
    #     tmp_gradients = tf.vectorized_map(
    #         compute_gradients,
    #         (x_train[pointer:pointer + actual_size[-1]], y_train[pointer:pointer + actual_size[-1]])
    #     )
    #     del tmp_gradients
    #     pointer += actual_size[-1]
    #     print(i)
    # print(f'Per-client gradients cost {time.time() - st}')
    #
    # pdb.set_trace()

    # st = time.time()
    # pointer = 0
    # actual_size = []
    # for i in range(config_manager.runtime_config.client_num):
    #     actual_size.append(client_data_size[i])
    #     client_residual[i] += np.concatenate([
    #         (-config_manager.model_config.learning_rate * e / float(actual_size[-1])).flatten()
    #         for e in compute_gradients(
    #             ml_model, x_train[pointer:pointer + actual_size[-1]],
    #             y_train[pointer:pointer + actual_size[-1]])]
    #     )
    #     pointer += actual_size[-1]
    # print(f'Per-client gradients cost {time.time() - st}')

    st = time.time()
    # stc_sparse = partial(stc, sparsity=config_manager.model_config.stc_sparsity)
    # with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #     w_delta = p.map(stc_sparse, client_residual)
    w_delta = []
    for i in range(config_manager.runtime_config.client_num):
        w_delta.append(stc(client_residual[i], config_manager.model_config.stc_sparsity))
        client_residual[i] -= w_delta[-1]
        print(i)
    print(f'Client STC cost {time.time() - st}')

    pdb.set_trace()

    st = time.time()
    receive_delta_w = np.average(w_delta, axis=0, weights=client_data_size / client_data_size.sum())
    server_grad_plus_residual = receive_delta_w + server_residual
    server_delta_w = stc(server_grad_plus_residual, config_manager.model_config.stc_sparsity)
    server_residual = server_grad_plus_residual - server_delta_w
    print(f'Server STC cost {time.time() - st}')

    del w_delta, server_grad_plus_residual

    cur_weights = ml_model.get_weights()
    pointer = 0
    for i in range(len(cur_weights)):
        cur_weights[i] += np.reshape(
            server_delta_w[pointer:pointer+np.prod(cur_weights[i].shape)], cur_weights[i].shape)

    del server_delta_w

    ml_model.set_weights(cur_weights)

    # Evaluate
    val_log = ml_model.evaluate(x_val, y_val, verbose=0, batch_size=batch_size)
    test_log = ml_model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print(
        f'Epoch {epoch} Val Loss {val_log[0]}, Val Acc {val_log[1]}, '
        f'Test Loss {test_log[0]}, Test Acc {test_log[1]}'
    )
    if val_log[0] < early_stopping_metric:
        early_stopping_metric = val_log[0]
        best_test_metric = test_log
        patience = 0
    else:
        patience += 1
    if patience > model_config.tolerance_num:
        print('Train Finished')
        print(f'Best Test Metric {test_log}')
        break
    del actual_size
    test_metric_each_round.append([epoch] + test_log)

# output_dir = config_manager.log_dir_path
# os.makedirs(output_dir, exist_ok=True)
# with open(os.path.join(
#         output_dir,
#         f'{UNIFIED_JOB_TIME}_{data_config.dataset_name}_{runtime_config.client_num}_fed_sgd_simulator.csv'
# ), 'w') as f:
#     f.write(', '.join(
#         [str(e) for e in [data_config.dataset_name, runtime_config.client_num, model_config.learning_rate]]
#     ) + '\n')
#     for e in test_metric_each_round:
#         f.write(', '.join([str(e1) for e1 in e]) + '\n')
#     f.write(f'Best Metric, {best_test_metric[0]}, {best_test_metric[1]}')
# write_history()
