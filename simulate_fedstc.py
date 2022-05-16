import pdb
import time

import numpy as np

from FedEval.run_util import *

import tensorflow as tf

# config_manager = ConfigurationManager()
config_manager: ConfigurationManager = ConfigurationManager.from_files('configs/debug')
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
print('->', 'Starting simulation on FedSGD for parameter tuning')
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
        client_data_size.append([
            len(client_data[-1]['x_train']), len(client_data[-1]['x_val']), len(client_data[-1]['x_test'])
        ])

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
batch_size = 1024

params_shape = [e.shape for e in ml_model.get_weights()]
params_shape_flatten = np.sum([np.prod(e) for e in params_shape])

server_residual = np.zeros([params_shape_flatten])
client_residual = np.zeros([config_manager.runtime_config.client_num, params_shape_flatten])


def vectorized_gradient(xy):
    with tf.GradientTape() as tape:
        x = tf.expand_dims(xy[0], 0)
        y = tf.expand_dims(xy[1], 0)
        y_hat = ml_model(x)
        loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
        loss = loss_op(y, y_hat)
        gradients = tape.gradient(loss, ml_model.trainable_variables)
        return gradients


for epoch in range(model_config.max_round_num):

    st = time.time()
    pointer = 0
    batched_gradients = []
    actual_size = []
    for i in range(config_manager.runtime_config.client_num):
        actual_size.append(client_data_size[i][0])
        batched_gradients.append(
            [e / float(actual_size[-1]) for e in
             compute_gradients(
                 ml_model,
                 x_train[pointer:pointer + actual_size[-1]],
                 y_train[pointer:pointer + actual_size[-1]])]
        )
        pointer += actual_size[-1]
    print(f'Per-client gradients cost {time.time() - st}')

    batched_gradients = []
    actual_size = []
    st = time.time()
    for i in range(0, len(x_train), batch_size):
        actual_size.append(min(batch_size, len(x_train) - i))
        batched_gradients.append(
            [e / float(actual_size[-1]) for e in
             compute_gradients(ml_model, x_train[i:i + batch_size], y_train[i:i + batch_size])]
        )
    print(f'Agg-sample gradients cost {time.time() - st}')

    actual_size = np.array(actual_size) / np.sum(actual_size)
    aggregated_gradients = []

    pdb.set_trace()

    for i in range(len(batched_gradients[0])):
        aggregated_gradients.append(np.average([e[i] for e in batched_gradients], axis=0, weights=actual_size))
    ml_model.optimizer.apply_gradients(zip(aggregated_gradients, ml_model.trainable_variables))
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
    del batched_gradients
    del actual_size
    test_metric_each_round.append([epoch] + test_log)
output_dir = config_manager.log_dir_path
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(
        output_dir,
        f'{UNIFIED_JOB_TIME}_{data_config.dataset_name}_{runtime_config.client_num}_fed_sgd_simulator.csv'
), 'w') as f:
    f.write(', '.join(
        [str(e) for e in [data_config.dataset_name, runtime_config.client_num, model_config.learning_rate]]
    ) + '\n')
    for e in test_metric_each_round:
        f.write(', '.join([str(e1) for e1 in e]) + '\n')
    f.write(f'Best Metric, {best_test_metric[0]}, {best_test_metric[1]}')
_write_history()