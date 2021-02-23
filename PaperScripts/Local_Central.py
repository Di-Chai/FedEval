import os
import yaml
import pickle
import datetime
import numpy as np
import tensorflow as tf

from FedEval.model import *

gpu = '0'


def client_local_train(client_id, data_config, model_config, runtime_config):
    # 1 Load data
    with open(os.path.join(data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
        local_data = pickle.load(f)

    # Build val and test data
    x_val, y_val, x_test, y_test = [], [], [], []
    for c_id in range(runtime_config['clients']['num_clients']):
        with open(os.path.join(data_config['data_dir'], 'client_%s.pkl' % c_id), 'rb') as f:
            tmp = pickle.load(f)
            x_val.append(tmp['x_val'])
            y_val.append(tmp['y_val'])
            x_test.append(tmp['x_test'])
            y_test.append(tmp['y_test'])
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # 3 Config Model
    input_params = {
        "inputs_shape": {'x': local_data['x_train'].shape[1:]},
        "targets_shape": {'y': local_data['y_train'].shape[1:]},
    }
    model_config[model_config['Model']].update(input_params)
    model_config[model_config['Model']]['lr'] = float(runtime_config['clients']['lr'])
    model_config[model_config['Model']]['gpu_device'] = gpu
    model_obj = eval(model_config['Model'] + "(**model_config[model_config['Model']])")

    model_obj.build()

    if model_config['Model'] in ['MobileNet', 'VGG16', 'ResNet50']:
        call_backs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, mode='max')
        ]
        model_obj.model.fit(
            x=local_data['x_train'], y=local_data['y_train'],
            epochs=max_epoch, batch_size=runtime_config['clients']['local_batch_size'],
            callbacks=call_backs, verbose=2,
            validation_data=(x_val, y_val),
        )
        test_accuracy = model_obj.model.evaluate(
            x=local_data['x_test'], y=local_data['y_test']
        )[1]
    else:
        model_obj.fit(
            train_data={'x': local_data['x_train'], 'y': local_data['y_train']},
            val_data={'x': x_val, 'y': y_val},
            output_names=('loss', 'accuracy'),
            op_names=('train_op',),
            evaluate_loss_name='loss',
            batch_size=runtime_config['clients']['local_batch_size'],
            max_epoch=runtime_config['server']['MAX_NUM_ROUNDS'],
            early_stop_method='native',
            early_stop_patience=runtime_config['server']['NUM_TOLERATE'],
            save_model=False,
        )
        test_accuracy = model_obj.predict(test_data={'x': x_test, 'y': y_test}, output_names=('accuracy',))['accuracy']

    model_obj.close()

    return np.mean(test_accuracy)


def central_train(data_config, model_config, runtime_config):
    data = []
    for c_id in range(runtime_config['clients']['num_clients']):
        with open(os.path.join(data_config['data_dir'], 'client_%s.pkl' % c_id), 'rb') as f:
            data.append(pickle.load(f))

    # 3 Config Model
    input_params = {
        "inputs_shape": {'x': data[0]['x_train'].shape[1:]},
        "targets_shape": {'y': data[0]['y_train'].shape[1:]},
    }
    model_config[model_config['Model']].update(input_params)
    model_config[model_config['Model']]['lr'] = float(runtime_config['clients']['lr'])
    model_config[model_config['Model']]['gpu_device'] = gpu
    model_obj = eval(model_config['Model'] + "(**model_config[model_config['Model']])")

    model_obj.build()

    x_train = np.concatenate([e['x_train'] for e in data])
    y_train = np.concatenate([e['y_train'] for e in data])
    x_val = np.concatenate([e['x_val'] for e in data])
    y_val = np.concatenate([e['y_val'] for e in data])
    x_test = np.concatenate([e['x_test'] for e in data])
    y_test = np.concatenate([e['y_test'] for e in data])

    if model_config['Model'] in ['MobileNet', 'VGG16']:
        call_backs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, mode='max')
        ]
        model_obj.model.fit(
            x=x_train, y=y_train, validation_data=(x_val, y_val),
            epochs=max_epoch, batch_size=runtime_config['clients']['local_batch_size'],
            callbacks=call_backs, verbose=2,
        )
        test_accuracy = model_obj.model.evaluate(x=x_test, y=y_test)[1]
    else:
        model_obj.fit(
            train_data={'x': x_train, 'y': y_train},
            val_data={'x': x_val, 'y': y_val},
            output_names=('loss', 'accuracy'),
            op_names=('train_op',),
            evaluate_loss_name='loss',
            batch_size=runtime_config['clients']['local_batch_size'],
            max_epoch=runtime_config['server']['MAX_NUM_ROUNDS'],
            early_stop_method='native',
            early_stop_patience=runtime_config['server']['NUM_TOLERATE'],
            save_model=False,
        )
        test_accuracy = model_obj.predict(test_data={'x': x_test, 'y': y_test}, output_names=('accuracy',))['accuracy']

    model_obj.close()

    return np.mean(test_accuracy)


if __name__ == '__main__':

    config_path = './'

    output_file_name = 'accuracy_metric_trials.txt'

    with open(os.path.join(config_path, '1_data_config.yml'), 'r') as f:
        data_config = yaml.load(f)

    with open(os.path.join(config_path, '2_model_config.yml'), 'r') as f:
        model_config = yaml.load(f)

    with open(os.path.join(config_path, '3_runtime_config.yml'), 'r') as f:
        runtime_config = yaml.load(f)

    params_run = [
        ['mnist', 'MLP', 'adam', 0, 'iid', 5e-4, 32],
        ['mnist', 'LeNet', 'adam', 0, 'iid', 5e-4, 32],

        ['femnist', 'LeNet', 'adam', 0, 'iid', 5e-4, 32],

        ['celeba', 'LeNet', 'adam', 0, 'iid', 5e-4, 4],
    ]

    num_clients = 100

    max_epoch = 5000

    repeat_time = 50

    for dataset, model, optimizer, non_iid, strategy, lr, batch_size in params_run:

        data_config['dataset'] = dataset
        data_config['non-iid'] = non_iid
        data_config['non-iid-strategy'] = strategy

        model_config['Model'] = model
        model_config[model]['optimizer'] = optimizer

        runtime_config['server']['MAX_NUM_ROUNDS'] = max_epoch
        runtime_config['clients']['local_batch_size'] = batch_size
        runtime_config['clients']['lr'] = lr
        runtime_config['clients']['num_clients'] = num_clients

        with open(os.path.join(config_path, '1_data_config.yml'), 'w') as f:
            yaml.dump(data_config, f)

        with open(os.path.join(config_path, '2_model_config.yml'), 'w') as f:
            yaml.dump(model_config, f)

        with open(os.path.join(config_path, '3_runtime_config.yml'), 'w') as f:
            yaml.dump(runtime_config, f)

        os.system('python 4_distribute_data.py --path %s' % config_path)

        accuracy_central = []
        for r in range(repeat_time):
            accuracy_central.append(central_train(data_config, model_config, runtime_config))
            print(accuracy_central[-1])

        print('Central Accuracy', accuracy_central)

        accuracy_clients = []
        for client_id in range(runtime_config['clients']['num_clients']):
            test_acc = client_local_train(client_id, data_config, model_config, runtime_config)
            accuracy_clients.append(test_acc)
            print(accuracy_clients[-1])

        if os.path.isfile(output_file_name) is False:
            with open('accuracy_metric_trials.txt', 'w') as f:
                f.write('time, dataset, model, optimizer, IID, IID-Strategy, batch-size, '
                        'LR, max-epoch, early-stop-patience, LocalAcc, CentralAcc')

        with open('accuracy_metric_trials.txt', 'a+') as f:
            data = [
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                dataset, model, optimizer,
                non_iid, strategy, batch_size, runtime_config['clients']['lr'],
                max_epoch, runtime_config['server']['NUM_TOLERATE'],
                np.mean(accuracy_clients), np.mean(accuracy_central),
            ]
            data = [str(e) for e in data]
            f.writelines(', '.join(data) + '\n')
