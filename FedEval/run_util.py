import argparse
import copy
import datetime
import json
import logging
import os
import platform
import shutil
import hickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams.update({'font.size': 14})

import requests
import yaml

from .config.configuration import *
from FedEval.run import generate_data, run
from FedEval.utils import ParamParser, LogAnalysis, History
from multiprocessing import Pool

sudo = ""


def check_status(host):
    try:
        status = requests.get('http://{}/status'.format(host), timeout=(300, 300))
        return {'success': True, 'data': json.loads(status.text)}
    except Exception as e:
        print('Error in checking', e)
        return {'success': False, 'data': None}


def local_stop():
    os.system(sudo + 'docker-compose stop')


def server_stop():

    import paramiko

    machines = ConfigurationManager().runtime_config.machines
    for name, machine in machines.items():
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        host = machine.addr
        port = machine.port
        user_name = machine.username
        remote_path = machine.work_dir_path
        key_file = machine.key_filename

        ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)
        _, stdout, stderr = ssh.exec_command(
            f'cd {remote_path};' + sudo +
            f'docker-compose --compatibility -f docker-compose-{name}.yml stop'
        )

        print(''.join(stdout.readlines()))
        print(''.join(stderr.readlines()))


def local_recursive_ls(path):
    files = os.listdir(path)
    results = []
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            results += local_recursive_ls(os.path.join(path, file))
        else:
            results.append(os.path.join(path, file))
    return results


def remote_recursive_ls(sftp, remote_dir_base, path):
    files = sftp.listdir(remote_dir_base + '/' + path)
    results = []
    for file in files:
        try:
            results += remote_recursive_ls(sftp, remote_dir_base, path + '/' + file)
        except FileNotFoundError:
            results.append(path + '/' + file)
    return results


def remote_recursive_mkdir(sftp, remote_path):
    p = remote_path.split('/')
    path = ''
    for i in p[1:-1]:
        path = path + '/'
        dirs = sftp.listdir(path)
        if i in dirs:
            path = path + i
        else:
            path = path + i
            sftp.mkdir(path)


def local_recursive_mkdir(local_path):
    p = local_path.split('/')
    path = '.'
    for i in p[:-1]:
        path += '/'
        dirs = os.listdir(path)
        if i in dirs:
            path = path + i
        else:
            path = path + i
            os.makedirs(path, exist_ok=True)
            print('new dir', path)


def upload_to_server(local_dirs, file_type=('.py', '.yml', '.css', '.html', 'eot', 'svg', 'ttf', 'woff')):
    
    import paramiko
    
    files = []
    for path in local_dirs:
        files += local_recursive_ls(path)

    def upload_check(file_name):
        for ft in file_type:
            if file_name.endswith(ft):
                return True
        return False

    files = [e for e in files if upload_check(e)]
    if platform.system().lower() == 'windows':
        files = [e.replace('\\', '/') for e in files]

    host_record = []
    machines = ConfigurationManager().runtime_config.machines
    for machine in machines.values():
        host = machine.addr
        port = machine.port
        user_name = machine.username
        remote_path = machine.work_dir_path
        key_file = machine.key_filename

        if f"{host}:{port}" in host_record:
            continue
        private_key = paramiko.RSAKey.from_private_key_file(key_file)
        trans = paramiko.Transport((host, port))
        trans.connect(username=user_name, pkey=private_key)
        sftp = paramiko.SFTPClient.from_transport(trans)
        for file in files:
            try:
                sftp.put(localpath=file, remotepath=remote_path + '/' + file)
            except FileNotFoundError:
                remote_recursive_mkdir(sftp, remote_path + '/' + file)
                sftp.put(localpath=file, remotepath=remote_path + '/' + file)
            print('Uploaded', file, 'to', user_name + '@' + host + ':' + str(port))
        trans.close()
        host_record.append(f"{host}:{port}")


def download_from_server(remote_dirs, file_type):

    import paramiko
    server = [m for m in ConfigurationManager(
    ).runtime_config.machines.values() if m.is_server][0]

    def download_check(file_name):
        for ft in file_type:
            if file_name.endswith(ft):
                return True
        return False

    host = server.addr
    port = server.port
    user_name = server.username
    remote_path = server.work_dir_path
    key_file = server.key_filename

    private_key = paramiko.RSAKey.from_private_key_file(key_file)
    trans = paramiko.Transport((host, port))
    trans.connect(username=user_name, pkey=private_key)
    sftp = paramiko.SFTPClient.from_transport(trans)

    files = []
    for path in remote_dirs:
        files += remote_recursive_ls(sftp, remote_path, path)
    files = [e for e in files if download_check(e)]

    for file in files:
        try:
            sftp.get(localpath=file, remotepath=remote_path + '/' + file)
        except FileNotFoundError:
            local_recursive_mkdir(file)
            sftp.get(localpath=file, remotepath=remote_path + '/' + file)
        print('Downloaded', file, 'from', user_name + '@' + host + ':' + str(port))
    trans.close()


def recursive_update_dict(target: dict, update: dict):
    for key in update:
        if key in target and isinstance(update[key], dict):
            target[key] = recursive_update_dict(target[key], update[key])
        else:
            target[key] = update[key]
    return target


def _handle_errors(error):
    logging.error(f'Subprocess error {error}')


def run_util(execution, mode, config, overwrite_config=False, skip_if_exit=True, **kwargs):

    if len(kwargs) > 0:
        print('*' * 40)
        print('Received parameter update')
        print(kwargs)
        print('*' * 40)

    data_config, model_config, runtime_config = ConfigurationManager.load_configs(config, serializer='yaml')
    # --- configurations modification area start ---
    if 'data_config' in kwargs:
        data_config = recursive_update_dict(data_config, kwargs['data_config'])
    if 'model_config' in kwargs:
        model_config = recursive_update_dict(
            model_config, kwargs['model_config'])
    if 'runtime_config' in kwargs:
        runtime_config = recursive_update_dict(
            runtime_config, kwargs['runtime_config'])

    if mode == 'local':
        # TODO(fgh) import dict keies from config module
        runtime_config['server']['host'] = 'server'
        runtime_config['server']['listen'] = 'server'
    if mode == 'remote':
        runtime_config['server']['host'] = runtime_config['machines']['server']['host']
        runtime_config['server']['listen'] = '0.0.0.0'
    # --- configurations modification area end ---

    UNIFIED_JOB_TIME = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
    os.environ['UNIFIED_JOB_TIME'] = UNIFIED_JOB_TIME

    cfg_mgr = ConfigurationManager(data_config, model_config, runtime_config)
    if overwrite_config:
        new_config_dir_path = config
    else:
        new_config_dir_path = config + '_' + cfg_mgr.config_unique_id[:10]
    cfg_mgr.to_files(new_config_dir_path, serializer='yaml')
    rt_cfg = cfg_mgr.runtime_config

    config_unique_id = cfg_mgr.config_unique_id

    if execution == 'upload':
        print('Uploading to the server')
        if rt_cfg.machines is None:
            raise ValueError('No machine config found, please check',
                             os.path.join(config, '3_runtime_config.yml'))
        upload_to_server(local_dirs=['FedEval', 'configs'], file_type=(
            '.py', '.yml', '.css', '.html', 'eot', 'svg', 'ttf', 'woff'))

    if execution == 'stop':
        if mode == 'local':
            local_stop()
        if mode == 'remote':
            server_stop()

    if skip_if_exit and os.path.isfile(os.path.join(cfg_mgr.history_record_path, 'history.json')):
        history = History(cfg_mgr.history_record_path)
        if history.check_exist(config_unique_id):
            print('#' * 40)
            print(f"Found existing log in history {history.query(config_unique_id)}")
            print('Skipping this run...')
            print('#' * 40)
            execution = None

    if execution == 'simulate_fedsgd':
        fed_sgd_simulator(UNIFIED_JOB_TIME)
        write_history()

    if execution == 'simulate_central':
        central_simulator(UNIFIED_JOB_TIME)
        write_history()

    if execution == 'simulate_local':
        local_simulator(UNIFIED_JOB_TIME)
        write_history()

    if execution == 'run':

        if mode == 'without-docker':
            generate_data(save_file=True)
            process_pool = Pool(rt_cfg.container_num + 1)
            process_pool.apply_async(run, args=(
                'server', new_config_dir_path, UNIFIED_JOB_TIME), error_callback=_handle_errors)
            for i in range(rt_cfg.container_num):
                process_pool.apply_async(run, args=(
                    'client', new_config_dir_path, UNIFIED_JOB_TIME, str(i)), error_callback=_handle_errors)
            process_pool.close()

        if mode == 'local':
            current_path = os.path.abspath('./')
            os.system(
                sudo + f'docker run -it --rm '
                       f'-e UNIFIED_JOB_TIME={UNIFIED_JOB_TIME} '
                       f'-v {current_path}:{current_path} '
                       f'-w {current_path} {rt_cfg.image_label} '
                       f'python3 -W ignore -m FedEval.run -f data -c {new_config_dir_path}'
            )
            os.system(
                sudo + f'docker run -it --rm '
                       f'-e UNIFIED_JOB_TIME={UNIFIED_JOB_TIME} '
                       f'-v {current_path}:{current_path} '
                       f'-w {current_path} {rt_cfg.image_label} '
                       f'python3 -W ignore -m FedEval.run -f compose-local -c {new_config_dir_path}'
            )
            os.system(sudo + 'docker-compose --compatibility up -d')

        if mode == 'remote':

            import paramiko

            upload_to_server(local_dirs=('FedEval', 'configs'))

            for m_name, machine in rt_cfg.machines.items():

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                host = machine.addr
                port = machine.port
                user_name = machine.username
                remote_path = machine.work_dir_path

                key_file = machine.key_filename
                ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)

                _, stdout, stderr = ssh.exec_command(
                    sudo + f'docker run -it --rm '
                           f'-e UNIFIED_JOB_TIME={UNIFIED_JOB_TIME} '
                           f'-v {remote_path}:{remote_path} '
                           f'-w {remote_path} {rt_cfg.image_label} '
                           f'python3 -W ignore -m FedEval.run -f data -c {new_config_dir_path}'
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

                _, stdout, stderr = ssh.exec_command(
                    sudo + f'docker run -it --rm '
                           f'-e UNIFIED_JOB_TIME={UNIFIED_JOB_TIME} '
                           f'-v {remote_path}:{remote_path} '
                           f'-w {remote_path} {rt_cfg.image_label} '
                           f'python3 -W ignore -m FedEval.run -f compose-server -c {new_config_dir_path}'
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

                if machine.is_server:
                    print('Start Server')
                    _, stdout, stderr = ssh.exec_command(
                        f'cd {remote_path};' +
                        sudo + 'docker-compose --compatibility -f docker-compose-server.yml up -d')
                else:
                    print('Start Clients', m_name)
                    _, stdout, stderr = ssh.exec_command(
                        f'cd {remote_path};' +
                        sudo + f'docker-compose --compatibility -f docker-compose-{m_name}.yml up -d')

                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

        print('Start succeed!')

        time.sleep(20)

        host = '127.0.0.1' if mode == 'local' else rt_cfg.central_server_addr
        port = rt_cfg.central_server_port

        status_url = f'http://{host}:{port}/status'
        print(f'Starting to monitor at {status_url}, check every 10 seconds')
        dashboard_url = f'http://{host}:{port}/dashboard'
        print(f'Check the dashboard at {dashboard_url}')

        check_status_result = check_status(host + ':' + str(port))
        current_round = None

        while True:
            if check_status_result['success']:
                if not check_status_result['data'].get('finished', False):
                    received_round = check_status_result['data'].get('rounds')
                    if received_round is not None and (current_round is None or current_round < received_round):
                        print('Running at Round %s' % received_round, 'Results', check_status_result['data'].get('results', 'unknown'))
                        current_round = received_round
                    time.sleep(10)
                else:
                    break
            else:
                print('Check failed, try later')
                time.sleep(10)
            check_status_result = check_status(host + ':' + str(port))

        status_data = check_status_result['data']
        if status_data is not None:

            log_dir = status_data['log_dir']
            log_file = log_dir + '/train.log'
            result_file = log_dir + '/results.json'

            if mode == 'remote':
                if log_dir.startswith('/FML/'):
                    log_dir = log_dir[5:]
                os.makedirs(log_dir, exist_ok=True)
                download_from_server(remote_dirs=[log_dir], file_type=[
                                     '.yml', '.json', '.log'])

        if mode == 'local':
            local_stop()

        if mode == 'remote':
            server_stop()

        if mode == 'without-docker':
            process_pool.terminate()

        write_history()

    if not overwrite_config:
        shutil.rmtree(new_config_dir_path)


def write_history():
    cfg_mgr = ConfigurationManager()
    if os.path.isfile(os.path.join(cfg_mgr.history_record_path, 'history.json')):
        with open(os.path.join(cfg_mgr.history_record_path, 'history.json'), 'r') as f:
            history = json.load(f)
    else:
        history = {}
    history[cfg_mgr.config_unique_id] = {'finished': True, 'log_path': cfg_mgr.log_dir_path}
    with open(os.path.join(cfg_mgr.history_record_path, 'history.json'), 'w') as f:
        json.dump(history, f)


def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
        loss = loss_op(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
    for i in range(len(gradients)):
        try:
            gradients[i] = gradients[i].numpy()
        except AttributeError:
            gradients[i] = tf.convert_to_tensor(gradients[i]).numpy()
    return gradients


def fed_sgd_simulator(UNIFIED_JOB_TIME):
    config_manager = ConfigurationManager()
    data_config = config_manager.data_config
    model_config = config_manager.model_config
    runtime_config = config_manager.runtime_config
    print('#' * 80)
    print('->', 'Starting simulation on FedSGD for parameter tuning')
    print('#' * 80)
    generate_data(True)
    print('#' * 80)
    print('->', 'Data Generated')
    print('#' * 80)
    client_data_name = [
        os.path.join(config_manager.data_dir_name, e) for e in os.listdir(config_manager.data_dir_name) if e.startswith('client')
    ]
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
    test_metric_each_round = []
    patience = 0
    batch_size = model_config.B
    for epoch in range(model_config.max_round_num):
        batched_gradients = []
        actual_size = []
        for i in range(0, len(x_train), batch_size):
            actual_size.append(min(batch_size, len(x_train) - i))
            batched_gradients.append(
                [e / float(actual_size[-1]) for e in
                 compute_gradients(ml_model, x_train[i:i + batch_size], y_train[i:i + batch_size])]
            )
        actual_size = np.array(actual_size) / np.sum(actual_size)
        aggregated_gradients = []

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
    write_history()


def central_simulator(UNIFIED_JOB_TIME):
    config_manager = ConfigurationManager()
    data_config = config_manager.data_config
    model_config = config_manager.model_config
    runtime_config = config_manager.runtime_config
    print('#' * 80)
    print('->', 'Starting simulation on central training')
    print('#' * 80)
    generate_data(True)
    print('#' * 80)
    print('->', 'Data Generated')
    print('#' * 80)
    client_data_name = [
        os.path.join(config_manager.data_dir_name, e) for e in os.listdir(config_manager.data_dir_name) if e.startswith('client')
    ]
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

    start_train_time = time.time()
    parameter_parser = ParamParser()
    ml_model = parameter_parser.parse_model()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=model_config.tolerance_num, restore_best_weights=True
    )
    output_dir = config_manager.log_dir_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_name = os.path.join(
        output_dir, f'{UNIFIED_JOB_TIME}_{data_config.dataset_name}_{runtime_config.client_num}_central_simulator.csv'
    )
    csv_logger = tf.keras.callbacks.CSVLogger(filename=log_file_name, append=True)
    ml_model.fit(
        x_train, y_train, batch_size=model_config.B, epochs=model_config.max_round_num,
        validation_data=(x_val, y_val), callbacks=[early_stop, csv_logger]
    )
    val_loss, val_acc = ml_model.evaluate(x_val, y_val, verbose=0, batch_size=model_config.B)
    test_loss, test_acc = ml_model.evaluate(x_test, y_test, verbose=0, batch_size=model_config.B)

    with open(log_file_name, 'a+') as f:
        f.write(', '.join(
            [str(e) for e in [data_config.dataset_name, runtime_config.client_num, model_config.learning_rate]]
        ) + '\n')
        f.write(f'Central Train Finished, Duration {time.time() - start_train_time}\n')
        f.write(f'Best VAL Metric, {val_loss}, {val_acc}\n')
        f.write(f'Best TEST Metric, {test_loss}, {test_acc}\n')
    write_history()


def local_simulator(UNIFIED_JOB_TIME):
    config_manager = ConfigurationManager()
    data_config = config_manager.data_config
    model_config = config_manager.model_config
    runtime_config = config_manager.runtime_config
    print('#' * 80)
    print('->', 'Starting simulation on local training')
    print('#' * 80)
    generate_data(True)
    print('#' * 80)
    print('->', 'Data Generated')
    print('#' * 80)
    client_data_name = [
        os.path.join(config_manager.data_dir_name, e) for e in os.listdir(config_manager.data_dir_name)
        if e.startswith('client')
    ]
    client_data_name = sorted(client_data_name, key=lambda x: int(x.split('_')[-1].strip('.pkl')))
    client_data = []
    for data_name in client_data_name:
        with open(data_name, 'r') as f:
            client_data.append(hickle.load(f))

    parameter_parser = ParamParser()
    ml_model = parameter_parser.parse_model()

    initial_weights = ml_model.get_weights()
    output_dir = config_manager.log_dir_path
    os.makedirs(output_dir, exist_ok=True)
    log_file_name = os.path.join(
        output_dir, f'{UNIFIED_JOB_TIME}_{data_config.dataset_name}_{runtime_config.client_num}_local_simulator.csv'
    )

    average_test_acc = []
    for i in range(len(client_data)):
        xy = client_data[i]
        start_time = time.time()
        with open(log_file_name, 'a+') as f:
            f.write(f'! Client {i} !\n')
        ml_model.set_weights(initial_weights)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=model_config.tolerance_num, restore_best_weights=True
        )
        ml_model.fit(
            xy['x_train'], xy['y_train'], batch_size=model_config.B, epochs=model_config.max_round_num,
            validation_data=(xy['x_val'], xy['y_val']), callbacks=[early_stop]
        )
        val_loss, val_acc = ml_model.evaluate(xy['x_val'], xy['y_val'], verbose=0, batch_size=model_config.B)
        test_loss, test_acc = ml_model.evaluate(xy['x_test'], xy['y_test'], verbose=0, batch_size=model_config.B)
        with open(log_file_name, 'a+') as f:
            f.write(f'Client {i} Finished Duration {time.time() - start_time}\n')
            f.write(f'Client {i} Best VAL Metric, {val_loss}, {val_acc}\n')
            f.write(f'Client {i} Best TEST Metric, {test_loss}, {test_acc}\n')
        average_test_acc.append(test_acc)
    with open(log_file_name, 'a+') as f:
        f.write(', '.join(
            [str(e) for e in [data_config.dataset_name, runtime_config.client_num, model_config.learning_rate]]
        ) + '\n')
        f.write(f'Average Best Test Metric, {np.mean(average_test_acc)}')
    write_history()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--execute', '-e', choices=(
        'run', 'stop', 'upload', 'log',
        'simulate_fedsgd', 'simulate_central', 'simulate_local'
    ),
                             help='Start or Stop the experiments')
    args_parser.add_argument('--mode', '-m', choices=('remote', 'local', 'without-docker'),
                             help='Run the experiments locally or remotely that presented the runtime_config')
    args_parser.add_argument('--config', '-c', default='./',
                             help='The path to the config files, defaults to be ./')
    args_parser.add_argument('--path', '-p', help='path')
    args = args_parser.parse_args()

    if args.execute == 'log':
        if args.path is None:
            raise ValueError('Please provide log_dir')
        LogAnalysis(args.path).to_csv('_'.join(args.path.split('/')) + '.csv')
    else:
        run_util(execution=args.execute, mode=args.mode, config=args.config, overwrite_config=False, skip_if_exit=True)
