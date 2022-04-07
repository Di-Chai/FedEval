import argparse
import copy
import datetime
import json
import os
import pdb
import platform
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams.update({'font.size': 14})

import requests
import yaml

from .config.configuration import *

sudo = ""


class LogAnalysis:

    def __init__(self, log_dir=os.path.join('log', 'server')):
        self.log_dir = log_dir
        self.logs = [e for e in os.listdir(log_dir) if not e.startswith('.')]

        self.configs = []
        self.results = []
        for log in self.logs:
            try:
                c1, c2, c3 = _load_config(os.path.join(log_dir, log))
                with open(os.path.join(log_dir, log, 'results.json'), 'r') as f:
                    results = json.load(f)
                self.results.append(results)
                self.configs.append({'data_config': c1, 'model_config': c2, 'runtime_config': c3})
                print('Get log', log)
            except FileNotFoundError:
                print('Config not found in', log, 'skip to next')
                continue
        
        self.omit_keys = [
            'runtime_config$$machines',
            'runtime_config$$server',
            'data_config$$random_seed'
        ]

        def check_omit(key):
            for e in self.omit_keys:
                if e in key:
                    return True
            return False

        self.key_templates = [
            [key_chain for key_chain in self.parse_dict_keys(e) if not check_omit(key_chain)]
            for e in self.configs
        ]

        self.key_templates = max(self.key_templates, key=lambda x: len(x))

        self.diff_keys = []
        for key_chain in self.key_templates:
            tmp_len = len(self.diff_keys)
            for i in range(len(self.configs)):
                for j in range(len(self.configs)):
                    if i == j:
                        continue
                    else:
                        if self.recursive_retrieve(self.configs[i], key_chain) != \
                                self.recursive_retrieve(self.configs[j], key_chain):
                            self.diff_keys.append(key_chain)
                            break
                if len(self.diff_keys) > tmp_len:
                    break

        self.csv_result_keys = [
            ['central_train$$test_accuracy', lambda x: [x] if x is None else [float(x)]],
            ['best_metric$$test_accuracy', lambda x: [float(x)]],
            ['total_time', lambda x: [int(x.split(':')[0])*60+int(x.split(':')[1])+int(x.split(':')[2])/60]],
            ['total_rounds', lambda x: [int(x)]],
            ['server_send', lambda x: [float(x)]],
            ['server_receive', lambda x: [float(x)]],
            ['time_detail', lambda x: eval(x)],
        ]

        self.configs_diff = self.retrieve_diff_configs()
        self.csv_results = self.parse_results()

        self.average_results = self.aggregate_csv_results()

    def plot(self, join_keys=('data_config$$dataset', ), label_keys=('model_config$$FedModel$$name', )):

        num_colors = 10
        line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        color_map = plt.get_cmap('gist_rainbow')

        aggregate_results = {}
        for i in range(len(self.configs)):
            record = self.results[i]
            join_keys_strings = '-'.join([str(self.recursive_retrieve(self.configs[i], e)) for e in join_keys])
            label_keys_strings = '-'.join([str(self.recursive_retrieve(self.configs[i], e)) for e in label_keys])
            # Find the best index
            val_loss_list = [
                record['info_each_round'][str(e + 1)]['val_loss'] for e in range(len(record['info_each_round']))
            ]
            # best_index = val_loss_list.index(min(val_loss_list))
            best_index = len(val_loss_list)
            if best_index <= 1:
                continue
            if self.configs[i]['data_config']['dataset'] == 'semantic140':
                test_acc_key = 'test_binary_accuracy'
            else:
                test_acc_key = 'test_accuracy'
            test_acc_list = [
                record['info_each_round'][str(e+1)][test_acc_key] for e in range(len(record['info_each_round']))
            ]
            test_acc_list = test_acc_list[:best_index]
            # CommRound to Accuracy
            cr_to_acc = [e + 1 for e in range(len(record['info_each_round']))][:best_index]
            # CommAmount to Accuracy
            ca_avg_round = (record['server_send'] + record['server_receive']) / len(record['info_each_round'])
            ca_avg_round_client = ca_avg_round / self.configs[i]['runtime_config']['server']['num_clients'] * 2**10  # MB
            ca_to_acc = [(e+1) * ca_avg_round_client for e in range(len(record['info_each_round']))][:best_index]
            # Time to Accuracy
            time_to_acc = [0] + [record['info_each_round'][str(e+1)]['timestamp'] -
                                 record['info_each_round']['1']['timestamp']
                                 for e in range(1, len(record['info_each_round']))][:best_index]
            
            if join_keys_strings not in aggregate_results:
                aggregate_results[join_keys_strings] = {}
            if label_keys_strings not in aggregate_results[join_keys_strings]:
                aggregate_results[join_keys_strings][label_keys_strings] = []

            assert len(cr_to_acc) == len(ca_to_acc) == len(time_to_acc) == len(test_acc_list)

            aggregate_results[join_keys_strings][label_keys_strings].append(
                [cr_to_acc, ca_to_acc, time_to_acc, test_acc_list])
        
        def multi_to_single(data, tag):
            max_length = max([len(e) for e in data])
            max_length_index = [len(e) for e in data].index(max_length)
            for i in range(len(data)):
                if len(data[i]) != max_length:
                    if tag == 'metric':
                        data[i] = data[i] + data[max_length_index][-(max_length-len(data[i])):]
                    elif tag == 'acc':
                        data[i] = data[i] + [data[i][-1]] * (max_length - len(data[i]))
                    else:
                        raise ValueError
            single_data = np.mean(data, axis=0)
            return single_data
        
        def plot_one_image(key, result_key):
            fig, ax = plt.subplots(1, 3, figsize=[30, 10])
            counter = 0
            for k2 in sorted(result_key.keys()):
                line0 = ax[0].plot(result_key[k2][0], result_key[k2][-1], label=k2)
                line1 = ax[1].plot(result_key[k2][1], result_key[k2][-1], label=k2)
                line2 = ax[2].plot(result_key[k2][2], result_key[k2][-1], label=k2)
                for line in [line0, line1, line2]:
                    # line[0].set_color(color_map(counter//len(line_styles)*float(len(line_styles))/num_colors))
                    # line[0].set_linestyle(line_styles[counter % len(line_styles)])
                    line[0].set_color(color_map(float(counter % num_colors) / num_colors))
                    line[0].set_linestyle(line_styles[counter//num_colors])
                counter += 1

            x_labels = ['CR', 'CA', 'Time']
            for i in range(3):
                ax[i].legend()
                ax[i].grid()
                ax[i].set_ylabel('Accuracy')
                ax[i].set_xlabel(x_labels[i])
            ax[1].set_title(key)
            fig.tight_layout()
            plt.savefig(os.path.join('log/images', '%s.png' % key), dpi=400)
            plt.close()

        for k1 in aggregate_results:
            for k2 in aggregate_results[k1]:
                if len(aggregate_results[k1][k2]) > 0:
                    aggregate_results[k1][k2] = [
                        multi_to_single([e[0] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[1] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[2] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[3] for e in aggregate_results[k1][k2]], tag='acc'),
                    ]
            
        if len(aggregate_results) == 0:
            print('Not data to plot')
            return None

        for key, result_list in aggregate_results.items():
            plot_one_image(key, result_list)
                
    def parse_dict_keys(self, config, front=''):
        dict_keys = []
        for key in config:
            if isinstance(config[key], dict):
                if len(front) == 0:
                    dict_keys += self.parse_dict_keys(config[key], key)
                else:
                    dict_keys += self.parse_dict_keys(config[key], front + '$$' + key)
            else:
                if len(front) == 0:
                    dict_keys.append(key)
                else:
                    dict_keys.append(front + '$$' + key)
        return dict_keys

    def recursive_retrieve(self, dict_data, string_keys):
        string_keys = string_keys.split('$$')
        for i in range(len(string_keys)):
            key = string_keys[i]
            if key not in dict_data:
                return None
            if isinstance(dict_data[key], dict) and i < (len(string_keys)-1):
                return self.recursive_retrieve(dict_data[key], '$$'.join(string_keys[1:]))
            else:
                return dict_data[key]

    def retrieve_diff_configs(self):
        results = []
        for i in range(len(self.configs)):
            results.append([
                self.recursive_retrieve(self.configs[i], e)
                for e in self.diff_keys if e not in self.omit_keys
            ])
        return results

    def parse_results(self):
        results = []
        for i in range(len(self.results)):
            self.csv_result_keys[1][0] = 'best_metric$$test_%s' %\
                                         self.configs[i]['model_config']['MLModel']['metrics'][0]
            tmp = []
            for key, process_func in self.csv_result_keys:
                tmp += process_func(self.recursive_retrieve(self.results[i], key))
            results.append(tmp)
        return results

    def aggregate_csv_results(self):
        import numpy as np
        average_results = {}
        for i in range(len(self.configs_diff)):
            key = '$$'.join([str(e) for e in self.configs_diff[i]])
            if key not in average_results:
                average_results[key] = []
            average_results[key].append(self.csv_results[i])
        results = [['Repeat'] + [e.split('$$')[-1] for e in self.diff_keys if e not in self.omit_keys]
                   + [e[0] for e in self.csv_result_keys]]
        for key in average_results:
            average = []
            std = []
            for k in range(len(average_results[key][0])):
                tmp = []
                for j in range(len(average_results[key])):
                    if average_results[key][j][k] is not None:
                        tmp.append(average_results[key][j][k])
                if len(tmp) > 0:
                    average.append('%.5f' % np.mean(tmp))
                    std.append('%.5f' % np.std(tmp))
                else:
                    average.append('NA')
                    std.append('NA')
            results.append(
                [len(average_results[key])] + key.split('$$') +
                ['%s(%s)' % (average[i], std[i]) for i in range(len(average))]
            )
        return results

    def to_csv(self, file_name='average_results.csv'):
        with open(file_name, 'w') as f:
            for e in self.average_results:
                f.write(', '.join([str(e1) for e1 in e]) + '\n')


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


def _load_config(path):
    with open(os.path.join(path, '1_data_config.yml'), 'r') as f:
        c1 = yaml.safe_load(f)
    with open(os.path.join(path, '2_model_config.yml'), 'r') as f:
        c2 = yaml.safe_load(f)
    with open(os.path.join(path, '3_runtime_config.yml'), 'r') as f:
        c3 = yaml.safe_load(f)
    return c1, c2, c3


def _save_config(c1, c2, c3, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, '1_data_config.yml'), 'w') as f:
        yaml.dump(c1, f)
    with open(os.path.join(path, '2_model_config.yml'), 'w') as f:
        yaml.dump(c2, f)
    with open(os.path.join(path, '3_runtime_config.yml'), 'w') as f:
        yaml.dump(c3, f)


def recursive_update_dict(target: dict, update: dict):
    for key in update:
        if key in target and isinstance(update[key], dict):
            target[key] = recursive_update_dict(target[key], update[key])
        else:
            target[key] = update[key]
    return target


def run(execution, mode, config, new_config_dir_path=None, **kwargs):

    if len(kwargs) > 0:
        print('*' * 40)
        print('Received parameter update')
        print(kwargs)
        print('*' * 40)

    data_config, model_config, runtime_config = _load_config(config)
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

    new_config_dir_path = new_config_dir_path or config
    cfg_mgr = ConfigurationManager(data_config, model_config, runtime_config)
    cfg_mgr.to_files(new_config_dir_path)
    rt_cfg = cfg_mgr.runtime_config

    if execution == 'upload':
        print('Uploading to the server')
        if rt_cfg.machines is None:
            raise ValueError('No machine config found, please check',
                             os.path.join(config, '3_runtime_config.yml'))
        upload_to_server(local_dirs=['FedEval', 'configs'], file_type=(
            '.py', '.yml', '.css', '.html', 'eot', 'svg', 'ttf', 'woff'))
        exit(0)

    if execution == 'stop':
        if mode == 'local':
            local_stop()
        if mode == 'remote':
            server_stop()
        exit(0)

    UNIFIED_JOB_ID = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')

    if execution == 'simulate_fedsgd':
        fed_sgd_simulator(UNIFIED_JOB_ID)
        exit(0)

    if mode == 'local':
        current_path = os.path.abspath('./')
        os.system(
            sudo + f'docker run -it --rm '
                   f'-e UNIFIED_JOB_ID={UNIFIED_JOB_ID} '
                   f'-v {current_path}:{current_path} '
                   f'-w {current_path} {rt_cfg.image_label} '
                   f'python3 -W ignore -m FedEval.run -f data -c {new_config_dir_path}'
        )
        os.system(
            sudo + f'docker run -it --rm '
                   f'-e UNIFIED_JOB_ID={UNIFIED_JOB_ID} '
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
                       f'-e UNIFIED_JOB_ID={UNIFIED_JOB_ID} '
                       f'-v {remote_path}:{remote_path} '
                       f'-w {remote_path} {rt_cfg.image_label} '
                       f'python3 -W ignore -m FedEval.run -f data -c {new_config_dir_path}'
            )
            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

            _, stdout, stderr = ssh.exec_command(
                sudo + f'docker run -it --rm '
                       f'-e UNIFIED_JOB_ID={UNIFIED_JOB_ID} '
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


def _compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
        loss = loss_op(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
    return [e.numpy() for e in gradients]


def fed_sgd_simulator(UNIFIED_JOB_ID):
    import hickle
    from FedEval.utils import ParamParser
    from FedEval.run import generate_data
    data_config = ConfigurationManager().data_config
    model_config = ConfigurationManager().model_config
    runtime_config = ConfigurationManager().runtime_config
    # rm the data
    shutil.rmtree(data_config.dir_name, ignore_errors=True)
    # and regenerate
    generate_data(True)
    client_data_name = [
        os.path.join(data_config.dir_name, e) for e in os.listdir(data_config.dir_name) if e.startswith('client')
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
    for epoch in range(model_config.max_round_num):
        batch_size = 8192
        batched_gradients = []
        actual_size = []
        for i in range(0, len(x_train), batch_size):
            actual_size.append(min(batch_size, len(x_train) - i))
            batched_gradients.append(
                [e / float(actual_size[-1]) for e in
                 _compute_gradients(ml_model, x_train[i:i + batch_size], y_train[i:i + batch_size])]
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
        del batched_gradients
        del actual_size
        test_metric_each_round.append([epoch] + test_log)
    output_dir = os.path.join(runtime_config.log_dir_path, 'fed_sgd_simulator')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{UNIFIED_JOB_ID}_fed_sgd_simulator.csv'), 'w') as f:
        f.write(', '.join(
            [str(e) for e in [data_config.dataset_name, runtime_config.client_num, model_config.learning_rate]]
        ) + '\n')
        for e in test_metric_each_round:
            f.write(', '.join([str(e1) for e1 in e]) + '\n')
        f.write(f'Best Metric, {best_test_metric[0]}, {best_test_metric[1]}')
    # rm the data
    shutil.rmtree(data_config.dir_name)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--execute', '-e', choices=('run', 'stop', 'upload', 'log', 'simulate_fedsgd'),
                             help='Start or Stop the experiments')
    args_parser.add_argument('--mode', '-m', choices=('remote', 'local'),
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
        run(execution=args.execute, mode=args.mode, config=args.config, new_config_dir_path=args.config + '_tmp')
