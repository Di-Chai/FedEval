import argparse
import copy
import json
import os
import platform
import time

import requests
import yaml

from .config.configuration import ConfigurationManager

sudo = ""


class LogAnalysis:

    def __init__(self, log_dir=os.path.join('log', 'server')):
        self.log_dir = log_dir
        self.logs = os.listdir(log_dir)

        self.configs = []
        self.results = []
        for log in self.logs:
            try:
                c1, c2, c3 = _load_config(os.path.join(log_dir, log))
            except FileNotFoundError:
                print('Config not found in', log, 'skip to next')
                continue
            self.configs.append({'data_config': c1, 'model_config': c2, 'runtime_config': c3})
            print('Get log', log)
            with open(os.path.join(log_dir, log, 'results.json'), 'r') as f:
                self.results.append(json.load(f))

        self.omit_keys = [
            'runtime_config$$machines',
            'runtime_config$$server'
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

        self.average_results = self.take_average()

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
        for key in string_keys:
            if key not in dict_data:
                return None
            if isinstance(dict_data[key], dict):
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

    def take_average(self):
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
        status = requests.get('http://{}/status'.format(host), timeout=(5, 5))
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
            f'cd {remote_path};'+ sudo + f'docker-compose -f docker-compose-{name}.yml stop')

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
    if mode == 'server':
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
        if mode == 'server':
            server_stop()
        exit(0)

    if mode == 'local':
        current_path = os.path.abspath('./')
        os.system(
            sudo + 'docker run -it --rm -v {0}:{0} -w {0} '
            '{1} python3 -W ignore -m FedEval.run -f data -c {2}'
            .format(current_path, rt_cfg.image_label, new_config_dir_path)
        )
        os.system(
            sudo + 'docker run -it --rm -v {0}:{0} -w {0} '
            '{1} python3 -W ignore -m FedEval.run -f compose-local -c {2}'
            .format(current_path, rt_cfg.image_label, new_config_dir_path)
        )
        os.system(sudo + 'docker-compose up -d')

    if mode == 'server':

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

            if not machine.is_server:
                _, stdout, stderr = ssh.exec_command(
                    sudo + 'docker run -i --rm -v {0}:{0} -w {0} '
                    '{1} python3 -W ignore -m FedEval.run -f data -c {2}'
                    .format(remote_path, rt_cfg.image_label, new_config_dir_path)
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

            _, stdout, stderr = ssh.exec_command(
                sudo + 'docker run -i --rm -v {0}:{0} -w {0} '
                '{1} python3 -W ignore -m FedEval.run -f compose-server -c {2}'
                .format(remote_path, rt_cfg.image_label, new_config_dir_path)
            )
            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

            if machine.is_server:
                print('Start Server')
                _, stdout, stderr = ssh.exec_command(
                    f'cd {remote_path};' +
                    sudo + 'docker-compose -f docker-compose-server.yml up -d')
            else:
                print('Start Clients', m_name)
                _, stdout, stderr = ssh.exec_command(
                    f'cd {remote_path};' +
                    sudo + f'docker-compose -f docker-compose-{m_name}.yml up -d')

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

        if mode == 'server':
            os.makedirs(log_dir, exist_ok=True)
            download_from_server(remote_dirs=[log_dir], file_type=[
                                 '.yml', '.json', '.log'])

    if mode == 'local':
        local_stop()

    if mode == 'server':
        server_stop()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--execute', '-e', choices=('run', 'stop', 'upload', 'log'),
                             help='Start or Stop the experiments')
    args_parser.add_argument('--mode', '-m', choices=('server', 'local'),
                             help='Run the experiments locally or at the server that presented the runtime_config')
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
