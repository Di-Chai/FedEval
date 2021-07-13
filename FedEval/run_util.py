import os
import json
import time
import yaml
import copy
import argparse
import paramiko
import requests
import platform
import numpy as np


class LogAnalysis:

    def __init__(self, log_dir=os.path.join('log', 'server')):
        self.log_dir = log_dir
        self.logs = os.listdir(log_dir)

        self.configs = []
        self.results = []
        for log in self.logs:
            c1, c2, c3 = load_config(os.path.join(log_dir, log))
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
            self.csv_result_keys[0][0] = 'best_metric$$test_%s' %\
                                         self.configs[i]['model_config']['MLModel']['metrics'][0]
            tmp = []
            for key, process_func in self.csv_result_keys:
                tmp += process_func(self.recursive_retrieve(self.results[i], key))
            results.append(tmp)
        return results

    def take_average(self):
        average_results = {}
        for i in range(len(self.configs_diff)):
            key = '$$'.join([str(e) for e in self.configs_diff[i]])
            if key not in average_results:
                average_results[key] = []
            average_results[key].append(self.csv_results[i])
        results = [['Repeat'] + [e.split('$$')[-1] for e in self.diff_keys if e not in self.omit_keys]
                   + [e[0] for e in self.csv_result_keys]]
        for key in average_results:
            average = np.mean(average_results[key], axis=0)
            std = np.std(average_results[key], axis=0)
            results.append(
                [len(average_results[key])] + key.split('$$') +
                ['%.5f(%.5f)' % (average[i], std[i]) for i in range(len(average))]
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
    os.system('sudo docker-compose stop')


def server_stop(runtime_config):

    for m_name in runtime_config['machines']:

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        host = runtime_config['machines'][m_name]['host']
        port = runtime_config['machines'][m_name]['port']
        user_name = runtime_config['machines'][m_name]['user_name']
        remote_path = runtime_config['machines'][m_name]['dir']

        key_file = runtime_config['machines'][m_name]['key']
        ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)
        if m_name == 'server':
            stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) +
                                                     'sudo docker-compose -f docker-compose-server.yml stop')
        else:
            stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) +
                                                     'sudo docker-compose -f docker-compose-{}.yml stop'.format(m_name))

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


def upload_to_server(machines, local_dirs, file_type=('.py', '.yml', '.css', '.html', 'eot', 'svg', 'ttf', 'woff')):
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
    for m_name in machines:
        host = machines[m_name]['host']
        port = machines[m_name]['port']
        user_name = machines[m_name]['user_name']
        remote_path = machines[m_name]['dir']
        key_file = machines[m_name]['key']
        if (host+str(port)) in host_record:
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
        host_record.append(host+str(port))


def download_from_server(machines, remote_dirs, file_type):

    def download_check(file_name):
        for ft in file_type:
            if file_name.endswith(ft):
                return True
        return False

    host_record = []
    for m_name in machines:
        host = machines[m_name]['host']
        port = machines[m_name]['port']
        user_name = machines[m_name]['user_name']
        remote_path = machines[m_name]['dir']
        key_file = machines[m_name]['key']
        if (host + str(port)) in host_record:
            continue

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
        host_record.append(host + str(port))


def load_config(path):
    with open(os.path.join(path, '1_data_config.yml'), 'r') as f:
        c1 = yaml.load(f)
    with open(os.path.join(path, '2_model_config.yml'), 'r') as f:
        c2 = yaml.load(f)
    with open(os.path.join(path, '3_runtime_config.yml'), 'r') as f:
        c3 = yaml.load(f)
    return c1, c2, c3


def save_config(c1, c2, c3, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, '1_data_config.yml'), 'w') as f:
        yaml.dump(c1, f)
    with open(os.path.join(path, '2_model_config.yml'), 'w') as f:
        yaml.dump(c2, f)
    with open(os.path.join(path, '3_runtime_config.yml'), 'w') as f:
        yaml.dump(c3, f)


def recursive_update_dict(target, update):
    for key in update:
        if key in target:
            if isinstance(update[key], dict):
                target[key] = recursive_update_dict(target[key], update[key])
            else:
                target[key] = update[key]
        else:
            target[key] = update[key]
    return target


def run(execution, mode, config, new_config=None, **kwargs):

    if len(kwargs) > 0:
        print('*' * 40)
        print('Received parameter update')
        print(kwargs)
        print('*' * 40)

    data_config, model_config, runtime_config = load_config(config)

    if execution == 'stop':
        if mode == 'local':
            local_stop()
        if mode == 'server':
            server_stop(runtime_config)
        exit(0)

    if execution == 'upload':
        print('Uploading to the server')
        if len(runtime_config.get('machines', None)) is None:
            raise ValueError('No machine config found, pleck check', os.path.join(config, '3_runtime_config.yml'))
        upload_to_server(
            runtime_config['machines'], local_dirs=['FedEval', 'configs'],
            file_type=('.py', '.yml', '.css', '.html', 'eot', 'svg', 'ttf', 'woff')
        )
        exit(0)

    if 'data_config' in kwargs:
        data_config = recursive_update_dict(data_config, kwargs['data_config'])
    if 'model_config' in kwargs:
        model_config = recursive_update_dict(model_config, kwargs['model_config'])
    if 'runtime_config' in kwargs:
        runtime_config = recursive_update_dict(runtime_config, kwargs['runtime_config'])

    new_config = new_config or config

    if mode == 'local':
        runtime_config['server']['host'] = 'server'
        runtime_config['server']['listen'] = 'server'
    if mode == 'server':
        runtime_config['server']['host'] = runtime_config['machines']['server']['host']
        runtime_config['server']['listen'] = '0.0.0.0'

    save_config(data_config, model_config, runtime_config, new_config)

    if mode == 'local':
        current_path = os.path.abspath('./')
        os.system(
            'sudo docker run -it --rm -v {0}:{0} -w {0} '
            '{1} python3 -W ignore -m FedEval.run -f data -c {2}'
            .format(current_path, runtime_config['docker']['image'], new_config)
        )
        os.system(
            'sudo docker run -it --rm -v {0}:{0} -w {0} '
            '{1} python3 -W ignore -m FedEval.run -f compose-local -c {2}'
            .format(current_path, runtime_config['docker']['image'], new_config)
        )
        os.system('sudo docker-compose up -d')

    if mode == 'server':

        upload_to_server(runtime_config['machines'], local_dirs=('FedEval', 'configs'))
        # upload_to_server(runtime_config['machines'], local_dirs=(new_config,))

        machine_name_list = list(runtime_config['machines'].keys())
        machine_name_list.remove('server')

        for m_name in ['server'] + machine_name_list:

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            host = runtime_config['machines'][m_name]['host']
            port = runtime_config['machines'][m_name]['port']
            user_name = runtime_config['machines'][m_name]['user_name']
            remote_path = runtime_config['machines'][m_name]['dir']

            key_file = runtime_config['machines'][m_name]['key']
            ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)

            if m_name != 'server':
                _, stdout, stderr = ssh.exec_command(
                    'sudo docker run -i --rm -v {0}:{0} -w {0} '
                    '{1} python3 -W ignore -m FedEval.run -f data -c {2}'
                    .format(remote_path, runtime_config['docker']['image'], new_config)
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

            _, stdout, stderr = ssh.exec_command(
                'sudo docker run -i --rm -v {0}:{0} -w {0} '
                '{1} python3 -W ignore -m FedEval.run -f compose-server -c {2}'
                .format(remote_path, runtime_config['docker']['image'], new_config)
            )
            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

            if m_name == 'server':
                print('Start Server')
                _, stdout, stderr = ssh.exec_command(
                    'cd {};'.format(remote_path) +
                    'sudo docker-compose -f docker-compose-server.yml up -d')
            else:
                print('Start Clients', m_name)
                _, stdout, stderr = ssh.exec_command(
                    'cd {};'.format(remote_path) +
                    'sudo docker-compose -f docker-compose-{}.yml up -d'.format(m_name))

            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

    print('Start succeed!')

    time.sleep(20)

    host = '127.0.0.1' if mode == 'local' else runtime_config['server']['host']
    port = runtime_config['server']['port']

    print('Starting to monitor at %s, check every 10 seconds' % ('http://{}/status'.format(host + ':' + str(port))))
    print('Check the dashboard at %s' % ('http://{}/dashboard'.format(host + ':' + str(port))))

    check_status_result = check_status(host + ':' + str(port))

    while True:
        if check_status_result['success']:
            if not check_status_result['data'].get('finished', False):
                print('Running at Round %s' % check_status_result['data'].get('rounds', -2))
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
            download_from_server(
                {'server': runtime_config['machines']['server']}, remote_dirs=[log_dir],
                file_type=['.yml', '.json', '.log']
            )

    if mode == 'local':
        local_stop()

    if mode == 'server':
        server_stop(runtime_config)


def local_central_trial(config, output_file=None, **kwargs):
    from .role import NormalTrain
    data_config, model_config, runtime_config = load_config(config)
    if 'data_config' in kwargs:
        data_config = recursive_update_dict(data_config, kwargs['data_config'])
    if 'model_config' in kwargs:
        model_config = recursive_update_dict(model_config, kwargs['model_config'])
    if 'runtime_config' in kwargs:
        runtime_config = recursive_update_dict(runtime_config, kwargs['runtime_config'])
    local_central_train = NormalTrain(
        data_config=data_config, model_config=model_config, runtime_config=runtime_config
    )
    output_file = output_file or 'local_central_trial.csv'
    local_central_train.run(output_file)


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
        run(execution=args.execute, mode=args.mode, config=args.config, new_config=args.config + '_tmp')
