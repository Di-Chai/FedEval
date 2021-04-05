import os
import json
import time
import yaml
import copy
import argparse
import requests

from .role import NormalTrain


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
    import paramiko
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


def upload_to_server(runtime_config, file_list=()):
    if len(file_list) > 0:
        scp_orders = ['scp -P {} -i {} -r ' + e + ' {}@{}:{}' for e in file_list]

        for m_name in runtime_config['machines']:
            host = runtime_config['machines'][m_name]['host']
            port = runtime_config['machines'][m_name]['port']
            user_name = runtime_config['machines'][m_name]['user_name']
            remote_path = runtime_config['machines'][m_name]['dir']
            key_file = runtime_config['machines'][m_name]['key']
            for instruction in scp_orders:
                os.system(instruction.format(port, key_file, user_name, host, remote_path))


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


def run(exec, mode, config, new_config=None, **kwargs):

    data_config, model_config, runtime_config = load_config(config)

    if exec == 'stop':
        if mode == 'local':
            local_stop()
        if mode == 'server':
            server_stop(runtime_config)
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
            '{1} python3 -m FedEval.run -f data -c {2}'
            .format(current_path, runtime_config['docker']['image'], new_config)
        )
        os.system(
            'sudo docker run -it --rm -v {0}:{0} -w {0} '
            '{1} python3 -m FedEval.run -f compose-local -c {2}'
            .format(current_path, runtime_config['docker']['image'], new_config)
        )
        os.system('sudo docker-compose up -d')

    if mode == 'server':

        upload_to_server(runtime_config, file_list=('FedEval', 'configs'))

        import paramiko
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
                    '{1} python3 -m FedEval.run -f data -c {2}'
                    .format(remote_path, runtime_config['docker']['image'], new_config)
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

            _, stdout, stderr = ssh.exec_command(
                'sudo docker run -i --rm -v {0}:{0} -w {0} '
                '{1} python3 -m FedEval.run -f compose-server -c {2}'
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
            server = runtime_config['machines']['server']
            os.system('scp -P {} -i {} {}@{}:{} ./{}'.format(
                server['port'], server['key'], server['user_name'],
                server['host'], server['dir'] + '/' + result_file, log_dir)
            )
            os.system('scp -P {} -i {} {}@{}:{} ./{}'.format(
                server['port'], server['key'], server['user_name'],
                server['host'], server['dir'] + '/' + log_file, log_dir)
            )
            save_config(data_config, model_config, runtime_config, log_dir)

    if mode == 'local':
        local_stop()

    if mode == 'server':
        server_stop(runtime_config)


def local_central_trial(config, output_file=None, **kwargs):
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
