import os
import json
import time
import yaml
import argparse
import requests


def check_status(host):
    try:
        status = requests.get('http://{}/status'.format(host))
        return {'success': True, 'data': json.loads(status.text)}
    except Exception as e:
        print('Error in checking', e)
        return {'success': False, 'data': None}


def local_stop(sudo):
    os.system(sudo + 'docker-compose stop')


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
            stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) + sudo +
                                                     'docker-compose -f docker-compose-server.yml stop')
        else:
            stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) + sudo +
                                                     'docker-compose -f docker-compose-{}.yml stop'.format(m_name))

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


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--exec', choices=('run', 'stop', 'upload'), required=True,
                             help="Please specify the exec type: run, stop, or upload")
    args_parser.add_argument('--mode', choices=('local', 'server'), required=True,
                             help="local for running on local Linux machine, set to server if you want to use "
                                  "remote server")
    args_parser.add_argument('--file', default='',
                             help='[Only work when mode=server] '
                                  'Files listed here will be automatically '
                                  'uploaded to the remote server in the config, '
                                  'e.g., --file FedEval,tf_wrapper,*.py '
                                  'leave the param empty if you already synced the files.')
    args_parser.add_argument('--config', help='Path to the config files (1_data_config.yml, 2.., 3..)', required=True)
    args_parser.add_argument('--dataset', default='mnist', help='dataset name, default to be mnist')
    args_parser.add_argument('--ml_model', default='LeNet', help='machine learning model name, default to be LeNet')
    args_parser.add_argument('--fed_model', default='FedSGD', help='federated model name, default to be FedSGD')
    args_parser.add_argument('--optimizer', default='adam')
    args_parser.add_argument('--upload_optimizer', default='False')
    args_parser.add_argument('--upload_sparsity', default=1.0, type=float)
    args_parser.add_argument('--upload_dismiss', default='')
    args_parser.add_argument('--lazy_update', default='True')
    args_parser.add_argument('--B', type=int, default=1)
    args_parser.add_argument('--C', type=float, default=1.0)
    args_parser.add_argument('--E', type=int, default=1)
    args_parser.add_argument('--num_tolerance', type=int, default=20)
    args_parser.add_argument('--num_clients', type=int, default=100)
    args_parser.add_argument('--max_epochs', type=int, default=1000)
    args_parser.add_argument('--non-iid', type=int, default=0)
    args_parser.add_argument('--non-iid-strategy', default='iid')
    args_parser.add_argument('--lr', default=5e-4)
    args_parser.add_argument('--output', default='experiment_result.csv')
    args_parser.add_argument('--sudo', default='True')

    current_path = os.path.abspath('./')
    args = args_parser.parse_args()

    sudo = 'sudo ' if args.sudo == 'True' else ''

    with open(os.path.join(args.config, '1_data_config.yml'), 'r') as f:
        data_config = yaml.load(f)

    with open(os.path.join(args.config, '2_model_config.yml'), 'r') as f:
        model_config = yaml.load(f)

    with open(os.path.join(args.config, '3_runtime_config.yml'), 'r') as f:
        runtime_config = yaml.load(f)

    if args.exec == 'stop':
        if args.mode == 'local':
            local_stop(sudo)
        if args.mode == 'server':
            server_stop(runtime_config)
        exit(0)

    if args.mode == 'server' and args.exec == 'upload':
        upload_to_server(runtime_config, args.file.split(','))
        exit(0)

    data_config['dataset'] = args.dataset
    data_config['non-iid'] = args.non_iid
    data_config['non-iid-strategy'] = args.non_iid_strategy

    model_config['MLModel']['name'] = args.ml_model
    model_config['MLModel'][args.ml_model]['optimizer'] = args.optimizer
    model_config['MLModel'][args.ml_model]['lr'] = args.lr
    model_config['FedModel']['name'] = args.fed_model
    model_config['FedModel']['upload_strategy']['upload_sparsity'] = args.upload_sparsity
    model_config['FedModel']['upload_strategy']['upload_optimizer'] = eval(args.upload_optimizer)
    model_config['FedModel']['train_strategy']['max_num_rounds'] = args.max_epochs
    model_config['FedModel']['train_strategy']['B'] = args.B
    model_config['FedModel']['train_strategy']['C'] = args.C
    model_config['FedModel']['train_strategy']['E'] = args.E
    model_config['FedModel']['train_strategy']['lazy_update'] = args.lazy_update
    model_config['FedModel']['train_strategy']['num_tolerance'] = args.num_tolerance

    if len(args.upload_dismiss) > 0 and args.upload_dismiss.lower != 'none':
        model_config['FedModel']['upload_strategy']['upload_dismiss'] = args.upload_dismiss.split(',')
    else:
        model_config['FedModel']['upload_strategy']['upload_dismiss'] = []

    if args.mode == 'local':
        runtime_config['server']['host'] = 'server'
        runtime_config['server']['listen'] = 'server'
    if args.mode == 'server':
        runtime_config['server']['host'] = runtime_config['machines']['server']['host']
        runtime_config['server']['listen'] = '0.0.0.0'
    runtime_config['server']['num_clients'] = args.num_clients

    with open(os.path.join(args.config, '1_data_config.yml'), 'w') as f:
        yaml.dump(data_config, f)

    with open(os.path.join(args.config, '2_model_config.yml'), 'w') as f:
        yaml.dump(model_config, f)

    with open(os.path.join(args.config, '3_runtime_config.yml'), 'w') as f:
        yaml.dump(runtime_config, f)

    if args.mode == 'local':
        os.system(
            sudo + 'docker run -it --rm -v {0}:{0} -w {0} '
                   '{1} python3 -m FedEval.run -f data -c {2}'
            .format(current_path, runtime_config['docker']['image'], args.config)
        )
        os.system(
            sudo + 'docker run -it --rm -v {0}:{0} -w {0} '
                   '{1} python3 -m FedEval.run -f compose-local -c {2}'
            .format(current_path, runtime_config['docker']['image'], args.config)
        )
        os.system(sudo + 'docker-compose up -d')

    if args.mode == 'server':

        import paramiko

        # Upload files if provided
        upload_to_server(runtime_config, args.file.split(','))

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
                    sudo + 'docker run -i --rm -v {0}:{0} -w {0} '
                           '{1} python3 -m FedEval.run -f data -c {2}'
                    .format(remote_path, runtime_config['docker']['image'], args.config)
                )
                print(''.join(stdout.readlines()))
                print(''.join(stderr.readlines()))

            _, stdout, stderr = ssh.exec_command(
                sudo + 'docker run -i --rm -v {0}:{0} -w {0} '
                       '{1} python3 -m FedEval.run -f compose-server -c {2}'
                .format(remote_path, runtime_config['docker']['image'], args.config)
            )
            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

            if m_name == 'server':
                print('Start Server')
                _, stdout, stderr = ssh.exec_command(
                    'cd {};'.format(remote_path) + sudo +
                    'docker-compose -f docker-compose-server.yml up -d')
            else:
                print('Start Clients', m_name)
                _, stdout, stderr = ssh.exec_command(
                    'cd {};'.format(remote_path) + sudo +
                    'docker-compose -f docker-compose-{}.yml up -d'.format(m_name))

            print(''.join(stdout.readlines()))
            print(''.join(stderr.readlines()))

    print('Start success')

    time.sleep(20)

    host = '127.0.0.1' if args.mode == 'local' else runtime_config['server']['host']
    port = runtime_config['server']['port']

    check_status_result = check_status(host + ':' + str(port))

    while True:
        if check_status_result['success']:
            if not check_status_result['data'].get('status', False):
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

        if args.mode == 'server':
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
            with open(result_file, 'r') as f:
                results = json.load(f)
        else:
            with open(result_file, 'r') as f:
                results = json.load(f)

        best_test_accuracy = '%.5f' % results['best_metric']['test_accuracy']

        total_time = results['total_time']
        total_rounds = results['total_rounds']
        server_send = results['server_send']
        server_receive = results['server_receive']

        result_list = [
            args.config, args.fed_model, args.dataset, args.ml_model, args.lazy_update, args.optimizer,
            args.upload_optimizer, args.upload_sparsity, args.upload_dismiss,
            args.non_iid, args.non_iid_strategy,
            args.B, args.C, args.E, args.lr, args.num_tolerance, 'None', 'None',
            best_test_accuracy, total_time
        ]

        result_list += ['%.5f' % e for e in eval(results['time_detail'])]
        result_list += [total_rounds, server_send, server_receive, result_file]
        # result_list += ['%.5f' % e for e in results['best_metric_full']['test_accuracy']]

        result_list = [str(e) for e in result_list]

        headers = [
            'Config', 'FedModel', 'dataset', 'ml-model', 'lazy_update', 'optimizer',
            'upload_optimizer', 'upload_sparsity', 'upload_dismiss',
            'IID', 'IID-Strategy', 'B', 'C', 'E', 'lr', 'ESPat', 'LocalAcc', 'CentralAcc', 'FLAcc',
            'TimeAll', 'Init', 'TrainReq', 'TrainRun', 'TrainSync', 'TrainAgg',
            'EvalReq', 'EvalRun', 'EvalSync', 'EvalAgg',
            'CommRound', 'CommAmount(SOut)', 'CommAmount(SIn)', 'ResultFile'
        ]

        if os.path.isfile(args.output) is False:
            with open(args.output, 'w') as f:
                f.write(', '.join(headers) + '\n')

        with open(args.output, 'a+') as f:
            f.write(', '.join(result_list) + '\n')

    if args.mode == 'local':
        local_stop(sudo)

    if args.mode == 'server':
        server_stop(runtime_config)
