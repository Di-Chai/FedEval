import os
import json
import time
import yaml
import argparse
import requests
import paramiko


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--path', default='./')
args_parser.add_argument('--dataset', default='cifar10')
args_parser.add_argument('--model', default='LeNet')
args_parser.add_argument('--optimizer', default='adam')
args_parser.add_argument('--upload_name_filter', default='')
args_parser.add_argument('--upload_sparse', default=1.0, type=float)
args_parser.add_argument('--upload_strategy', default='no-compress')
args_parser.add_argument('--B', type=int, default=1)
args_parser.add_argument('--C', type=float, default=1.0)
args_parser.add_argument('--E', type=int, default=1)
args_parser.add_argument('--num_clients', type=int, default=100)
args_parser.add_argument('--max_epochs', type=int, default=5000)
args_parser.add_argument('--non-iid', type=int, default=0)
args_parser.add_argument('--non-iid-strategy', default='iid')
args_parser.add_argument('--lr', default=5e-4)
args_parser.add_argument('--file_name', default='experiment_result.txt')
args_parser.add_argument('--sudo', default='sudo')

scp_orders = [
    'scp -P {} -i {} *.py *.yml {}@{}:{}',
    'scp -P {} -i {} -r tf_wrapper {}@{}:{}/',
    'scp -P {} -i {} -r Trials {}@{}:{}/',
    'scp -P {} -i {} -r configs {}@{}:{}/',
    'scp -P {} -i {} -r FedEval/dataset {}@{}:{}/FedEval/',
    'scp -P {} -i {} -r FedEval/attack {}@{}:{}/FedEval/',
    'scp -P {} -i {} -r FedEval/role {}@{}:{}/FedEval/',
    'scp -P {} -i {} -r FedEval/model {}@{}:{}/FedEval/',
    'scp -P {} -i {} -r FedEval/utils {}@{}:{}/FedEval/',
]

args = args_parser.parse_args()

with open(os.path.join(args.path, '1_data_config.yml'), 'r') as f:
    data_config = yaml.load(f)

with open(os.path.join(args.path, '2_model_config.yml'), 'r') as f:
    model_config = yaml.load(f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'r') as f:
    runtime_config = yaml.load(f)


machines = runtime_config['machines'].copy()
server_host = machines['server']['host']
server = machines.pop('server')
machine_list = [[e[0]] + [e[1]['host'], e[1]['port'], e[1]['user_name'], e[1]['key'], e[1]['dir']]
                for e in machines.items()]
machine_list = [['server', server['host'], server['port'], server['user_name'],
                 server['key'], server['dir']]] + machine_list
current_path = os.path.dirname(os.path.abspath(__file__))


def check_status():
    try:
        status = requests.get('http://{}:8200/status'.format(server_host))
        return {'success': True, 'data': json.loads(status.text)}
    except Exception as e:
        print('Error in checking', e)
        return {'success': False, 'data': None}


data_config['dataset'] = args.dataset

model_config['Model'] = args.model
model_config[args.model]['optimizer'] = args.optimizer
model_config['upload']['upload_sparse'] = args.upload_sparse
model_config['upload']['upload_strategy'] = args.upload_strategy

if len(args.upload_name_filter) > 0 and args.upload_name_filter.lower != 'none':
    model_config['upload']['upload_name_filter'] = args.upload_name_filter.split(',')
else:
    model_config['upload']['upload_name_filter'] = []

runtime_config['server']['host'] = server_host
runtime_config['server']['listen'] = '0.0.0.0'
runtime_config['server']['MAX_NUM_ROUNDS'] = args.max_epochs
runtime_config['server']['MIN_NUM_WORKERS'] = args.num_clients
runtime_config['server']['NUM_CLIENTS_CONTACTED_PER_ROUND'] = int(runtime_config['server']['MIN_NUM_WORKERS'] * args.C)
runtime_config['clients']['local_batch_size'] = args.B
runtime_config['clients']['local_rounds'] = args.E
runtime_config['clients']['num_clients'] = args.num_clients
runtime_config['clients']['lr'] = args.lr

data_config['non-iid'] = args.non_iid
data_config['non-iid-strategy'] = args.non_iid_strategy

with open(os.path.join(args.path, '1_data_config.yml'), 'w') as f:
    yaml.dump(data_config, f)

with open(os.path.join(args.path, '2_model_config.yml'), 'w') as f:
    yaml.dump(model_config, f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'w') as f:
    yaml.dump(runtime_config, f)

for m_name, host, port, user_name, key_file, remote_path in machine_list:
    for instruction in scp_orders:
        os.system(instruction.format(port, key_file, user_name, host, remote_path))

sudo = 'sudo ' if args.sudo == 'sudo' else ''

stdout_wait_list = []
for m_name, host, port, user_name, key_file, remote_path in machine_list:

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)

    if m_name != 'server':
        _, stdout, _ = ssh.exec_command(
            sudo + 'docker run --rm -v {0}:{0} -w {0} '
                   '{1} python3 4_distribute_data.py --path {2}'.format(remote_path, runtime_config['docker']['image'],
                                                                        args.path)
        )
        stdout.readlines()

    _, stdout, _ = ssh.exec_command(
        sudo + 'docker run --rm -v {0}:{0} -w {0} '
               '{1} python3 5_generate_docker_compose_server.py --path {2}'.format(remote_path,
                                                                                   runtime_config['docker']['image'],
                                                                                   args.path)
    )
    stdout.readlines()

    if m_name == 'server':
        print('Start Server')
        _, stdout, _ = ssh.exec_command(
            'cd {};'.format(remote_path) + sudo +
            'docker-compose -f docker-compose-server.yml up -d')
    else:
        print('Start Clients', m_name)
        _, stdout, _ = ssh.exec_command(
            'cd {};'.format(remote_path) + sudo +
            'docker-compose -f docker-compose-{}.yml up -d'.format(m_name))

    stdout_wait_list.append(stdout)

for stdout in stdout_wait_list:
    stdout.readlines()

print('Start success')

sleep_time = 30

time.sleep(sleep_time)

check_status_result = check_status()

while True:
    if check_status_result['success']:
        if not check_status_result['data'].get('status', False):
            print('Running at Round %s' % check_status_result['data'].get('rounds', -2))
            time.sleep(sleep_time)
        else:
            break
    else:
        print('Check failed, try later')
        time.sleep(sleep_time)
    check_status_result = check_status()

status_data = check_status_result['data']
if status_data is not None:

    log_file = status_data['log_file']

    os.system('scp -P {} -i {} {}@{}:{} ./log'.format(server['port'], server['key'], server['user_name'],
                                                      server['host'], server['dir'] + '/' + log_file))

    with open('log/' + log_file.split('/')[-1], 'r') as f:
        log_data = f.readlines()

    try:
        from screen_short import get_image
        get_image('http://{}:{}/dashboard'.format(server_host, server['port']),
                  'log/' + log_file.split('/')[-1].replace('.log', '.png'))
    except Exception as e:
        print('Screen shot failed', e)

    log_data = log_data[-10:]
    best_metrics = [e.split('-')[-1].strip(' \n') for e in log_data]
    best_test_accuracy = [e for e in best_metrics if 'get best' in e and 'accuracy' in e]
    best_test_accuracy = '%.5f' % float(best_test_accuracy[0].split(' ')[-1])

    total_time = log_data[-5].strip(' \n').split(' ')[-1]
    time_detail = eval(log_data[-4].strip(' \n').split('Time Detail:')[-1])
    time_detail = ['%.5f' % e for e in time_detail]
    total_rounds = log_data[-3].strip(' \n').split(' ')[-1]
    server_send = '%.5f' % float(log_data[-2].strip(' \n').split(' ')[-1])
    server_receive = '%.5f' % float(log_data[-1].strip(' \n').split(' ')[-1])

    result_list = [args.dataset, args.model, args.optimizer, args.upload_name_filter,
                   args.non_iid, args.non_iid_strategy,
                   args.upload_strategy, args.upload_sparse,
                   args.B, args.C, args.E, runtime_config['clients']['lr'],
                   runtime_config['server']['NUM_TOLERATE'], 'PC', 'None', 'None',
                   best_test_accuracy, total_time] + time_detail + \
                  [total_rounds, server_send, server_receive, log_file.split('/')[-1]]
    result_list = [str(e) for e in result_list]

    with open(args.file_name, 'a+') as f:
        f.write(', '.join(result_list) + '\n')


for m_name, host, port, user_name, key_file, remote_path in machine_list:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=user_name, key_filename=key_file)
    if m_name == 'server':
        stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) + sudo +
                                                 'docker-compose -f docker-compose-server.yml stop')
        stdout.readlines()
    else:
        stdin, stdout, stderr = ssh.exec_command('cd {};'.format(remote_path) + sudo +
                                                 'docker-compose -f docker-compose-{}.yml stop'.format(m_name))
        stdout.readlines()
