import os
import json
import time
import yaml
import argparse
import requests

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
args_parser.add_argument('--docker_env', default='YES')

args = args_parser.parse_args()

with open(os.path.join(args.path, '1_data_config.yml'), 'r') as f:
    data_config = yaml.load(f)

with open(os.path.join(args.path, '2_model_config.yml'), 'r') as f:
    model_config = yaml.load(f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'r') as f:
    runtime_config = yaml.load(f)

current_path = os.path.dirname(os.path.abspath(__file__))


def check_status():
    try:
        status = requests.get('http://127.0.0.1:8200/status')
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

runtime_config['server']['host'] = 'server'
runtime_config['server']['listen'] = 'server'
runtime_config['server']['MAX_NUM_ROUNDS'] = args.max_epochs
runtime_config['server']['MIN_NUM_WORKERS'] = args.num_clients
runtime_config['server']['NUM_CLIENTS_CONTACTED_PER_ROUND'] = int(runtime_config['server']['MIN_NUM_WORKERS'] * args.C)
runtime_config['clients']['local_batch_size'] = args.B
runtime_config['clients']['local_rounds'] = args.E
runtime_config['clients']['num_clients'] = args.num_clients
runtime_config['clients']['lr'] = args.lr

data_config['non-iid'] = args.non_iid
data_config['non-iid-strategy'] = args.non_iid_strategy
data_config['upload_strategy'] = args.upload_strategy

with open(os.path.join(args.path, '1_data_config.yml'), 'w') as f:
    yaml.dump(data_config, f)

with open(os.path.join(args.path, '2_model_config.yml'), 'w') as f:
    yaml.dump(model_config, f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'w') as f:
    yaml.dump(runtime_config, f)

sudo = 'sudo ' if args.sudo == 'sudo' else ''

if args.docker_env.upper() == "YES":
    os.system(
        sudo + 'docker run --rm -v {0}:{0} -w {0} '
               '{1} python3 4_distribute_data.py'.format(current_path, runtime_config['docker']['image'])
    )
    os.system(
        sudo + 'docker run --rm -v {0}:{0} -w {0} '
               '{1} python3 5_generate_docker_compose.py'.format(current_path, runtime_config['docker']['image'])
    )
else:
    os.system('python 4_distribute_data.py')
    os.system('python 5_generate_docker_compose.py')

os.system(sudo + 'docker-compose up -d')

print('Start success')

time.sleep(20)

check_status_result = check_status()

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
    check_status_result = check_status()

status_data = check_status_result['data']
if status_data is not None:
    log_file = status_data['log_file']
    with open(log_file, 'r') as f:
        log_data = f.readlines()

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

os.system(sudo + 'docker-compose stop')
