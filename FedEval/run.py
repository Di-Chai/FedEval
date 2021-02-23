import os
import yaml
import pickle
import argparse
import numpy as np

from .model import *
from .dataset import FedImage
from .role import Client, Server


def generate_data(data_config, model_config, runtime_config):
    data = FedImage(dataset=data_config['dataset'],
                    data_dir=data_config['data_dir'],  # for saving
                    flatten=True if model_config['MLModel']['name'] == 'MLP' else False,
                    normalize=data_config['normalize'],
                    train_val_test=data_config['train_val_test'],
                    num_clients=runtime_config['server']['num_clients'])

    if data_config['non-iid']:
        data.iid_data(sample_size=data_config['sample_size'])
    else:
        data.non_iid_data(non_iid_class=data_config['non-iid-class'],
                          strategy=data_config['non-iid-strategy'],
                          shared_data=data_config['shared_data'], sample_size=data_config['sample_size'])


def run(role, data_config, model_config, runtime_config):
    ml_model_name = model_config['MLModel']['name']
    ml_model_configs = model_config['MLModel']
    dataset_name = data_config['dataset']

    # 2 Config Model
    if ml_model_name == "MLP":
        input_params = {
            "inputs_shape": {'x': [np.prod(data_config['input_shape'][dataset_name]['image'])]},
            "targets_shape": {'y': data_config['input_shape'][dataset_name]['label']},
        }
    else:
        input_params = {
            "inputs_shape": {'x': data_config['input_shape'][dataset_name]['image']},
            "targets_shape": {'y': data_config['input_shape'][dataset_name]['label']},
        }

    ml_model_configs[ml_model_name].update(input_params)
    model = eval(ml_model_name + "(**ml_model_configs[ml_model_name])")

    if role == 'client':
        # 3 Config data
        client_id = os.environ.get('CLIENT_ID', '0')
        with open(os.path.join(data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
            data = pickle.load(f)

        # 4 Init the client
        Client(server_host=runtime_config['server']['host'],
               server_port=runtime_config['server']['port'],
               model=model,
               train_data={'x': data['x_train'], 'y': data['y_train']},
               val_data={'x': data['x_val'], 'y': data['y_val']},
               test_data={'x': data['x_test'], 'y': data['y_test']},
               fed_model_name=model_config['FedModel']['name'],
               train_strategy=model_config['FedModel']['train_strategy'],
               upload_strategy=model_config['FedModel']['upload_strategy'],
               client_name="Client_%s" % client_id)

    if role == 'server':
        # 3 Init the server
        server = Server(model=model, server_config=runtime_config['server'],
                        fed_model_name=model_config['FedModel']['name'],
                        train_strategy=model_config['FedModel']['train_strategy'],
                        upload_strategy=model_config['FedModel']['upload_strategy'])
        server.start()


def generate_docker_compose_server(runtime_config, path):
    
    project_path = os.path.abspath('./')

    server_template = {
        'image': runtime_config['docker']['image'],
        'ports': ['{0}:{0}'.format(runtime_config['server']['port'])],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'command': 'sh -c "python3 -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server'
    }

    client_template = {
        'image': runtime_config['docker']['image'],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
    }

    with open('docker-compose-server.yml', 'w') as f:
        no_alias_dumper = yaml.dumper.SafeDumper
        no_alias_dumper.ignore_aliases = lambda self, data: True
        yaml.dump({
            'version': "2",
            'services': {'server': server_template}
        }, f, default_flow_style=False, Dumper=no_alias_dumper)

    machines = runtime_config['machines']
    machines.pop('server')
    assert sum([v['capacity'] for _, v in machines.items()]) >= runtime_config['server']['num_clients']

    remain_clients = runtime_config['server']['num_clients']
    counter = 0
    for m_k in machines:
        dc = {'services': {}, 'version': '2'}
        for i in range(min(remain_clients, machines[m_k]['capacity'])):
            client_id = counter + i
            tmp = client_template.copy()
            tmp['container_name'] = 'client%s' % client_id
            tmp['command'] = 'sh -c ' \
                             '"export CLIENT_ID={} ' \
                             '&& tc qdisc add dev eth0 root tbf rate {} latency 10ms burst 60000kb ' \
                             '&& python3 -m FedEval.run -f run -r client -c {}"'.format(
                client_id, runtime_config['clients']['bandwidth'], path)
            dc['services']['client_%s' % client_id] = tmp

        counter += min(remain_clients, machines[m_k]['capacity'])
        remain_clients -= counter

        with open("docker-compose-%s.yml" % m_k, 'w') as f:
            no_alias_dumper = yaml.dumper.SafeDumper
            no_alias_dumper.ignore_aliases = lambda self, data: True
            yaml.dump(dc, f, default_flow_style=False, Dumper=no_alias_dumper)


def generate_docker_compose_local(runtime_config, path):

    project_path = os.path.abspath('./')

    server_template = {
        'image': runtime_config['docker']['image'],
        'ports': ['{0}:{0}'.format(runtime_config['server']['port'])],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'command': 'sh -c "python3 -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server',
        'networks': ['server-clients']
    }

    client_template = {
        'image': runtime_config['docker']['image'],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'networks': ['server-clients']
    }

    dc = {
        'version': "2",
        'networks': {'server-clients': {'driver': 'bridge'}},
        'services': {'server': server_template}
    }

    counter = 0
    for client_id in range(counter, counter + runtime_config['server']['num_clients']):
        tmp = client_template.copy()
        tmp['container_name'] = 'client%s' % client_id
        tmp['command'] = 'sh -c ' \
                         '"export CLIENT_ID={0} ' \
                         '&& tc qdisc add dev eth0 root tbf rate {1} latency 50ms burst 15kb ' \
                         '&& python3 -m FedEval.run -f run -r client -c {2}"'.format(
            client_id,
            runtime_config['clients']['bandwidth'],
            path)
        dc['services']['client_%s' % client_id] = tmp

    with open("docker-compose.yml", 'w') as f:
        no_alias_dumper = yaml.dumper.SafeDumper
        no_alias_dumper.ignore_aliases = lambda self, data: True
        yaml.dump(dc, f, default_flow_style=False, Dumper=no_alias_dumper)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--function', '-f', choices=('run', 'data', 'compose-local', 'compose-server'),
                             help='Run the server/clients, or generate the data')
    args_parser.add_argument('--role', '-r', choices=('server', 'client'),
                             help='Role of current party, should be server or client')
    args_parser.add_argument('--config', '-c', default='./',
                             help='The path to the config files, defaults to be ./')

    args = args_parser.parse_args()

    # 1 Load the configs
    with open(os.path.join(args.config, '1_data_config.yml'), 'r') as f:
        data_config = yaml.load(f)

    with open(os.path.join(args.config, '2_model_config.yml'), 'r') as f:
        model_config = yaml.load(f)

    with open(os.path.join(args.config, '3_runtime_config.yml'), 'r') as f:
        runtime_config = yaml.load(f)

    if args.function == 'data':

        generate_data(data_config=data_config, model_config=model_config, runtime_config=runtime_config)

    if args.function == 'run':

        run(role=args.role, data_config=data_config, model_config=model_config, runtime_config=runtime_config)

    if args.function == 'compose-local':

        generate_docker_compose_local(runtime_config, args.config)

    if args.function == 'compose-server':

        generate_docker_compose_server(runtime_config, args.config)

