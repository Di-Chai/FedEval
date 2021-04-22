import os
import yaml
import pickle
import argparse
import numpy as np

from .model import *
from .dataset import *
from .role import Client, Server


def generate_data(data_config, model_config, runtime_config, save_file=True):
    try:
        data = eval(data_config['dataset'])(
            output_dir=data_config['data_dir'],  # for saving
            flatten=True if model_config['MLModel']['name'] == 'MLP' else False,
            normalize=data_config['normalize'],
            train_val_test=data_config['train_val_test'],
            num_clients=runtime_config['server']['num_clients']
        )
    except ModuleNotFoundError:
        print('Invalid dataset name', data_config['dataset'])
        return None

    if not data_config['non-iid']:
        print('Generating IID data')
        clients_data = data.iid_data(sample_size=data_config['sample_size'], save_file=save_file)
    else:
        print('Generating Non-IID data')
        clients_data = data.non_iid_data(
            non_iid_class=data_config['non-iid-class'],
            strategy=data_config['non-iid-strategy'],
            shared_data=data_config['shared_data'], sample_size=data_config['sample_size'],
            save_file=save_file
        )
    return clients_data


def run(role, data_config, model_config, runtime_config):
    if role == 'client':
        Client(data_config=data_config, model_config=model_config, runtime_config=runtime_config)

    if role == 'server':
        # 3 Init the server
        server = Server(data_config=data_config, model_config=model_config, runtime_config=runtime_config)
        server.start()


def generate_docker_compose_server(runtime_config, path):
    project_path = os.path.abspath('./')

    server_template = {
        'image': runtime_config['docker']['image'],
        'ports': ['{0}:{0}'.format(runtime_config['server']['port'])],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'command': 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server'
    }

    client_template = {
        'image': runtime_config['docker']['image'],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        # 'runtime': 'nvidia',
        # 'environment': ['NVIDIA_VISIBLE_DEVICES=all']
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
                             '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                client_id, runtime_config['clients']['bandwidth'], path)
            dc['services']['client_%s' % client_id] = tmp

        counter += min(remain_clients, machines[m_k]['capacity'])
        remain_clients -= min(remain_clients, machines[m_k]['capacity'])

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
        'command': 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server',
        'networks': ['server-clients']
    }

    client_template = {
        'image': runtime_config['docker']['image'],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'networks': ['server-clients'],
        # 'runtime': 'nvidia',
        # 'environment': ['NVIDIA_VISIBLE_DEVICES=all']
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
                         '&& python3 -W ignore -m FedEval.run -f run -r client -c {2}"'.format(
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
