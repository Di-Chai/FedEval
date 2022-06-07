import argparse
import copy
import os
import multiprocessing
from math import ceil

import yaml

from .config import (DEFAULT_D_CFG_FILENAME_YAML, DEFAULT_MDL_CFG_FILENAME_YAML,
                     DEFAULT_RT_CFG_FILENAME_YAML, ConfigurationManager)
from .dataset import *
from .model import *
from .role import Client, Server


UNIFIED_JOB_TIME = os.getenv('UNIFIED_JOB_TIME')


def generate_data(save_file=True):
    # TODO(fgh) move this function into dataset module
    d_cfg = ConfigurationManager().data_config
    try:
        data = eval(d_cfg.dataset_name)()
    except ModuleNotFoundError:
        print('Invalid dataset name', d_cfg.dataset_name)
        return None

    if save_file and not data.need_regenerate:
        return None

    if d_cfg.iid:
        print('Generating IID data')
        clients_data = data.iid_data(save_file=save_file)
    else:
        print('Generating Non-IID data')
        # TODO&Q (fgh) what does the "shared_data" stands for?
        # Di: In some FL mechanisms, the clients jointly create a shared datasets to do something, e.g., solving
        #     the non-iid issue, or the server may need a centralized and trustful datasets to detect model poisoning.
        # Di: temporally remove the shared data, could be added in the future if it's still needed
        # clients_data = data.non_iid_data(
        #     shared_data=data_config['shared_data'], save_file=save_file)
        clients_data = data.non_iid_data(save_file=save_file)
    return clients_data


def run(role: str, config_path=None, unified_job_time=None, container_id=None):
    if config_path:
        # init configs (when running without docker)
        ConfigurationManager.from_files(config_path)
    if unified_job_time:
        os.environ['UNIFIED_JOB_TIME'] = unified_job_time
    if role == 'client':
        if container_id:
            os.environ['CONTAINER_ID'] = container_id
        Client()
    elif role == 'server':
        server = Server()
        server.start()
    else:
        raise NotImplementedError


def generate_docker_compose_server(path):
    project_path = os.path.abspath('./')
    rt_cfg = ConfigurationManager().runtime_config

    server_template = {
        'image': rt_cfg.image_label,
        'ports': ['{0}:{0}'.format(rt_cfg.central_server_port)],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'command': None,
        'container_name': 'server',
        'environment': [f'machine_id={rt_cfg.server_machine.addr}:{rt_cfg.server_machine.port}']
    }

    if rt_cfg.limit_network_resource:
        server_template['command'] = 'sh -c "' \
                                     'tc qdisc add dev eth0 handle 1: root htb default 11' \
                                     '&& tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbps' \
                                     '&& tc class add dev eth0 parent 1:1 classid 1:11 htb rate {}' \
                                     '&& tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {}' \
                                     '&& python3 -W ignore -m FedEval.run -f run -r server -c {}' \
                                     '"'.format(rt_cfg.bandwidth_download, rt_cfg.latency, path)
    else:
        server_template['command'] = 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path)

    client_template = {
        'image': rt_cfg.image_label,
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'environment': [],
        'deploy': {'resources': {'limits': {'cpus': max(1, int(multiprocessing.cpu_count() / rt_cfg.container_num))}}}
    }

    if UNIFIED_JOB_TIME is not None:
        server_template['environment'].append(f"UNIFIED_JOB_TIME={UNIFIED_JOB_TIME}")
        client_template['environment'].append(f"UNIFIED_JOB_TIME={UNIFIED_JOB_TIME}")

    if rt_cfg.gpu_enabled:
        client_template['runtime'] = 'nvidia'
        server_template['runtime'] = 'nvidia'
        server_template['environment'].append('NVIDIA_VISIBLE_DEVICES=0')

    with open('docker-compose-server.yml', 'w') as f:
        no_alias_dumper = yaml.dumper.SafeDumper
        no_alias_dumper.ignore_aliases = lambda self, data: True
        yaml.dump({
            'version': "2",
            'services': {'server': server_template}
        }, f, default_flow_style=False, Dumper=no_alias_dumper)

    client_machines = rt_cfg.client_machines
    # Distribute the containers to different machines
    machine_capacity_sum = sum([m.capacity for m in client_machines.values()])
    assert machine_capacity_sum >= rt_cfg.container_num

    remain_clients = rt_cfg.client_num
    counter = 0
    for m_name, client_machine in client_machines.items():
        dc = {'services': {}, 'version': '2'}
        num_container_cur_machine = ceil(client_machine.capacity / machine_capacity_sum * rt_cfg.container_num)
        num_container_cur_machine = min(
            remain_clients, num_container_cur_machine)
        for i in range(num_container_cur_machine):
            container_id = counter + i
            tmp = copy.deepcopy(client_template)
            tmp['container_name'] = 'container%s' % container_id
            if rt_cfg.limit_network_resource:
                tmp['command'] = 'sh -c ' \
                                 '"export CONTAINER_ID={} ' \
                                 '&& tc qdisc add dev eth0 handle 1: root htb default 11 ' \
                                 '&& tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbps ' \
                                 '&& tc class add dev eth0 parent 1:1 classid 1:11 htb rate {} ' \
                                 '&& tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {} ' \
                                 '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                    container_id, rt_cfg.bandwidth_upload, rt_cfg.latency, path)
            else:
                tmp['command'] = 'sh -c ' \
                                 '"export CONTAINER_ID={} ' \
                                 '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                    container_id, path)
            nvidia_device_env = (container_id % rt_cfg.gpu_num) if rt_cfg.gpu_enabled else -1
            tmp['environment'].append(f'NVIDIA_VISIBLE_DEVICES={nvidia_device_env}')
            tmp['environment'].append(f'machine_id={client_machine.addr}:{client_machine.port}')
            dc['services'][f'container_{container_id}'] = tmp

        counter += num_container_cur_machine
        remain_clients -= num_container_cur_machine

        with open(f"docker-compose-{m_name}.yml", 'w') as f:
            no_alias_dumper = yaml.dumper.SafeDumper
            no_alias_dumper.ignore_aliases = lambda self, data: True
            yaml.dump(dc, f, default_flow_style=False, Dumper=no_alias_dumper)


def generate_docker_compose_local(path):
    project_path = os.path.abspath('./')
    rt_cfg = ConfigurationManager().runtime_config

    server_template = {
        'image': rt_cfg.image_label,
        'ports': ['{0}:{0}'.format(rt_cfg.central_server_port)],
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'command': None,
        'container_name': 'server',
        'networks': ['server-clients'],
        'environment': []
    }

    if rt_cfg.limit_network_resource:
        server_template['command'] = 'sh -c "' \
                                     'tc qdisc add dev eth0 handle 1: root htb default 11' \
                                     '&& tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbps' \
                                     '&& tc class add dev eth0 parent 1:1 classid 1:11 htb rate {}' \
                                     '&& tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {}' \
                                     '&& python3 -W ignore -m FedEval.run -f run -r server -c {}' \
                                     '"'.format(rt_cfg.bandwidth_download, rt_cfg.latency, path)
    else:
        server_template['command'] = 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path)

    client_template = {
        'image': rt_cfg.image_label,
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'networks': ['server-clients'],
        'environment': [],
        'deploy': {'resources': {'limits': {'cpus': max(1, int(multiprocessing.cpu_count() / rt_cfg.container_num))}}}
    }

    if UNIFIED_JOB_TIME is not None:
        server_template['environment'].append(f"UNIFIED_JOB_TIME={UNIFIED_JOB_TIME}")
        client_template['environment'].append(f"UNIFIED_JOB_TIME={UNIFIED_JOB_TIME}")

    if rt_cfg.gpu_enabled:
        client_template['runtime'] = 'nvidia'
        server_template['runtime'] = 'nvidia'
        server_template['environment'].append('NVIDIA_VISIBLE_DEVICES=0')

    dc = {
        'version': "2",
        'networks': {'server-clients': {'driver': 'bridge'}},
        'services': {'server': server_template}
    }

    for container_id in range(rt_cfg.container_num):
        tmp = copy.deepcopy(client_template)
        tmp['container_name'] = 'container%s' % container_id
        if rt_cfg.limit_network_resource:
            tmp['command'] = 'sh -c ' \
                             '"export CONTAINER_ID={} ' \
                             '&& tc qdisc add dev eth0 handle 1: root htb default 11 ' \
                             '&& tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbps ' \
                             '&& tc class add dev eth0 parent 1:1 classid 1:11 htb rate {} ' \
                             '&& tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {} ' \
                             '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                container_id, rt_cfg.bandwidth_upload, rt_cfg.latency, path)
        else:
            tmp['command'] = 'sh -c ' \
                             '"export CONTAINER_ID={} ' \
                             '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                container_id, path)
        if rt_cfg.gpu_enabled:
            tmp['environment'].append('NVIDIA_VISIBLE_DEVICES=%s' % (container_id % rt_cfg.gpu_num))
        else:
            tmp['environment'].append('NVIDIA_VISIBLE_DEVICES=-1')
        tmp['depends_on'] = ['server']
        dc['services']['container_%s' % container_id] = tmp

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

    # init configurations
    ConfigurationManager.from_files(args.config)

    if args.function == 'data':
        generate_data()
    elif args.function == 'run':
        run(args.role)
    elif args.function == 'compose-local':
        generate_docker_compose_local(args.config)
    elif args.function == 'compose-server':
        generate_docker_compose_server(args.config)
    else:
        print("unknown function(--function, -f) arg.")
