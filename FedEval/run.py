import argparse
import copy
import os
from math import ceil

import yaml

from .config import (DEFAULT_D_CFG_FILENAME, DEFAULT_MDL_CFG_FILENAME,
                     DEFAULT_RT_CFG_FILENAME, ConfigurationManager)
from .dataset import *
from .model import *
from .role import Client, Server


def generate_data(save_file=True):
    # TODO(fgh) move this function into dataset module
    d_cfg = ConfigurationManager().data_config
    try:
        data = eval(d_cfg.dataset_name)()
    except ModuleNotFoundError:
        print('Invalid dataset name', data_config['dataset'])
        return None

    if d_cfg.iid:
        print('Generating IID data')
        clients_data = data.iid_data(save_file=save_file)
    else:
        print('Generating Non-IID data')
        # TODO&Q (fgh) what does the "shared_data" stands for?
        clients_data = data.non_iid_data(
            shared_data=data_config['shared_data'], save_file=save_file)
    return clients_data


# def start_clients(params):
#     cid, c1, c2, c3 = params
#     Client(data_config=c1, model_config=c2, runtime_config=c3,)


def run(role: str):
    ConfigurationManager()  # init configs
    if role == 'client':
        Client()
        # from multiprocessing import Pool
        # n_jobs = 20
        # with Pool(processes=n_jobs) as pool:
        #     pool.map(start_clients, [[str(e), data_config, model_config, runtime_config] for e in range(n_jobs)])
        # p = Pool()
        # for i in range(1, 3):
        #     p.apply_async(start_clients, args=(data_config, model_config, runtime_config, str(i)))
        # print('Waiting for all subprocesses done...')
        # p.close()
        # p.join()

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
        'command': 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server'
    }

    client_template = {
        'image': rt_cfg.image_label,
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'environment': []
    }

    if rt_cfg.gpu_enabled:
        client_template['runtime'] = 'nvidia'

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
            tmp['command'] = 'sh -c ' \
                             '"export CONTAINER_ID={} ' \
                             '&& tc qdisc add dev eth0 root tbf rate {} latency 10ms burst 60000kb ' \
                             '&& python3 -W ignore -m FedEval.run -f run -r client -c {}"'.format(
                container_id, rt_cfg.client_bandwidth, path)
            nvidia_device_env = (container_id % rt_cfg.gpu_num) if rt_cfg.gpu_enabled else -1
            tmp['environment'].append(f'NVIDIA_VISIBLE_DEVICES={nvidia_device_env}')
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
        'command': 'sh -c "python3 -W ignore -m FedEval.run -f run -r server -c {}"'.format(path),
        'container_name': 'server',
        'networks': ['server-clients']
    }

    client_template = {
        'image': rt_cfg.image_label,
        'volumes': ['%s:/FML' % project_path],
        'working_dir': '/FML',
        'cap_add': ['NET_ADMIN'],
        'networks': ['server-clients'],
        'environment': []
    }

    if rt_cfg.gpu_enabled:
        client_template['runtime'] = 'nvidia'

    dc = {
        'version': "2",
        'networks': {'server-clients': {'driver': 'bridge'}},
        'services': {'server': server_template}
    }

    for container_id in range(rt_cfg.container_num):
        tmp = copy.deepcopy(client_template)
        tmp['container_name'] = 'container%s' % container_id
        tmp['command'] = 'sh -c ' \
                         '"export CONTAINER_ID={0} ' \
                         '&& tc qdisc add dev eth0 root tbf rate {1} latency 50ms burst 15kb ' \
                         '&& python3 -W ignore -m FedEval.run -f run -r client -c {2}"'.format(
            container_id,
            rt_cfg.client_bandwidth,
            path)
        if rt_cfg.gpu_enabled:
            tmp['environment'].append('NVIDIA_VISIBLE_DEVICES=%s' % (container_id % rt_cfg.gpu_num))
        else:
            tmp['environment'].append('NVIDIA_VISIBLE_DEVICES=-1')
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

    # load configs
    with open(os.path.join(args.config, DEFAULT_D_CFG_FILENAME), 'r') as f:
        data_config = yaml.load(f)
    with open(os.path.join(args.config, DEFAULT_MDL_CFG_FILENAME), 'r') as f:
        model_config = yaml.load(f)
    with open(os.path.join(args.config, DEFAULT_RT_CFG_FILENAME), 'r') as f:
        runtime_config = yaml.load(f)
    # init configurations
    ConfigurationManager(data_config, model_config, runtime_config)

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
