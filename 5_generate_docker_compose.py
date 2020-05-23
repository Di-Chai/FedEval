import os
import argparse
import yaml

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--path', default='./')

args = args_parser.parse_args()

with open(os.path.join(args.path, '1_data_config.yml'), 'r') as f:
    data_config = yaml.load(f)

with open(os.path.join(args.path, '2_model_config.yml'), 'r') as f:
    model_config = yaml.load(f)

with open(os.path.join(args.path, '3_runtime_config.yml'), 'r') as f:
    runtime_config = yaml.load(f)

# TMP to make tf_wrapper & FedEval available
project_path = os.path.dirname(os.path.abspath(__file__))

server_template = {
    'image': runtime_config['docker']['image'],
    'ports': ['{0}:{0}'.format(runtime_config['server']['port'])],
    'volumes': ['%s:/FML' % project_path],
    'working_dir': '/FML',
    'cap_add': ['NET_ADMIN'],
    'command': 'sh -c "python3 server.py --path {}"'.format(args.path),
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
for client_id in range(counter, counter + runtime_config['clients']['num_clients']):
    tmp = client_template.copy()
    tmp['container_name'] = 'client%s' % client_id
    tmp['command'] = 'sh -c ' \
                     '"export CLIENT_ID={0} ' \
                     '&& tc qdisc add dev eth0 root tbf rate {1} latency 50ms burst 15kb ' \
                     '&& python3 client.py --path {2}"'.format(client_id,
                                                               runtime_config['clients']['bandwidth'],
                                                               args.path)
    dc['services']['client_%s' % client_id] = tmp

with open("docker-compose.yml", 'w') as f:
    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    yaml.dump(dc, f, default_flow_style=False, Dumper=noalias_dumper)