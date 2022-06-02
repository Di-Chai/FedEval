from dataclasses import dataclass
import os
import copy
import yaml
import datetime

default_config = {
    'networks': {
        'secureml': {'driver': 'bridge', 'ipam': {'config': [{'subnet': '171.20.0.0/24'}]}}}, 
    'services': {
        'c1': {
            'cap_add': ['NET_ADMIN'], 
            'command': 'sh -c "tc qdisc add dev eth0 handle 1: root htb default 11 && tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbit && tc class add dev eth0 parent 1:1 classid 1:11 htb rate {} && tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {} && ./bin/secure_ML_synthetic 1 8001 {}"',
             'container_name': 'secureml_c1', 'image': 'mpc:v1', 'networks': {'secureml': {'ipv4_address': '171.20.0.2'}}, 'working_dir': '/root/Secure-ML/build/'
        },
        'c2': {
            'cap_add': ['NET_ADMIN'],
            'command': 'sh -c "tc qdisc add dev eth0 handle 1: root htb default 11 && tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbit && tc class add dev eth0 parent 1:1 classid 1:11 htb rate {} && tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {} && ./bin/secure_ML_synthetic 2 8001 {} 171.20.0.2"',
             'container_name': 'secureml_c2', 'image': 'mpc:v1', 'networks':{'secureml': {'ipv4_address': '171.20.0.3'}}, 'working_dir': '/root/Secure-ML/build/'
        }}, 
    'version': '2'
}


def run(i, b, d):
    print('#'*40)
    print(f'Running with iteration {i} bandwidth {b} delay {d}')
    tmp_config = copy.deepcopy(default_config)
    tmp_config['services']['c1']['command'] = tmp_config['services']['c1']['command'].format(b, d, i)
    tmp_config['services']['c2']['command'] = tmp_config['services']['c2']['command'].format(b, d, i)
    with open('docker-compose.yml', 'w') as f:
        yaml.dump(tmp_config, f)
    os.system('docker-compose up')
    curr_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.system(f'docker logs secureml_c1 > log/{curr_date}_{i*1000}_{b}_{d}_c1.log')
    os.system(f'docker logs secureml_c2 > log/{curr_date}_{i*1000}_{b}_{d}_c2.log')
    os.system('docker rm secureml_c1')
    os.system('docker rm secureml_c2')


# Fix the bandwidth, Change the latency
bandwidth = '10000Mbit'
iterations = 10  # num_samples = iterations * 1000
for delay in ['0ms', '5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms']:
    run(iterations, bandwidth, delay)

# Fix the latency, Change the bandwidth
delay = '0ms'
iterations = 10  # num_samples = iterations * 1000
for bandwidth in ['10Mbit', '100Mbit', '1000Mbit', '10000Mbit'][::-1]:
    run(iterations, bandwidth, delay)

# Fix the bandwidth and latency, Change the number of samples (i.e., iterations)
delay = '25ms'
bandwidth = '1000Mbit'
for iterations in [10, 100, 1000, 2000, 3000, 4000, 5000]:
    run(iterations, bandwidth, delay)
