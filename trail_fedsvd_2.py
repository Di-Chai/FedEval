import os
import argparse
import socket

from FedEval.run_util import _load_config, _save_config


config_dir = os.path.join('configs', 'FedSVD')

c1, c2, c3 = _load_config(config_dir)

c1['dataset'] = 'vertical_linear_regression_memmap'
c1['feature_size'] = 500
c1['sample_size'] = 10000

c2['FedModel']['name'] = 'FedSVD'
c2['FedModel']['block_size'] = 1000
c2['FedModel']['fedsvd_mode'] = 'lr'
c2['FedModel']['fedsvd_top_k'] = -1
c2['FedModel']['fedsvd_lr_l2'] = 0

c3['server']['num_clients'] = 2
c3['docker']['num_containers'] = 2


# for latency in ['0ms', '5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms']:
#     c3['communication']['limit_network_resource'] = True
#     c3['communication']['bandwidth_upload'] = '10000Mbit'
#     c3['communication']['bandwidth_download'] = '10000Mbit'
#     c3['communication']['latency'] = latency
#     _save_config(c1, c2, c3, config_dir)
#     os.system(f'sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -m FedEval.run_util -m local -c {config_dir} -e run')

# for bandwidth in ['10Mbit', '100Mbit', '1000Mbit', '10000Mbit'][::-1]:
#     c3['communication']['limit_network_resource'] = True
#     c3['communication']['bandwidth_upload'] = bandwidth
#     c3['communication']['bandwidth_download'] = bandwidth
#     c3['communication']['latency'] = '0ms'
#     _save_config(c1, c2, c3, config_dir)
#     os.system(f'sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -m FedEval.run_util -m local -c {config_dir} -e run')

# for sample_size in [
#     1000000, 2000000, 3000000, 4000000, 5000000, 6000000
# ]:
#     c1['sample_size'] = sample_size
#     _save_config(c1, c2, c3, config_dir)
#     os.system('sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -m FedEval.run_util -m local -c configs/FedSVD -e run')

# c3['communication']['limit_network_resource'] = True
# c3['communication']['latency'] = '25ms'
# c3['communication']['bandwidth_upload'] = '1024Mbit'
# c3['communication']['bandwidth_download'] = '1024Mbit'
# for sample_size in [
#     1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000,
#     9000000, 10000000, 20000000, 30000000, 40000000, 50000000
# ]:
#     c1['sample_size'] = sample_size
#     _save_config(c1, c2, c3, config_dir)
#     os.system('sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -m FedEval.run_util -m local -c configs/FedSVD -e run')


# Precision
c1['feature_size'] = 100000
c1['sample_size'] = 100000

c2['FedModel']['name'] = 'FedSVD'
c2['FedModel']['block_size'] = 1000
c2['FedModel']['fedsvd_mode'] = 'svd'
c2['FedModel']['fedsvd_top_k'] = -1
c2['FedModel']['fedsvd_lr_l2'] = 0
c3['communication']['limit_network_resource'] = False

c3['server']['num_clients'] = 2
c3['docker']['num_containers'] = 2
c3['log']['log_dir'] = 'log/precision'

for dataset in ['wine', 'mnist_matrix']:
    c1['dataset'] = dataset
    _save_config(c1, c2, c3, config_dir)
    os.system(f'python -m FedEval.run_util -m local -c {config_dir} -e run')
