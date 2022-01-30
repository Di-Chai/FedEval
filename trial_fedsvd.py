import os
import enum
import argparse
import socket

from FedEval.run_util import _load_config, _save_config


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--mode', '-m', choice=('svd', 'lr', 'pca'))
args_parser.add_argument('--task', '-t', choices=('latency', 'bandwidth', 'large_scale', 'precision', 'large_scale_recsys'))
args = args_parser.parse_args()

svd_mode = args.mode
task = args.task

# python = "sudo /home/ubuntu/.virtualenvs/chaidi/bin/python"
python = "python"

# Base Config files
config_dir = os.path.join('configs', 'FedSVD')
c1, c2, c3 = _load_config(config_dir)

c1['random_seed'] = 100
c3['log']['log_dir'] = f'log/{task}'

c2['FedModel']['name'] = 'FedSVD'
c2['FedModel']['block_size'] = 1000
c2['FedModel']['fedsvd_mode'] = svd_mode
c2['FedModel']['fedsvd_top_k'] = 10 if svd_mode == 'pca' else -1  # By default, we compute the top 10 PC for PCA tasks
c2['FedModel']['fedsvd_lr_l2'] = 0


if task == 'latency':
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c1['sample_size'] = 10000
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = True
    c3['communication']['bandwidth_upload'] = '10000Mbit'
    c3['communication']['bandwidth_download'] = '10000Mbit'
    
    for latency in ['0ms', '5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms']:
        c3['communication']['latency'] = latency
        _save_config(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'bandwidth':
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c1['sample_size'] = 10000
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = True
    # No additional latency
    c3['communication']['latency'] = '0ms'

    for bandwidth in ['10Mbit', '100Mbit', '1000Mbit', '10000Mbit']:
        c3['communication']['bandwidth_upload'] = bandwidth
        c3['communication']['bandwidth_download'] = bandwidth
        _save_config(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'large_scale':
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = True
    c3['communication']['latency'] = '25ms'  # RTT=50ms
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    for sample_size in [
        1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000,
        9000000, 10000000, 20000000, 30000000, 40000000, 50000000
    ]:
        c1['sample_size'] = sample_size
        _save_config(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'precision':
    # If you are not using synthetic data, then the following feature_size and 
    #   sample_size are just upper bound of the data.  So we set to a very large 
    #   number here, and the actual data size is determined by the real-world data.
    c1['feature_size'] = 100000
    c1['sample_size'] = 100000
    c3['communication']['limit_network_resource'] = False
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2

    for dataset in ['wine', 'mnist_matrix', 'ml100k_lr', 'synthetic_matrix_horizontal']:
        if 'synthetic' in dataset:
            c1['feature_size'] = 1000
            c1['sample_size'] = 50000
        c1['dataset'] = dataset
        _save_config(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'large_scale_recsys':
    # If you are not using synthetic data, then the following feature_size and 
    #   sample_size are just upper bound of the data.  So we set to a very large 
    #   number here, and the actual data size is determined by the real-world data.
    c1['feature_size'] = 10000000
    c1['sample_size'] = 10000000
    # We only test the large_scale_recsys on svd task
    c2['FedModel']['name'] = 'FedSVD'
    c2['FedModel']['block_size'] = 1000
    c2['FedModel']['fedsvd_mode'] = 'svd'
    c2['FedModel']['fedsvd_top_k'] = -1
    c2['FedModel']['fedsvd_lr_l2'] = 0
    # same network setting as large_scale
    c3['communication']['limit_network_resource'] = True
    c3['communication']['latency'] = '25ms'
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    # Set the dataset
    c1['dataset'] = 'ml25m_matrix_memmap'

    _save_config(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
