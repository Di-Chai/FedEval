import os
import pdb
import sys
import argparse

fed_eval_path = os.path.abspath('../../')
os.chdir(fed_eval_path)
sys.path.append(fed_eval_path)
from FedEval.config import ConfigurationManager

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--mode', '-m', choices=('svd', 'lr', 'pca'))
args_parser.add_argument('--task', '-t', choices=(
    'latency', 'bandwidth', 'vary_clients', 'large_scale', 'precision',
    'large_scale_recsys', 'small_scale', 'block_size', 'optimization', 'disk'
))
args = args_parser.parse_args()

svd_mode = args.mode
task = args.task

python = "python"

# Base Config files
config_dir = os.path.join('configs', 'FedSVD')
c1, c2, c3 = ConfigurationManager.load_configs(config_dir)

c1['random_seed'] = 100
c3['log']['log_dir'] = f'log/{task}_{svd_mode}'

c2['FedModel']['name'] = 'FedSVD'
c2['FedModel']['block_size'] = 1000
c2['FedModel']['fedsvd_mode'] = svd_mode
c2['FedModel']['fedsvd_top_k'] = 10 if svd_mode == 'pca' else -1  # By default, we compute the top 10 PC for PCA tasks
c2['FedModel']['fedsvd_lr_l2'] = 0
c2['FedModel']['fedsvd_opt_1'] = True
c2['FedModel']['fedsvd_opt_2'] = True
c2['FedModel']['fedsvd_debug_evaluate'] = False

if task == 'latency':
    if svd_mode == 'lr':
        c1['dataset'] = 'vertical_linear_regression'
        c1['feature_size'] = 500  # per client
        c1['sample_size'] = 10000
    else:
        c1['dataset'] = 'synthetic_matrix_horizontal'
        c1['feature_size'] = 100
        c1['sample_size'] = 1000  # per client
        # c1['feature_size'] = 1000
        # c1['sample_size'] = 5000  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = True
    c3['communication']['bandwidth_upload'] = '10000Mbit'
    c3['communication']['bandwidth_download'] = '10000Mbit'

    for latency in ['0ms', '5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms']:
        c3['communication']['latency'] = latency
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'bandwidth':
    if svd_mode == 'lr':
        c1['dataset'] = 'vertical_linear_regression'
        c1['feature_size'] = 500  # per client
        c1['sample_size'] = 10000
    else:
        c1['dataset'] = 'synthetic_matrix_horizontal'
        c1['feature_size'] = 100
        c1['sample_size'] = 1000  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = True
    # No additional latency
    c3['communication']['latency'] = '0ms'

    for bandwidth in ['10Mbit', '100Mbit', '1000Mbit', '10000Mbit']:
        c3['communication']['bandwidth_upload'] = bandwidth
        c3['communication']['bandwidth_download'] = bandwidth
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'vary_clients':
    assert svd_mode == 'svd'
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c1['feature_size'] = 1000
    c3['communication']['limit_network_resource'] = False
    for per_client_sample_size in [1000, 2000, 3000, 4000, 5000]:
        c1['sample_size'] = per_client_sample_size  # per client
        for num_clients in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            c3['server']['num_clients'] = num_clients
            c3['docker']['num_containers'] = num_clients
            ConfigurationManager.save_configs(c1, c2, c3, config_dir)
            os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'small_scale':
    # Comparing to PPDSVD
    assert svd_mode == 'svd'
    c1['feature_size'] = 1000
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c3['communication']['limit_network_resource'] = True
    c3['communication']['latency'] = '25ms'
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    for per_client_sample_size in [5000]:  # [5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 1000, 1500]:
        c1['sample_size'] = per_client_sample_size  # per client
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'large_scale':
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c3['communication']['limit_network_resource'] = False
    c3['communication']['latency'] = '25ms'  # RTT=50ms
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    for sample_size in [
        1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 20000000, 30000000,
        40000000, 50000000
    ]:
        c1['sample_size'] = sample_size
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'precision':
    # If you are not using synthetic data, then the following feature_size and 
    #   sample_size are just upper bound of the data.  So we set to a very large 
    #   number here, and the actual data size is determined by the real-world data.
    c1['feature_size'] = 100000
    c1['sample_size'] = 100000
    if svd_mode == 'pca':
        c2['FedModel']['fedsvd_top_k'] = 3
    c3['communication']['limit_network_resource'] = False
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    # Do Evaluation
    c2['FedModel']['fedsvd_debug_evaluate'] = True

    for dataset in ['wine', 'mnist_matrix', 'ml100k_lr', 'synthetic_matrix_horizontal']:
        c1['dataset'] = dataset
        if 'synthetic' in dataset:
            if svd_mode == 'lr':
                c1['feature_size'] = 500
                c1['sample_size'] = 10000
                c1['dataset'] = 'vertical_linear_regression'
            else:
                c1['feature_size'] = 1000
                c1['sample_size'] = 5000
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
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
    # By default, we run truncated svd on recsys data, and get the top 256 PCs
    c2['FedModel']['fedsvd_top_k'] = 256
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

    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'optimization':
    assert svd_mode == 'svd'
    c1['feature_size'] = 10000
    c1['sample_size'] = 25000
    c3['communication']['limit_network_resource'] = False
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    # No Opt
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c2['FedModel']['fedsvd_opt_1'] = False
    c2['FedModel']['fedsvd_opt_2'] = False
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    # Opt 1
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c2['FedModel']['fedsvd_opt_1'] = True
    c2['FedModel']['fedsvd_opt_2'] = False
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    # Opt 1,2
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c2['FedModel']['fedsvd_opt_1'] = True
    c2['FedModel']['fedsvd_opt_2'] = True
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    # Opt 1,2,3
    c1['dataset'] = 'synthetic_matrix_horizontal_memmap'
    c2['FedModel']['fedsvd_opt_1'] = True
    c2['FedModel']['fedsvd_opt_2'] = True
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

    c3['communication']['limit_network_resource'] = False
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2

    # Large-scale LR, Opt 1,2
    c1['dataset'] = 'vertical_linear_regression'
    c1['feature_size'] = 500  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c1['sample_size'] = 10000000
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    # Large-scale LR, Opt 1,2,3
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c1['sample_size'] = 10000000
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'block_size':
    assert svd_mode == 'svd'
    c1['dataset'] = 'synthetic_matrix_horizontal'
    c1['feature_size'] = 10000
    c1['sample_size'] = 50000
    c3['communication']['limit_network_resource'] = True
    c3['communication']['latency'] = '25ms'
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    for block_size in [10, 100, 1000, 10000]:
        c2['FedModel']['fedsvd_mode'] = 'svd'
        c2['FedModel']['fedsvd_top_k'] = -1
        c2['FedModel']['block_size'] = block_size  # per client
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    for block_size in [10, 100, 1000, 10000]:
        c2['FedModel']['fedsvd_mode'] = 'svd'
        c2['FedModel']['fedsvd_top_k'] = 100
        c2['FedModel']['block_size'] = block_size  # per client
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    for block_size in [10, 100, 1000, 10000]:
        c2['FedModel']['fedsvd_mode'] = 'pca'
        c2['FedModel']['fedsvd_top_k'] = 10
        c2['FedModel']['block_size'] = block_size  # per client
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
    c1['dataset'] = 'vertical_linear_regression'
    c1['feature_size'] = 5000
    c1['sample_size'] = 100000
    for block_size in [10, 100, 1000, 10000]:
        c2['FedModel']['fedsvd_mode'] = 'lr'
        c2['FedModel']['fedsvd_top_k'] = -1
        c2['FedModel']['block_size'] = block_size  # per client
        ConfigurationManager.save_configs(c1, c2, c3, config_dir)
        os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')

if task == 'disk':
    assert svd_mode == 'lr'
    c3['communication']['limit_network_resource'] = True
    c3['communication']['latency'] = '25ms'
    c3['communication']['bandwidth_upload'] = '1024Mbit'
    c3['communication']['bandwidth_download'] = '1024Mbit'
    c1['dataset'] = 'vertical_linear_regression_memmap'
    c1['feature_size'] = 500  # per client
    c3['server']['num_clients'] = 2
    c3['docker']['num_containers'] = 2
    c1['sample_size'] = 5000000
    ConfigurationManager.save_configs(c1, c2, c3, config_dir)
    os.system(f'{python} -m FedEval.run_util -m local -c {config_dir} -e run')
