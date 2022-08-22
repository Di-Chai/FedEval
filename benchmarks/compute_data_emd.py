import os
import sys

import hickle

father_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(father_dir)
os.chdir(father_dir)

import copy

from FedEval.config import ConfigurationManager
from FedEval.run import generate_data
from FedEval.utils import get_emd
from multiprocessing import Process, Queue


def print_error(value):
    print("error: ", value)


def generate_data_single_thread(configs, queue):
    data_cfg, model_cfg, runtime_cfg = configs
    cfg_manager = ConfigurationManager(data_cfg, model_cfg, runtime_cfg)
    generate_data(True)
    queue.put([cfg_manager.runtime_config.client_num, cfg_manager.data_dir_name])


def generate_data_and_compute_emd(configs):
    data_queue = Queue()
    p = Process(target=generate_data_single_thread, args=(configs, data_queue, ))
    p.start()
    p.join()

    client_num, client_data_path = data_queue.get()
    client_data = []
    for cid in range(client_num):
        with open(os.path.join(client_data_path, f'client_{cid}.pkl'), 'r') as f:
            client_data.append(hickle.load(f))
    
    results = [
        configs[0]['dataset'], configs[0]['non-iid'], configs[0]['random_seed']
    ]
    print('Running', results)
    emd = get_emd(client_data)
    results.append(emd)
    print('Finished', results)
    return results


if __name__ == '__main__':

    c1, c2, c3 = ConfigurationManager.load_configs('configs/debug')

    datasets = ['mnist', 'femnist', 'celeba', 'sentiment140', 'shakespeare']

    target_configs = []

    for dataset in datasets:
        for non_iid in [True, False]:
            for random_seed in [0]:

                data_config = copy.deepcopy(c1)
                model_config = copy.deepcopy(c2)
                runtime_config = copy.deepcopy(c3)

                data_config['dataset'] = dataset
                data_config['random_seed'] = random_seed

                if dataset == 'mnist':
                    data_config['sample_size'] = 700
                    runtime_config['server']['num_clients'] = 100

                if dataset == 'femnist':
                    data_config['sample_size'] = None
                    runtime_config['server']['num_clients'] = 1000

                if dataset == 'celeba':
                    data_config['sample_size'] = None
                    runtime_config['server']['num_clients'] = 1000

                if dataset == 'sentiment140':
                    data_config['sample_size'] = None
                    runtime_config['server']['num_clients'] = 1000

                if dataset == 'shakespeare':
                    runtime_config['server']['num_clients'] = 100
                    data_config['normalize'] = False

                data_config['non-iid'] = non_iid
                if non_iid:
                    if dataset == 'mnist':
                        data_config['non-iid-strategy'] = 'average'
                        data_config['non-iid-class'] = 1
                    else:
                        data_config['non-iid-strategy'] = 'natural'

                target_configs.append([data_config, model_config, runtime_config])

    file_name = 'emd.csv'
    if os.path.isfile(file_name):
        with open('emd.csv', 'r') as f:
            records = f.readlines()
            records = set(['_'.join(e.split(', ')[:3]) for e in records])
    else:
        records = []

    final_results = []
    for tc in target_configs:
        trial_id = f"{tc[0]['dataset']}_{str(tc[0]['non-iid'])}_{str(tc[0]['random_seed'])}"
        if trial_id in records:
            print('Finished', trial_id)
            continue
        final_results.append(generate_data_and_compute_emd(tc))
        with open('emd.csv', 'a') as f:
            for fr in final_results[-1:]:
                f.write(', '.join([str(e) for e in fr]) + '\n')
