import shutil
import sys
from trial import *
from FedEval.config import ConfigurationManager
from FedEval.run_util import recursive_update_dict


def run_once(d_cfg, m_cfg, r_cfg):
    c1, c2, c3 = ConfigurationManager.load_configs(config)
    c1 = recursive_update_dict(c1, d_cfg)
    c2 = recursive_update_dict(c2, m_cfg)
    c3 = recursive_update_dict(c3, r_cfg)
    unique_id = ConfigurationManager.generate_unique_id(data_config=c1, model_config=c2, runtime_config=c3)
    unique_config_path = config + f'_{unique_id}'
    ConfigurationManager.save_configs(c1, c2, c3, unique_config_path)
    os.system(f'{sys.executable} -W ignore -m FedEval.run_util -m without-docker -c {unique_config_path} -e run')
    shutil.rmtree(unique_config_path)


if __name__ == '__main__':

    params['runtime_config']['server']['host'] = "127.0.0.1"
    params['runtime_config']['server']['listen'] = "127.0.0.1"
    params['runtime_config']['server']['port'] = 8010
    params['runtime_config']['log']['console_log_level'] = 'ERROR'

    for seed in range(repeat):
        params['data_config']['random_seed'] = seed
        if args.tune is None:
            run_once(params['data_config'], params['model_config'], params['runtime_config'])
        else:
            print('Tuning', args.tune)
            if args.tune == 'lr':
                for lr in tune_params['lr']:
                    params['model_config']['MLModel']['optimizer']['lr'] = lr
                    run_once(params['data_config'], params['model_config'], params['runtime_config'])
            else:
                raise ValueError('Unknown tuning params', args.tune)
