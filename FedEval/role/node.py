import logging
import os
import time
from abc import ABCMeta

from ..config import ConfigurationManager


class Node(metaclass=ABCMeta):
    """the basic of a node in federated learning network.
    This class should be inherited instead of directly instantiate. 

    Attributes:
        name (str): the name of this node instance.
        fed_model (FedStrategyInterface): federated strategy instance
            constructed according to the given configurations.
    """

    def _init_logger(self, logger_name: str, log_dir_name: str):
        # TODO(fgh): move log-related into a standalone module
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        self.log_dir = os.path.join(
            ConfigurationManager().log_dir_path, log_dir_name, time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, 'train.log')
        fh = logging.FileHandler(log_file, encoding='utf8')
        fh.setLevel(logging.INFO)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        # self.fed_model.set_logger(self.logger)

    @staticmethod
    def _config_gpu(container_id=None):
        import tensorflow as tf
        cfg_mgr = ConfigurationManager()
        if cfg_mgr.runtime_config.gpu_enabled:
            # Please set CUDA_VISIBLE_DEVICES if not using docker
            CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            if len(CUDA_VISIBLE_DEVICES) > 1:
                if container_id is not None:
                    selected_gpu = int(container_id) % len(CUDA_VISIBLE_DEVICES)
                    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES[selected_gpu]
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES[0]
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_devices = tf.config.list_logical_devices('GPU')
                    print(
                        cfg_mgr.role, len(gpus), "Physical GPUs,", len(logical_devices), "Logical GPUs"
                    )
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)  # TODO(fgh) expose this exception
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = -1
