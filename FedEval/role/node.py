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

    def __init__(self, name: str) -> None:
        self.name = name

    def _init_logger(self, logger_name: str, log_dir_name: str):
        # TODO(fgh): move log-related into a standalone module
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        rt_cfg = ConfigurationManager().runtime_config
        self.log_dir = os.path.join(
            rt_cfg.log_dir_path, log_dir_name, time_str)
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
