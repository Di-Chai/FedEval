import logging
import os
import time
from abc import abstractmethod
from enum import Enum
import psutil
from typing import Tuple

from .node import Node
from .role import Role

from ..strategy import *


class SocketIOEvents(Enum):
    """basic SocketIO life-cycle event names."""
    Connect = 'connect'
    Disconnect = 'disconnect'
    Reconnect = 'reconnect'


class ServerSocketIOEvents(SocketIOEvents, Enum):
    """server-side SocketIO event handles' name."""
    WakeUp = 'client_wake_up'
    Ready = 'client_ready'
    ResponseUpdate = 'client_update'
    ResponseEvaluate = 'client_evaluate'


class ClientSocketIOEvents(SocketIOEvents, Enum):
    """client-side SocketIO event handles' name."""
    Init = 'init'
    RequestUpdate = 'request_update'
    RequestEvaluate = 'request_evaluate'
    Stop = 'stop'


class FlaskNode(Node):
    """an implementation of node.Node based on flask framework
    and SocketIO plugin in flask. This class should be inherited
    instead of instantiated.

    Attributes:
        log_dir (str): the local path of log directory. 
        logger: a logger with INFO-lv threshold in file side
            and ERROR-lv threshold in terminal side.
    """
    @abstractmethod
    def _init_flask_service(self):
        raise NotImplementedError

    def _init_logger(self, logger_name: str, log_dir_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        self.log_dir = os.path.join(self.runtime_config.get(
            'log_dir', 'log'), log_dir_name, time_str)
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
        self.fed_model.set_logger(self.logger)

    def __init__(self, name: str, data_config, model_config, runtime_config, role: Role) -> None:
        super().__init__(name, data_config, model_config, runtime_config, role)
        self._init_flask_service()

    @staticmethod
    def _get_comm_in_and_out() -> Tuple[int, int]:
        """retrieve network traffic counter.

        Returns:
            Tuple[int, int]: (the number of received bytes, the number of sent bytes)
        """
        return get_comm_in_and_out_linux()


def get_comm_in_and_out_linux() -> Tuple[int, int]:
    """retrieve network traffic counter in linux platforms (with the support from psutil lib).

    Returns:
        Tuple[int, int]: (the number of received bytes, the number of sent bytes)
    """
    eth0_info = psutil.net_io_counters(pernic=True).get('eth0')
    bytes_recv = 0 if eth0_info is None else eth0_info.bytes_recv
    bytes_sent = 0 if eth0_info is None else eth0_info.bytes_sent
    return bytes_recv, bytes_sent
