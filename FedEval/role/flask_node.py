import logging
import os
import time
from enum import Enum
from functools import wraps
from typing import Callable, Tuple, Any

import psutil
from flask import Flask
from flask_socketio import SocketIO, emit

from .model_weights_io import ModelWeightsFlaskHandler, ModelWeightsIoInterface
from .node import Node
from .role import Role
from .service_interface import ServerFlaskInterface

Sid = Any           # from SocketIO
ContainerId = int   # to identify container

class SocketIOEvent(Enum):
    """basic SocketIO life-cycle event names."""
    Connect = 'connect'
    Disconnect = 'disconnect'
    Reconnect = 'reconnect'


class ServerSocketIOEvent(SocketIOEvent, Enum):
    """server-side SocketIO event handles' name."""
    WakeUp = 'client_wake_up'
    Ready = 'client_ready'
    ResponseUpdate = 'client_update'
    ResponseEvaluate = 'client_evaluate'


class ClientSocketIOEvent(SocketIOEvent, Enum):
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

    def __init__(self, name: str, data_config, model_config, runtime_config, role: Role) -> None:
        super().__init__(name, data_config, model_config, runtime_config, role)

        if self.role == Role.Client:
            server_host, server_port = self.fed_model.param_parser.parse_server_addr(self.name)
            weights_download_url = f'http://{server_host}:{server_port}{ServerFlaskInterface.DownloadPattern}'
            self._model_weights_io_handler: ModelWeightsIoInterface = ModelWeightsFlaskHandler(
                weights_download_url)
            self._sio = SocketIO(server_host, server_port)

            self.on = FlaskNode._con(self._sio.on) # client-side handler register
            self.invoke = self._cinvoke
            self.wait = self._sio.wait
        elif self.role == Role.Server:
            self._host, self._port = self.fed_model.param_parser.parse_server_addr()
            current_path = os.path.dirname(os.path.abspath(__file__))
            self._app = Flask(__name__, template_folder=os.path.join(current_path, 'templates'),
                            static_folder=os.path.join(current_path, 'static'))
            self._app.config['SECRET_KEY'] = 'secret!' # TODO(fgh) move into configurations
            self._socketio = SocketIO(self._app, max_http_buffer_size=10 ** 20, async_handlers=True,
                                     ping_timeout=3600, ping_interval=1800, cors_allowed_origins='*')

            self.on = FlaskNode._son(self._socketio.on) # server-side handler register
            self.invoke = self._sinvoke
            self.route = self._app.route # server-side router
        else:
            raise NotImplementedError

    def _run_server(self) -> None:
        if self._role == Role.Server:
            self._socketio.run(self._app, host=self._host, port=self._port)
        else:
            raise TypeError("This is not a central server.")

    @staticmethod
    def __event2message(event: SocketIOEvent) -> str:
        return event.value

    @classmethod
    def _con(cls, func: Callable):
        @wraps(func)
        def wrapper(event: ClientSocketIOEvent, *args, **kwargs):
            return func(FlaskNode.__event2message(event), *args, **kwargs)
        return wrapper

    @classmethod
    def _son(cls, func: Callable):
        @wraps(func)
        def wrapper(event: ServerSocketIOEvent, *args, **kwargs):
            return func(FlaskNode.__event2message(event), *args, **kwargs)
        return wrapper

    @staticmethod
    def _cinvoke(event: ServerSocketIOEvent, *args, **kwargs):
        return emit(FlaskNode.__event2message(event), *args, **kwargs)

    @staticmethod
    def _sinvoke(event: ClientSocketIOEvent, *args, **kwargs):
        return emit(FlaskNode.__event2message(event), *args, **kwargs)

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
