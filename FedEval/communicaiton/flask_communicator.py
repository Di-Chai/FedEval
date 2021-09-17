import os
from functools import partial, wraps
from platform import platform
from typing import Any, Callable, Tuple

import psutil
from flask import Flask
from flask_socketio import SocketIO as ServerSocketIO
from flask_socketio import emit
from socketIO_client import SocketIO as ClientSocketIO

from ..config import ConfigurationManager, ServerFlaskInterface
from .communicator import Communicatior
from .events import ClientSocketIOEvent, ServerSocketIOEvent, SocketIOEvent
from .model_weights_io import ModelWeightsFlaskHandler, ModelWeightsIoInterface

Sid = Any           # from SocketIO


def __event2message(event: SocketIOEvent) -> str:
    return event.value


def _get_comm_in_and_out_linux() -> Tuple[int, int]:
    """retrieve network traffic counter in linux platforms (with the support from psutil lib).

    Returns:
        Tuple[int, int]: (the number of received bytes, the number of sent bytes)
    """
    eth0_info = psutil.net_io_counters(pernic=True).get('eth0')
    bytes_recv = 0 if eth0_info is None else eth0_info.bytes_recv
    bytes_sent = 0 if eth0_info is None else eth0_info.bytes_sent
    return bytes_recv, bytes_sent


class FlaskCommunicator(Communicatior):
    """an implementation of node.Node based on flask framework
    and SocketIO plugin in flask. This class should be inherited
    instead of instantiated.

    Attributes:
        log_dir (str): the local path of log directory. 
        logger: a logger with INFO-lv threshold in file side
            and ERROR-lv threshold in terminal side.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_comm_in_and_out() -> Tuple[int, int]:
        """retrieve network traffic counter.

        Raises:
            NotImplementedError: raised when called at unsupported
                platforms or unknown platforms.

        Returns:
            Tuple[int, int]: (the number of received bytes, the number of sent bytes)
        """
        platform_str = platform.system().lower()
        if platform_str == 'linux':
            return _get_comm_in_and_out_linux()
        elif platform_str == 'windows':
            raise NotImplementedError(
                f'Unsupported function at {platform_str} platform.')
        else:
            raise NotImplementedError("Unknown platform.")


class ClientFlaskCommunicator(FlaskCommunicator):
    def __init__(self) -> None:
        super().__init__()
        rt_cfg = ConfigurationManager().runtime_config
        self._host = rt_cfg.central_server_addr
        self._port = rt_cfg.central_server_port

        weights_download_url = f'http://{self._host}:{self._port}{ServerFlaskInterface.DownloadPattern.value}'
        self._model_weights_io_handler: ModelWeightsIoInterface = ModelWeightsFlaskHandler(
            weights_download_url)
        self._sio = ClientSocketIO(self._host, self._port)

        self.on = self._con  # client-side handler register
        self.invoke = partial(self._cinvoke, self)
        self.wait = self._sio.wait

    def _con(self, event: ClientSocketIOEvent, *on_args, **on_kwargs):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._sio.on(__event2message(event), func, *on_args, **on_kwargs)
            return wrapper
        return decorator

    @staticmethod
    def _cinvoke(self, event: ServerSocketIOEvent, *args, **kwargs):
        return self._sio.emit(__event2message(event), *args, **kwargs)


class ServerFlaskCommunicator(FlaskCommunicator):
    def __init__(self) -> None:
        super().__init__()
        rt_cfg = ConfigurationManager().runtime_config
        self._host = rt_cfg.central_server_listen_at
        self._port = rt_cfg.central_server_port

        current_path = os.path.dirname(os.path.abspath(__file__))
        self._app = Flask(__name__, template_folder=os.path.join(current_path, 'templates'),
                          static_folder=os.path.join(current_path, 'static'))
        self._app.config['SECRET_KEY'] = rt_cfg.secret_key
        self._socketio = ServerSocketIO(self._app, max_http_buffer_size=10 ** 20,
                                        async_handlers=True, ping_timeout=3600,
                                        ping_interval=1800, cors_allowed_origins='*')

        # server-side handler register
        self.on = FlaskCommunicator._son(self._socketio.on)
        self.invoke = self._sinvoke
        self.route = self._app.route  # server-side router

    @classmethod
    def _son(cls, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            event: ServerSocketIOEvent = args[0]
            return func(__event2message(event), *args[1:], **kwargs)
        return wrapper

    @staticmethod
    def _sinvoke(event: ClientSocketIOEvent, *args, **kwargs):
        return emit(__event2message(event), *args, **kwargs)

    def _run_server(self) -> None:
        self._socketio.run(self._app, host=self._host, port=self._port)
