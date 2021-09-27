import os
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from flask import Flask, request
from flask_socketio import SocketIO as ServerSocketIO
from flask_socketio import emit
from socketIO_client import SocketIO as ClientSocketIO

from ..config import ConfigurationManager, ServerFlaskInterface
from ..config.role import ClientId
from ..role.container import ClientNodeContextManager, NodeId
from .communicator import Communicatior
from .events import ClientSocketIOEvent, ServerSocketIOEvent, SocketIOEvent
from .model_weights_io import ModelWeightsFlaskHandler, ModelWeightsIoInterface

Sid = Any           # from SocketIO


def _event2message(event: SocketIOEvent) -> str:
    return event.value


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

    def on(self, event: ClientSocketIOEvent, *on_args, **on_kwargs):
        def decorator(func: Callable):
            self._sio.on(_event2message(event), func, *on_args, **on_kwargs)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return None
            return wrapper
        return decorator

    def invoke(self, event: ServerSocketIOEvent, *args, **kwargs):
        return self._sio.emit(_event2message(event), *args, **kwargs)

    def wait(self, seconds=None, **kw) -> None:
        self._sio.wait(seconds=seconds, **kw)

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
        self._socketio = ServerSocketIO(self._app, max_http_buffer_size=1e20,
                                        async_handlers=True, ping_timeout=3600,
                                        ping_interval=1800, cors_allowed_origins='*')

        # server-side handler register
        self.on = ServerFlaskCommunicator._son(self._socketio.on)
        self.route = ServerFlaskCommunicator._route(self._app.route)
        self._client_node_ctx_mgr = ClientNodeContextManager()

    @classmethod
    def _son(cls, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            event: ServerSocketIOEvent = args[0]
            return func(_event2message(event), *args[1:], **kwargs)
        return wrapper

    def invoke(self, event: ClientSocketIOEvent, *args, callee: Optional[ClientId]=None, **kwargs):
        room_id = request.sid
        if callee is not None:
            with self._client_node_ctx_mgr.get_by_client(callee) as c_node_ctx:
                room_id = c_node_ctx.comm_id
        return emit(_event2message(event), *args, room=room_id, **kwargs)

    def invoke_all(self,
                   event: ClientSocketIOEvent,
                   payload: Optional[Dict[str, Any]] = None,
                   *args,
                   callees: Optional[Iterable[ClientId]] = None,
                   **kwargs):
        msg = _event2message(event)
        if callees is None:
            if payload is None:
                return emit(msg, *args, **kwargs, broadcast=True)
            else:
                return emit(msg, payload, *args, **kwargs, broadcase=True)
        else:
            res = list()
            for node_id, client_ids in self._client_node_ctx_mgr.cluster_by_node(callees).items():
                if payload is None:
                    payload = dict()
                payload['selected_clients'] = list(client_ids)
                with self._client_node_ctx_mgr.get_by_node(node_id) as c_node_ctx:
                    room_id = c_node_ctx.comm_id
                    res.append(emit(msg, payload, *args, room=room_id, **kwargs))
            return res

    def run_server(self) -> None:
        self._socketio.run(self._app, host=self._host, port=self._port)

    @classmethod
    def _route(cls, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def handle_disconnection(self) -> Iterable[ClientId]:
        return self._client_node_ctx_mgr.deactivate_by_comm(request.sid)

    def handle_reconnection(self) -> Iterable[ClientId]:
        with self._client_node_ctx_mgr.get_by_comm(request.sid) as ctx:
            self._client_node_ctx_mgr.recover_from_deactivation(ctx.id)
            recovered_client_ids = ctx.client_ids
        return recovered_client_ids

    def activate(self, node_id: NodeId, client_ids: Iterable[ClientId]) -> None:
        self._client_node_ctx_mgr.activate(node_id, request.sid, client_ids)

    @property
    def ready_client_ids(self) -> Iterable[ClientId]:
        return self._client_node_ctx_mgr.online_client_ids
