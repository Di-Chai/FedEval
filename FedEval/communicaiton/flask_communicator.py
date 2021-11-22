import os
import logging
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from flask import Flask, request
from flask_socketio import SocketIO as ServerSocketIO
from flask_socketio import emit
from socketIO_client import SocketIO as ClientSocketIO

from ..config import ClientId, ConfigurationManager
from ..role.container import NodeId
from .communicator import ClientCommunicator, ServerCommunicator
from .events import ClientEvent, ServerEvent, event2message

Sid = Any           # from SocketIO


class ClientFlaskCommunicator(ClientCommunicator):
    def __init__(self) -> None:
        super().__init__()
        self._sio = ClientSocketIO(self._host, self._port)

    def on(self, event: ClientEvent, *on_args, **on_kwargs):
        def decorator(func: Callable):
            self._sio.on(event2message(event), func, *on_args, **on_kwargs)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return None
            return wrapper
        return decorator

    def invoke(self, event: ServerEvent, *args, **kwargs):
        return self._sio.emit(event2message(event), *args, **kwargs)

    def wait(self, **kw) -> None:
        self._sio.wait(**kw)


class ServerFlaskCommunicator(ServerCommunicator):
    def __init__(self) -> None:
        super().__init__()
        static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'role')
        self._app = Flask(__name__, template_folder=os.path.join(static_path, 'templates'),
                          static_folder=os.path.join(static_path, 'static'))
        self._app.config['SECRET_KEY'] = ConfigurationManager().runtime_config.secret_key
        log = logging.getLogger('werkzeug')
        log.setLevel(eval(f'logging.{ConfigurationManager().runtime_config.console_log_level}'))
        self._socketio = ServerSocketIO(self._app, max_http_buffer_size=1e20,
                                        async_handlers=True, ping_timeout=3600,
                                        ping_interval=1800, cors_allowed_origins='*')

    def on(self, event: ClientEvent) -> Callable[[Callable], Any]:
        return self._socketio.on(event2message(event))

    def route(self, rule: str, **options: Any):
        return self._app.route(rule, **options)

    def invoke(self, event: ClientEvent, *args, callee: Optional[ClientId]=None, **kwargs):
        room_id = request.sid
        if callee is not None:
            with self._client_node_ctx_mgr.get_by_client(callee) as c_node_ctx:
                room_id = c_node_ctx.comm_id
        return emit(event2message(event), *args, room=room_id, **kwargs)

    def invoke_all(self,
                   event: ClientEvent,
                   payload: Optional[Dict[str, Any]] = None,
                   *args,
                   callees: Optional[Iterable[ClientId]] = None,
                   **kwargs):
        msg = event2message(event)
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

    def handle_disconnection(self) -> Iterable[ClientId]:
        return self._handle_disconnection(request.sid)

    def handle_reconnection(self) -> Iterable[ClientId]:
        return self._handle_reconnection(request.sid)

    def activate(self, node_id: NodeId, client_ids: Iterable[ClientId]) -> None:
        self._activate(node_id, request.sid, client_ids)
