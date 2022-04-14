import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial, wraps
from json import dumps as dump_into_json_str
from json import loads as load_from_json_str
from logging import getLogger
from queue import Queue
from threading import Lock, Thread, local
from time import sleep
from typing import (Any, Callable, Dict, Generator, Iterable, Optional, Tuple,
                    Union)

from flask import Flask
from grpc import insecure_channel, server

from ..config import ConfigurationManager
from ..config.role import ClientId
from ..role.container import CommunicationId, NodeId
from .comm_pb2 import (EvaluationResult, ProtocolMessage, Registration,
                       RoundModelInfo, UpdateResult)
from .comm_pb2_grpc import (FederatedLearningServicer, FederatedLearningStub,
                            add_FederatedLearningServicer_to_server)
from .communicator import ClientCommunicator, ServerCommunicator
from .events import ClientEvent, ServerEvent

comm_thread_ctx = local()


class _Master(FederatedLearningServicer):
    def __init__(self) -> None:
        super().__init__()
        self._handlers: Dict[ServerEvent, Callable] = dict()
        self._invokation_queues: Dict[CommunicationId, Queue[MsgType]] = dict()
        self._invokation_queue_locks: Dict[CommunicationId, Lock] = dict()
        self._req_queues: Dict[CommunicationId, Queue] = dict()

        self.comm_id_dict: Dict[NodeId, CommunicationId] = dict()

    def _has_handler(self, event: ServerEvent) -> bool:
        return event in self._handlers

    def _check_handler_registration(self, event: ServerEvent) -> None:
        if not self._has_handler(event):
            raise NotImplementedError(f'{event} has not been registrated.')

    def _ensure_req_queue_existance(self, comm_id: CommunicationId) -> Queue:
        if comm_id not in self._req_queues:
            self._req_queues[comm_id] = Queue()
        return self._req_queues[comm_id]

    def _ensure_invokation_queue_existance(self, comm_id: CommunicationId) -> Tuple[Queue, Lock]:
        if comm_id not in self._invokation_queues:
            self._invokation_queues[comm_id] = Queue()
        if comm_id not in self._invokation_queue_locks:
            self._invokation_queue_locks[comm_id] = Lock()
        return self._invokation_queues[comm_id], self._invokation_queue_locks[comm_id]

    def _receive_request(self, request_iterator: Generator, comm_id: CommunicationId) -> None:
        req_queue = self._ensure_req_queue_existance(comm_id)
        _logger = logging.getLogger("Server")
        for req in request_iterator:
            req_queue.put_nowait(req)
            _logger.debug(f'[comm_id: {comm_id}]put req into req_queue: {req}')

    def connect(self, request_iterator, context):
        comm_id = context.peer()
        comm_thread_ctx.comm_id = comm_id
        invokation_queue, invokation_lock = self._ensure_invokation_queue_existance(comm_id)
        Thread(target=partial(self._receive_request, request_iterator, comm_id)).start()
        sleep(0.01)

        req_queue = self._req_queues[comm_id]
        _logger = getLogger('Server')

        req = req_queue.get()
        while True:
            if req is not None:
                _logger.debug(
                    f'[comm_id: {comm_thread_ctx.comm_id}]Request: {req}')
                try:
                    event = ServerEvent(req.event)
                    self._check_handler_registration(event)
                    if event == ServerEvent.Connect or event == ServerEvent.Disconnect or event == ServerEvent.Reconnect:
                        self._handlers[event]()
                    elif event == ServerEvent.WakeUp:
                        self._handlers[event]()
                    elif event == ServerEvent.Ready:
                        registration = req.registration
                        container_id, client_ids = registration.container_id, registration.client_ids
                        self.comm_id_dict[container_id] = comm_id
                        self._handlers[event](container_id, client_ids)
                    elif event == ServerEvent.ResponseUpdate:
                        update_result = req.update_result
                        response = {
                            'cid': update_result.cid,
                            'round_number': update_result.round_number,
                            'local_round_number': update_result.local_round_number,
                            'weights': update_result.weights,
                            'train_size': update_result.train_size,
                            'train_loss': update_result.train_loss,
                            'time_start_update': update_result.time_start_update,
                            'time_finish_update': update_result.time_finish_update,
                            'time_receive_request': update_result.time_receive_request,
                        }
                        self._handlers[event](response)
                    elif event == ServerEvent.ResponseEvaluate:
                        evaluation_result = req.evaluation_result
                        response = {
                            'cid': evaluation_result.cid,
                            'round_number': evaluation_result.round_number,
                            'local_round_number': evaluation_result.local_round_number,
                            'time_start_evaluate': evaluation_result.time_start_evaluate,
                            'time_finish_evaluate': evaluation_result.time_finish_evaluate,
                            'time_receive_request': evaluation_result.time_receive_request,
                            'evaluate': load_from_json_str(evaluation_result.distribute_evaluate),
                        }
                        self._handlers[event](response)
                    else:
                        raise NotImplementedError(f'unknown server event: {event}.')
                except Exception as e:
                    _logger.error(f'[comm_id: {comm_thread_ctx.comm_id}] error: {e}')

            while invokation_queue.qsize() > 0:
                invokation = None
                try:
                    # if not invokation_lock.acquire(blocking=False):
                    #     sleep(1e-3)
                    #     continue
                    invokation = invokation_queue.get_nowait()
                except:
                    invokation = None
                finally:
                    pass
                    # if invokation_lock.locked():
                    #     invokation_lock.release()

                if invokation == None:
                    break
                _logger.debug(f'[comm_id: {comm_thread_ctx.comm_id}]Invokation: {invokation}')

                event, args, kwargs = invokation
                payload, args = args[0], args[1:]
                if event == ClientEvent.Init or event == ClientEvent.Stop:
                    yield ProtocolMessage(event=event.value, empty=None)
                elif event == ClientEvent.RequestUpdate or event == ClientEvent.RequestEvaluate:
                    yield ProtocolMessage(event=event.value, round_model_info=payload)
                else:
                    raise NotImplementedError(f'unreachable client event: {event}.')

            try:
                req = req_queue.get_nowait() if req_queue.qsize() > 0 else None
            except:
                req = None

    def on(self, event: ServerEvent, handler: Callable) -> None:
        assert callable(handler)
        self._handlers[event] = handler

    def put_invokation(self, comm_id: CommunicationId, invokation: Any):
        invokation = deepcopy(invokation)
        invokation_queue, lock = self._ensure_invokation_queue_existance(comm_id)
        have_put = False
        _logger = logging.getLogger("Server")
        while not have_put:
            try:
                # if not lock.acquire(timeout=0.01):
                #     continue
                if not invokation_queue.full():
                    invokation_queue.put_nowait(invokation)
                    have_put = True
                    _logger.debug(f'[comm_id: {comm_id}]put invokation into queue: {invokation}')
            except:
                sleep(0.01)
            finally:
                pass
                # if lock.locked():
                #     lock.release()


MsgType = Optional[Union[EvaluationResult, Registration, RoundModelInfo, UpdateResult]]


class ServerGrpcCommunicator(ServerCommunicator):
    def __init__(self) -> None:
        super().__init__()

        # Flask services
        current_path = os.path.dirname(os.path.abspath(__file__))
        self._app = Flask(__name__, template_folder=os.path.join(current_path, 'templates'),
                          static_folder=os.path.join(current_path, 'static'))
        self._app.config['SECRET_KEY'] = ConfigurationManager().runtime_config.secret_key

        # gRPC
        self._master = _Master()

    def handle_disconnection(self) -> Iterable[ClientId]:
        return self._handle_disconnection(comm_thread_ctx.comm_id)

    def handle_reconnection(self) -> Iterable[ClientId]:
        return self._handle_reconnection(comm_thread_ctx.comm_id)

    def activate(self, node_id: NodeId, client_ids: Iterable[ClientId]) -> None:
        comm_id = self._master.comm_id_dict[node_id]
        self._activate(node_id, comm_id, client_ids)

    def invoke(self, event: ClientEvent, *args, callee: Optional[ClientId] = None, **kwargs):
        comm_id = comm_thread_ctx.comm_id
        if callee is not None:
            with self._client_node_ctx_mgr.get_by_client(callee) as c_node_ctx:
                comm_id = c_node_ctx.comm_id
        self._master.put_invokation(comm_id, (event, (None,) + args, kwargs))

    def invoke_all(self, event: ClientEvent, payload: Optional[Dict[str, Any]] = None, *args, callees: Optional[Iterable[ClientId]] = None, **kwargs):

        def get_invokation(selected_client_ids: Optional[Iterable[ClientId]] = None) -> Tuple[ClientEvent, Tuple, Dict[str, Any]]:
            _payload = dict() if payload is None else payload
            if selected_client_ids is not None:
                _payload['selected_clients'] = list(selected_client_ids)
            return (event, (_payload,) + args, kwargs)

        if callees is None: # broadcast
            for client_id in self._client_node_ctx_mgr.online_client_ids:
                with self._client_node_ctx_mgr.get_by_client(client_id) as _ctx:
                    self._master.put_invokation(_ctx.comm_id, get_invokation())
        else:   # multicast
            for node_id, client_ids in self._client_node_ctx_mgr.cluster_by_node(callees).items():
                with self._client_node_ctx_mgr.get_by_node(node_id) as _ctx:
                    self._master.put_invokation(_ctx.comm_id, get_invokation(client_ids))

    def on(self, event: ClientEvent) -> Callable[[Callable], Any]:
        def decorator(func: Callable):
            assert callable(func)
            self._master.on(event, func)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return None
            return wrapper
        return decorator

    def route(self, rule: str, **options: Any):
        return self._app.route(rule, **options)

    def run_server(self) -> None:
        _grpc_server = server(ThreadPoolExecutor(
            max_workers=int(ConfigurationManager().runtime_config.container_num * 1.5)))
        add_FederatedLearningServicer_to_server(self._master, _grpc_server)
        comm_port = ConfigurationManager().runtime_config.comm_port
        _grpc_server.add_insecure_port(f'0.0.0.0:{comm_port}')
        _grpc_server.start()
        self._app.run(host=self._host, port=self._port)
        # _grpc_server.wait_for_termination()


class ClientGrpcCommunicator(ClientCommunicator):
    def __init__(self) -> None:
        super().__init__()
        self._handlers: Dict[ClientEvent, Callable] = dict()
        self._invokation_queue: Queue[MsgType] = Queue()
        comm_port = ConfigurationManager().runtime_config.comm_port
        self._channel = insecure_channel(
            f'{self._host}:{comm_port}', options=(('grpc.enable_http_proxy', 0),))

    def _on(self, event: ClientEvent, handler: Callable) -> None:
        assert callable(handler)
        self._handlers[event] = handler

    def _has_handler(self, event: ClientEvent) -> bool:
        return event in self._handlers

    def _check_handler_registration(self, event: ClientEvent) -> None:
        if not self._has_handler(event):
            raise NotImplementedError(f'{event} has not been registrated.')

    def on(self, event: ClientEvent, *on_args, **on_kwargs):
        def decorator(func: Callable):
            self._on(event, func)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return None
            return wrapper
        return decorator

    def invoke(self, event: ServerEvent, *args, **kwargs):
        invokation = deepcopy((event, args, kwargs))
        have_put = False
        while not have_put:
            try:
                self._invokation_queue.put_nowait(invokation)
                have_put = True
            except:
                sleep(0.01)

    def _invokation_generator(self):
        # yield msg that should be sent to the central server but queued
        while True:
            while self._invokation_queue.qsize() > 0:
                try:
                    invokation = self._invokation_queue.get_nowait()
                except:
                    invokation = None
                    sleep(0.01)
                if invokation is None:
                    continue

                event, args, kwargs = invokation
                if event == ServerEvent.Connect or event == ServerEvent.Disconnect or event == ServerEvent.Reconnect:
                    raise ValueError(f'clients should not issue such event to the central server: {event}')
                elif event == ServerEvent.WakeUp:
                    yield ProtocolMessage(event=event.value, empty=None)
                elif event == ServerEvent.Ready:
                    yield ProtocolMessage(
                        event=event.value,
                        registration = Registration(
                            container_id=args[0],
                            client_ids=args[1],
                        ),
                    )
                elif event == ServerEvent.ResponseUpdate:
                    result = args[0]
                    yield ProtocolMessage(
                        event=event.value,
                        update_result=UpdateResult(
                            cid=result['cid'],
                            round_number=result['round_number'],
                            local_round_number=result['local_round_number'],
                            weights=result['weights'],
                            train_size=result['train_size'],
                            train_loss=result['train_loss'],
                            time_receive_request=result['time_receive_request'],
                            time_start_update=result['time_start_update'],
                            time_finish_update=result['time_finish_update'],
                        ),
                    )
                elif event == ServerEvent.ResponseEvaluate:
                    result = args[0]
                    yield ProtocolMessage(
                        event=event.value,
                        evaluation_result=EvaluationResult(
                            cid=result['cid'],
                            round_number=result['round_number'],
                            local_round_number=result['local_round_number'],
                            time_receive_request=result['time_receive_request'],
                            time_start_evaluate=result['time_start_evaluate'],
                            time_finish_evaluate=result['time_finish_evaluate'],
                            evaluate=dump_into_json_str(result['evaluate']),
                        ),
                    )
                else:
                    raise ValueError(f'unreachable client event: {event}')

            if self._invokation_queue.qsize() == 0:
                sleep(0.01)

    def wait(self, **kw) -> None:
        stub = FederatedLearningStub(self._channel)
        sleep(0.2 * ConfigurationManager().runtime_config.container_num)
        responses = stub.connect(self._invokation_generator())
        for response in responses:
            event = ClientEvent(response.event)
            if event == ClientEvent.Connect or event == ClientEvent.Disconnect or event == ClientEvent.Reconnect:
                self._handlers[event]()
            elif event == ClientEvent.Init:
                self._handlers[event]()
            elif event == ClientEvent.Stop:
                self._handlers[event]()
                return
            elif event == ClientEvent.RequestUpdate or event == ClientEvent.RequestEvaluate:
                round_model_info = response.round_model_info
                round_model_info = {
                    'round_number': round_model_info.round_number,
                    'weights_file_name': round_model_info.weights_file_name,
                    'selected_clients': round_model_info.selected_clients,
                }
                self._handlers[event](round_model_info)
            else:
                raise ValueError(f'unreachable server event: {event}')
