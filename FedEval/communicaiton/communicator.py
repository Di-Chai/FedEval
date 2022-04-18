from abc import ABC, abstractmethod
from platform import system
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import psutil

from ..communicaiton.events import ClientEvent, ServerEvent
from ..config.configuration import ConfigurationManager
from ..config.role import ClientId
from ..role.container import ClientNodeContextManager, CommunicationId, NodeId


def _get_comm_in_and_out_linux() -> Tuple[int, int]:
    """retrieve network traffic counter in linux platforms (with the support from psutil lib).

    Returns:
        Tuple[int, int]: (the number of received bytes, the number of sent bytes)
    """
    eth0_info = psutil.net_io_counters(pernic=True).get('eth0')
    bytes_recv = 0 if eth0_info is None else eth0_info.bytes_recv
    bytes_sent = 0 if eth0_info is None else eth0_info.bytes_sent
    return bytes_recv, bytes_sent


class Communicatior(ABC):
    @staticmethod
    def get_comm_in_and_out() -> Tuple[int, int]:
        """retrieve network traffic counter.

        Raises:
            NotImplementedError: raised when called at unsupported
                platforms or unknown platforms.

        Returns:
            Tuple[int, int]: (the number of received bytes, the number of sent bytes)
        """
        platform_str = system().lower()
        if platform_str == 'linux' or platform_str == 'darwin' or platform_str == 'windows':
            return _get_comm_in_and_out_linux()
        elif platform_str == 'windows':
            raise NotImplementedError(
                f'Unsupported function at {platform_str} platform.')
        else:
            raise NotImplementedError("Unknown platform.")


class ClientCommunicator(Communicatior):
    def __init__(self) -> None:
        super().__init__()
        rt_cfg = ConfigurationManager().runtime_config
        self._host = rt_cfg.central_server_addr
        self._port = rt_cfg.central_server_port

    @abstractmethod
    def on(self, event: ClientEvent, *on_args, **on_kwargs):
        pass

    @abstractmethod
    def invoke(self, event: ServerEvent, *args, **kwargs):
        pass

    @abstractmethod
    def wait(self, **kw) -> None:
        pass


class ServerCommunicator(Communicatior):
    def __init__(self) -> None:
        super().__init__()
        rt_cfg = ConfigurationManager().runtime_config
        self._host = rt_cfg.central_server_listen_at
        self._port = rt_cfg.central_server_port
        self._client_node_ctx_mgr = ClientNodeContextManager()

    @abstractmethod
    def handle_disconnection(self) -> Iterable[ClientId]:
        pass

    def _handle_disconnection(self, comm_id: CommunicationId) -> Iterable[ClientId]:
        return self._client_node_ctx_mgr.deactivate_by_comm(comm_id)

    @abstractmethod
    def handle_reconnection(self) -> Iterable[ClientId]:
        pass

    def _handle_reconnection(self, comm_id: CommunicationId) -> Iterable[ClientId]:
        with self._client_node_ctx_mgr.get_by_comm(comm_id) as ctx:
            self._client_node_ctx_mgr.recover_from_deactivation(ctx.id)
            recovered_client_ids = ctx.client_ids
        return recovered_client_ids

    @abstractmethod
    def activate(self, node_id: NodeId, client_ids: Iterable[ClientId]) -> None:
        pass

    def _activate(self, node_id: NodeId, comm_id: CommunicationId, client_ids: Iterable[ClientId]) -> None:
        self._client_node_ctx_mgr.activate(node_id, comm_id, client_ids)

    @abstractmethod
    def invoke(self, event: ClientEvent, *args, callee: Optional[ClientId] = None, **kwargs):
        pass

    @abstractmethod
    def invoke_all(self, event: ClientEvent, payload: Optional[Dict[str, Any]] = None, *args, callees: Optional[Iterable[ClientId]] = None, **kwargs):
        pass

    @abstractmethod
    def on(self, event: ClientEvent) -> Callable[[Callable], Any]:
        pass

    @abstractmethod
    def route(self, rule: str, **options: Any):
        pass

    @property
    def ready_client_ids(self) -> Iterable[ClientId]:
        return sorted(self._client_node_ctx_mgr.online_client_ids)
    
    @abstractmethod
    def run_server(self) -> None:
        pass
