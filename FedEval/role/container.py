import os
from contextlib import contextmanager
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Union)

from ..config import ClientId, ConfigurationManager
from ..strategy import *  # for strategy construction
from ..utils.utils import obj_to_pickle_string, pickle_string_to_obj

ContainerId = int   # to identify container

class ClientContext:
    def __init__(self, id: ClientId, fed_strategy: type, temp_dir_path: str) -> None:
        self._strategy: Optional[FedStrategyInterface] = fed_strategy()
        self._strategy.load_data_with(id)

        self._local_train_round: int = 0
        self._host_params_round: int = -1
        self._id: ClientId = id
        self._sleep_files_base_path = os.path.join(temp_dir_path, f'client_{id}_fed_model')
        self.__ml_model_tmp: Optional[Any] = None

    @property
    def host_params_round(self) -> int:
        return self._host_params_round

    @host_params_round.setter
    def host_params_round(self, v: int):
        self._host_params_round = v

    def step_forward_host_params_round(self) -> int:
        self._host_params_round += 1
        return self.host_params_round

    @property
    def local_train_round(self) -> int:
        return self._local_train_round

    @local_train_round.setter
    def local_train_round(self, v: int) -> int:
        self._local_train_round = v

    def step_forward_local_train_round(self) -> int:
        self._local_train_round += 1
        return self.local_train_round

    @property
    def id(self) -> ClientId:
        return self._id

    @property
    def strategy(self) -> FedStrategyInterface:
        assert self._strategy is not None
        return self._strategy

    def sleep(self) -> None:
        self.__ml_model_tmp = self._strategy.ml_model
        ClientContext._save(self._strategy, self._sleep_files_base_path)
        self._strategy = None

    @staticmethod
    def _save(strategy, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        # ML model weights
        ml_model = strategy.ml_model
        ml_model.save_weights(os.path.join(
            path, 'ml_model.h5'), save_format='h5')
        strategy.ml_model = None
        # Data
        data = {}
        data_keys = ['train_data', 'val_data', 'test_data']
        for key in data_keys:
            if hasattr(strategy, key):
                data[key] = getattr(strategy, key)
                setattr(strategy, key, None)
        data_file_path = os.path.join(path, 'data.pkl')
        if len(data) > 0 and os.path.isfile(data_file_path) is False:
            obj_to_pickle_string(data, data_file_path)
        # Other attributes
        obj_to_pickle_string(strategy, os.path.join(path, 'fed_model.pkl'))

    def wake_up(self) -> None:
        if self.awake:
            return
        path = self._sleep_files_base_path
        # Other attributes
        new_strategy = pickle_string_to_obj(os.path.join(path, 'fed_model.pkl'))
        # ML model weights
        new_strategy.ml_model = self.__ml_model_tmp
        self.__ml_model_tmp = None
        new_strategy.ml_model.load_weights(os.path.join(path, 'ml_model.h5'))
        # Data
        data = pickle_string_to_obj(os.path.join(path, 'data.pkl'))
        data_keys = ['train_data', 'val_data', 'test_data']
        for key in data_keys:
            setattr(new_strategy, key, data[key])

        self._strategy = new_strategy

    @property
    def awake(self) -> bool:
        return self.__ml_model_tmp is None

    @property
    def sleeping(self) -> bool:
        return not self.awake


class ClientContextManager:
    def __init__(self, id: Union[ContainerId, str], tmp_dir_path: str) -> None:
        self._id: ContainerId = id if isinstance(id, ContainerId) else int(id)

        fed_model_type: type = eval(
            ConfigurationManager().model_config.strategy_name)
        client_id_list = self._allocate_client_ids()
        self._clients: Dict[ClientId, ClientContext] = dict()
        for cid in client_id_list:
            client_ctx = ClientContext(cid, fed_model_type, tmp_dir_path)
            client_ctx.sleep()
            self._clients[cid] = client_ctx

        self._curr_client_ctx: Optional[ClientContext] = None

    def _allocate_client_ids(self) -> Sequence[ClientId]:
        '''allocate cid for the clients hold by this container and
        initiate round counters.

        client_cids allocation examples:
        # case 0:
        Given: container_num: 2, client_num: 13
        Thus:
        num_clients_in_each_container -> 6
        num_clients % num_containers -> 1
        ## container_0
        client_cids: [0..=6]
        ## container_1
        client_cids: [7..=12]

        # case 1:
        Given: container_num: 3, client_num: 13
        Thus:
        num_clients_in_each_container -> 4
        num_clients % num_containers -> 1
        ## container_0
        client_cids: [0..=4]
        ## container_1
        client_cids: [5..=8]
        ## container_2
        client_cids: [9..=12]
        '''
        rt_cfg = ConfigurationManager().runtime_config
        num_containers = rt_cfg.container_num
        num_clients = rt_cfg.client_num
        num_clients_in_this_container = num_clients // num_containers
        cid_start = self._id * num_clients_in_this_container

        num_clients_remainder = num_clients % num_containers
        if num_clients_remainder != 0:
            if num_clients_remainder > self._id:
                num_clients_in_this_container += 1
                cid_start += self._id
            else:
                cid_start += num_clients_remainder

        _cid_list: List[ClientId] = list(range(cid_start, cid_start + num_clients_in_this_container))
        return _cid_list

    @property
    def container_id(self) -> ContainerId:
        return self._id

    @property
    def client_ids(self) -> Iterable[ClientId]:
        return self._clients.keys()

    @contextmanager
    def get(self, client_id: ClientId) -> ClientContext:
        # TODO(fgh) add unittest for this method
        if client_id not in self._clients:
            raise ValueError(f'unknown client with an ID of {client_id}')
        if self._curr_client_ctx.id != client_id:
            self._curr_client_ctx.sleep()
            self._curr_client_ctx = self._clients[client_id]
            self._curr_client_ctx.wake_up()
        try:
            yield self._curr_client_ctx
        finally:
            pass


CommunicationId = Any   # TODO(fgh) into Generic Type
NodeId = ContainerId    # TODO(fgh) into Generic Type

class ClientNodeContext:
    def __init__(self, id: NodeId, comm_id: CommunicationId, client_ids: Iterable[ClientId]) -> None:
        self._id: NodeId = id
        self._comm_id: CommunicationId = comm_id
        self._client_ids: Set[ClientId] = set(client_ids)

    @property
    def id(self) -> NodeId:
        return self._id

    @property
    def comm_id(self) -> CommunicationId:
        return self._comm_id

    @property
    def client_ids(self) -> Iterable[ClientId]:
        return list(self._client_ids)

    @client_ids.setter
    def client_ids(self, value: Iterable[ClientId]):
        self._client_ids = set(value)


class ClientNodeContextManager:
    def __init__(self) -> None:
        self._client_ids : Dict[ClientId, NodeId] = dict()
        self._comm_ids: Dict[NodeId, CommunicationId] = dict()
        self._nodes: Dict[CommunicationId, ClientNodeContext] = dict()
        self._offline_node_ids: Set[NodeId] = set()

    def activate(self, node_id: NodeId, comm_id: CommunicationId, client_ids: Iterable[ClientId]):
        if comm_id in self._nodes:
            node = self._nodes[comm_id]
            assert node.id == node_id
            if node_id in self._offline_node_ids:
                self._offline_node_ids.remove(node_id)
            node.client_ids = client_ids
        else:
            new_node = ClientNodeContext(node_id, comm_id, client_ids)
            self._nodes[comm_id] = new_node

        self._comm_ids[node_id] = comm_id
        for client_id in client_ids:
            self._client_ids[client_id] = node_id

    def recover_from_deactivation(self, node_id: NodeId):
        assert node_id in self._offline_node_ids
        self._offline_node_ids.remove(node_id)

    def deactivate_by_node_id(self, node_id: NodeId) -> Iterable[ClientId]:
        """mark the given node as offline and return the corresponding offline clients.

        Args:
            node_id (NodeId): the id of the offline node

        Returns:
            Iterable[ClientId]: the client on the given offline node.
        """
        self._offline_node_ids.add(node_id)
        return list(self._nodes[self._comm_ids[node_id]].client_ids)

    def deactivate_by_comm(self, comm_id: CommunicationId) -> Iterable[ClientId]:
        self._offline_node_ids.add(self._nodes[comm_id].id)
        return list(self._nodes[comm_id].client_ids)

    @contextmanager
    def get_by_node(self, node_id: NodeId) -> ClientNodeContext:
        comm_id = self._comm_ids[node_id]
        try:
            yield self._nodes[comm_id]
        finally:
            pass

    @contextmanager
    def get_by_client(self, client_id: ClientId) -> ClientNodeContext:
        node_id = self._client_ids[client_id]
        comm_id = self._comm_ids[node_id]
        try:
            yield self._nodes[comm_id]
        finally:
            pass

    @contextmanager
    def get_by_comm(self, comm_id: CommunicationId) -> ClientNodeContext:
        try:
            yield self._nodes[comm_id]
        finally:
            pass

    def cluster_by_node(self, selected_clients: Iterable[ClientId]) -> Mapping[NodeId, Iterable[ClientId]]:
        selected_clients = set(selected_clients)
        cluster: Dict[NodeId, Optional[List[ClientId]]] = dict()
        for client_id in selected_clients:
            node_id = self._client_ids[client_id]
            if node_id not in cluster:
                cluster[node_id] = list()
            cluster[node_id].append(client_id)
        return cluster

    @property
    def online_client_ids(self) -> Iterable[ClientId]:
        return [client_id for client_id, node_id in self._client_ids.items()
                if node_id not in self._offline_node_ids]
