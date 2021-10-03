from enum import Enum

from ..config import ConfigurationManager
from .communicator import ClientCommunicator, ServerCommunicator
from .events import ClientEvent, ConnectionEvent, ServerEvent
from .flask_communicator import (ClientFlaskCommunicator,
                                 ServerFlaskCommunicator, Sid)
from .grpc_communicator import ClientGrpcCommunicator, ServerGrpcCommunicator
from .model_weights_io import (ModelWeightsFlaskHandler,
                               ModelWeightsIoInterface,
                               server_best_weight_filename,
                               weights_filename_pattern)


class CommunicationMethod(Enum):
    gRPC = 'gRPC'
    SocketIO = 'SocketIO'

def get_client_communicator() -> ClientCommunicator:
    comm_method = ConfigurationManager().runtime_config.comm_method
    comm_method = CommunicationMethod(comm_method)
    if comm_method == CommunicationMethod.gRPC:
        return ClientGrpcCommunicator()
    elif comm_method == CommunicationMethod.SocketIO:
        return ClientFlaskCommunicator()
    else:
        raise NotImplementedError(f'unreachable code')

def get_server_communicator() -> ServerCommunicator:
    comm_method = ConfigurationManager().runtime_config.comm_method
    comm_method = CommunicationMethod(comm_method)
    if comm_method == CommunicationMethod.gRPC:
        return ServerGrpcCommunicator()
    elif comm_method == CommunicationMethod.SocketIO:
        return ServerFlaskCommunicator()
    else:
        raise NotImplementedError(f'unreachable code')
