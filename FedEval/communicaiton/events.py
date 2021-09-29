from enum import Enum


class ConnectionEvent(Enum):
    """basic SocketIO life-cycle event names."""
    Connect = 'connect'
    Disconnect = 'disconnect'
    Reconnect = 'reconnect'


class ServerEvent(ConnectionEvent, Enum):
    """server-side SocketIO event handles' name."""
    WakeUp = 'client_wake_up'
    Ready = 'client_ready'
    ResponseUpdate = 'client_update'
    ResponseEvaluate = 'client_evaluate'


class ClientEvent(ConnectionEvent, Enum):
    """client-side SocketIO event handles' name."""
    Init = 'init'
    RequestUpdate = 'request_update'
    RequestEvaluate = 'request_evaluate'
    Stop = 'stop'

def event2message(event: ConnectionEvent) -> str:
    return event.value
