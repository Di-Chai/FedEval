# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import comm_pb2 as comm__pb2


class FederatedLearningStub(object):
    """message TrainingStatus {
    bool finished = 1;
    int64 rounds = 2;
    string log_dir = 3;
    # results = 4;
    }

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.connect = channel.stream_stream(
                '/FederatedLearning/connect',
                request_serializer=comm__pb2.ProtocolMessage.SerializeToString,
                response_deserializer=comm__pb2.ProtocolMessage.FromString,
                )


class FederatedLearningServicer(object):
    """message TrainingStatus {
    bool finished = 1;
    int64 rounds = 2;
    string log_dir = 3;
    # results = 4;
    }

    """

    def connect(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FederatedLearningServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'connect': grpc.stream_stream_rpc_method_handler(
                    servicer.connect,
                    request_deserializer=comm__pb2.ProtocolMessage.FromString,
                    response_serializer=comm__pb2.ProtocolMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'FederatedLearning', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FederatedLearning(object):
    """message TrainingStatus {
    bool finished = 1;
    int64 rounds = 2;
    string log_dir = 3;
    # results = 4;
    }

    """

    @staticmethod
    def connect(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/FederatedLearning/connect',
            comm__pb2.ProtocolMessage.SerializeToString,
            comm__pb2.ProtocolMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)