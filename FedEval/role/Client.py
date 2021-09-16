import os
import time
from typing import Any, Callable, Mapping

from ..config import ConfigurationManager, Role
from ..utils.utils import obj_to_pickle_string
from .container import ClientContextManager
from .flask_node import FlaskNode, ServerSocketIOEvent
from .model_weights_io import weights_filename_pattern


class Client(FlaskNode):
    """a client node implementation based on FlaskNode."""

    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self):
        ConfigurationManager().role = Role.Client
        super().__init__('client')
        container_id = int(os.environ.get('CONTAINER_ID', 0))
        self._init_logger(container_id)
        self._ctx_mgr = ClientContextManager(container_id, self.log_dir)

        self._register_handles()
        self.start()

    def _init_logger(self, container_id):
        super()._init_logger('container', f'Container{container_id}')

    def _register_handles(self):
        from . import ClientSocketIOEvent

        @self.on(ClientSocketIOEvent.Connect)
        def on_connect():
            print('connect')
            self.logger.info('connect')

        @self.on(ClientSocketIOEvent.Disconnect)
        def on_disconnect():
            print('disconnect')
            self.logger.info('disconnect')

        @self.on(ClientSocketIOEvent.Reconnect)
        def on_reconnect():
            print('reconnect')
            self.logger.info('reconnect')

        @self.on(ClientSocketIOEvent.Init)
        def on_init():
            self.logger.info('on init')
            self.logger.info("local model initialized done.")
            self.invoke(ClientSocketIOEvent.Ready, [self._ctx_mgr.container_id] + self._cid_list)

        @self.on(ClientSocketIOEvent.RequestUpdate)
        def on_request_update(data_from_server: Mapping[str, Any], response_with: Callable):

            # Mark the receive time
            time_receive_request = time.time()

            # Call backs for debug, could be removed in the future
            response_with('Train Received by', self._ctx_mgr.container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            self.logger.info('Debug: selected clients' + str(selected_clients))

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_update = time.time()
                self.logger.info(f"### Round {current_round}, Cid {cid} ###")
                with self._ctx_mgr.get(cid) as client_ctx:
                    client_ctx.step_forward_local_train_round()
                    self.logger.debug('Debug: %s' % (current_round - client_ctx.host_params_round))

                    # Download the parameter if the local model is not the latest
                    if (current_round - client_ctx.host_params_round) > 1:
                        client_ctx.host_params_round = current_round - 1
                        weights_file_name = weights_filename_pattern.format(client_ctx.host_params_round)
                        weights = self._model_weights_io_handler.fetch_params(weights_file_name)
                        client_ctx.strategy.set_host_params_to_local(weights, current_round=current_round)
                        self.logger.info(f"train received model: {weights_file_name}")

                    # fit on local and retrieve new uploading params
                    train_loss, train_data_size = client_ctx.strategy.fit_on_local_data()
                    self.logger.info(f"Local train loss {train_loss}")

                    upload_data = client_ctx.strategy.retrieve_local_upload_info()
                    weights_as_string = obj_to_pickle_string(upload_data)
                    time_finish_update = time.time()    # Mark the update finish time

                    response = {
                        'cid': client_ctx.id,
                        'round_number': current_round,
                        'local_round_number': client_ctx.local_train_round,
                        'weights': weights_as_string,
                        'train_size': train_data_size,
                        'train_loss': train_loss,
                        'time_start_update': time_start_update,
                        'time_finish_update': time_finish_update,
                        'time_receive_request': time_receive_request,
                    }

                    # TMP
                    self.logger.info(
                        f"Local train time: {time_finish_update - time_start_update}")

                    self.logger.info("Emit client_update")
                    try:
                        self.invoke(ClientSocketIOEvent.ResponseUpdate, response)
                        self.logger.info("sent trained model to server")
                    except Exception as e:
                        self.logger.error(e)
                    self.logger.info(f"Client {client_ctx.id} Emited update")

        @self.on(ClientSocketIOEvent.RequestEvaluate)
        def on_request_evaluate(data_from_server: Mapping[str, Any], response_with: Callable):

            time_receive_evaluate = time.time()

            # Call backs
            response_with('Evaluate Received by', self._ctx_mgr.container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_evaluate = time.time()
                with self._ctx_mgr.get(cid) as client_ctx:
                    # Download the latest weights
                    weights_file_name = data_from_server.get('weights_file_name')
                    if weights_file_name is None and current_round > client_ctx.host_params_round:
                        weights_file_name = weights_filename_pattern.format(current_round)

                    if weights_file_name:
                        client_ctx.host_params_round = current_round
                        weights = self._model_weights_io_handler.fetch_params(weights_file_name)
                        client_ctx.strategy.set_host_params_to_local(weights, current_round=current_round)
                        self.logger.info(f"eval received model: {weights_file_name}")

                    evaluate = client_ctx.strategy.local_evaluate()

                    self.logger.info("Local Evaluate" + str(evaluate))

                    time_finish_evaluate = time.time()

                    response = {
                        'cid': client_ctx.id,
                        'round_number': current_round,
                        'local_round_number': client_ctx.local_train_round,
                        'time_start_evaluate': time_start_evaluate,
                        'time_finish_evaluate': time_finish_evaluate,
                        'time_receive_request': time_receive_evaluate,
                        'evaluate': evaluate
                    }

                    self.logger.info("Emit client evaluate")
                    try:
                        self.invoke(ClientSocketIOEvent.ResponseEvaluate, response)
                        self.logger.info("sent evaluation results to server")
                    except Exception as e:
                        self.logger.error(e)
                    self.logger.info(f"Client {client_ctx.id} Emited evaluate")

        @self.on(ClientSocketIOEvent.Stop)
        def on_stop():
            # TODO(fgh): stop all the clients
            for cid in self._ctx_mgr.client_ids:
                with self._ctx_mgr.get(cid) as client_ctx:
                    client_ctx.strategy.client_exit_job(self)
            print("Federated training finished ...")
            exit(0)

    def start(self):
        self.invoke(ServerSocketIOEvent.WakeUp)
        self.logger.info("sent wakeup")
        self.wait()
