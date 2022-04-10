import os
import time
from typing import Any, Mapping

from ..communicaiton import ModelWeightsHandler, get_client_communicator
from ..communicaiton.events import *
from ..config import ConfigurationManager, Role, ServerFlaskInterface
from ..utils.utils import obj_to_pickle_string
from .container import ClientContextManager
from .logger import HyperLogger
from .node import Node


class Client(Node):
    """a client node implementation based on FlaskNode."""

    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self):
        cfg_mgr = ConfigurationManager()
        cfg_mgr.role = Role.Client
        super().__init__()
        container_id = int(os.environ.get('CONTAINER_ID', 0))
        self._init_logger(container_id)
        self._ctx_mgr = ClientContextManager(container_id, self._hyper_logger._log_dir_path)
        self._ctx_mgr.set_logger(self.logger)
        # self._communicator = ClientFlaskCommunicator()
        self._communicator = get_client_communicator()

        central_server_service_addr = cfg_mgr.runtime_config.central_server_addr
        listen_port = cfg_mgr.runtime_config.central_server_port
        download_url_pattern = f'{central_server_service_addr}:{listen_port}{ServerFlaskInterface.DownloadPattern.value}'
        self._model_weights_io_handler = ModelWeightsHandler(download_url_pattern)
        self._register_handles()
        self.start()

    def _init_logger(self, container_id):
        self._hyper_logger = HyperLogger('container', f'Container{container_id}')
        self.logger = self._hyper_logger.get()

    @property
    def log_dir(self):
        return self._hyper_logger.log_dir_path

    def _register_handles(self):
        @self._communicator.on(ClientEvent.Connect)
        def on_connect():
            print('connect')
            self.logger.info('connect')

        @self._communicator.on(ClientEvent.Disconnect)
        def on_disconnect():
            print('disconnect')
            self.logger.info('disconnect')

        @self._communicator.on(ClientEvent.Reconnect)
        def on_reconnect():
            print('reconnect')
            self.logger.info('reconnect')

        @self._communicator.on(ClientEvent.Init)
        def on_init():
            self.logger.info('on init')
            self.logger.info("local model initialized done.")
            self._communicator.invoke(
                ServerEvent.Ready, self._ctx_mgr.container_id, list(self._ctx_mgr.client_ids))

        @self._communicator.on(ClientEvent.RequestUpdate)
        def on_request_update(data_from_server: Mapping[str, Any]):

            # Mark the receive time
            time_receive_request = time.time()

            # Get the selected clients and weights information
            selected_clients = data_from_server['selected_clients']
            current_round = data_from_server['round_number']
            encoded_weights_file_path: str = data_from_server['weights_file_name']

            rt_cfg = ConfigurationManager().runtime_config
            if rt_cfg.comm_fast_mode:
                shared_parameter = None

            for cid in selected_clients:
                time_start_update = time.time()
                self.logger.info(f"### Round {current_round}, Cid {cid} ###")
                with self._ctx_mgr.get(cid) as client_ctx:
                    client_ctx.step_forward_local_train_round()
                    # Download the parameter if the local model is not the latest
                    if (current_round - client_ctx.host_params_round) > 1:
                        client_ctx.host_params_round = current_round - 1
                        if rt_cfg.comm_fast_mode:
                            if shared_parameter is None:
                                shared_parameter = self._model_weights_io_handler.fetch_params(encoded_weights_file_path)
                            client_ctx.strategy.set_host_params_to_local(shared_parameter, current_round=current_round)
                        else:
                            weights = self._model_weights_io_handler.fetch_params(encoded_weights_file_path)
                            client_ctx.strategy.set_host_params_to_local(weights, current_round=current_round)
                        self.logger.info(f"train received model: {encoded_weights_file_path}")

                    # fit on local and retrieve new uploading params
                    client_fit_results = client_ctx.strategy.fit_on_local_data()
                    if client_fit_results:
                        train_loss, train_data_size = client_fit_results
                    else:
                        train_loss, train_data_size = -1, -1
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

                    self.logger.info("Emit client_update")
                    try:
                        self._communicator.invoke(
                            ServerEvent.ResponseUpdate, response)
                        self.logger.info("sent trained model to server")
                    except Exception as e:
                        self.logger.error(e)
                    self.logger.info(f"Client {client_ctx.id} Emited update")

        @self._communicator.on(ClientEvent.RequestEvaluate)
        def on_request_evaluate(data_from_server: Mapping[str, Any]):

            time_receive_evaluate = time.time()

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            current_round = data_from_server['round_number']

            rt_cfg = ConfigurationManager().runtime_config
            if rt_cfg.comm_fast_mode:
                shared_parameter = None

            # Download the latest weights
            encoded_weights_file_path: str = data_from_server['weights_file_name']
            for cid in selected_clients:
                time_start_evaluate = time.time()
                with self._ctx_mgr.get(cid) as client_ctx:
                    client_ctx.host_params_round = current_round
                    if rt_cfg.comm_fast_mode:
                        if shared_parameter is None:
                            shared_parameter = self._model_weights_io_handler.fetch_params(encoded_weights_file_path)
                        client_ctx.strategy.set_host_params_to_local(shared_parameter, current_round=current_round)
                    else:
                        weights = self._model_weights_io_handler.fetch_params(encoded_weights_file_path)
                        client_ctx.strategy.set_host_params_to_local(weights, current_round=current_round)
                    self.logger.info(f"eval received model: {encoded_weights_file_path}")

                    evaluate = client_ctx.strategy.local_evaluate()
                    evaluate = evaluate or {}

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
                        self._communicator.invoke(
                            ServerEvent.ResponseEvaluate, response)
                        self.logger.info("sent evaluation results to server")
                    except Exception as e:
                        self.logger.error(e)
                    self.logger.info(f"Client {client_ctx.id} Emited evaluate")

        @self._communicator.on(ClientEvent.Stop)
        def on_stop():
            for cid in self._ctx_mgr.client_ids:
                with self._ctx_mgr.get(cid) as client_ctx:
                    client_ctx.strategy.client_exit_job(self)
            print("Federated training finished ...")
            exit(0)

    def start(self):
        self._communicator.invoke(ServerEvent.WakeUp)
        self.logger.info("sent wakeup")
        self._communicator.wait()
