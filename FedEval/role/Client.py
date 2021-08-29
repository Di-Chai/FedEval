import os
import time
from typing import Dict, List

from socketIO_client import SocketIO
from utils.utils import obj_to_pickle_string

from ..strategy import *
from .flask_node import FlaskNode
from .role import Role
from .service_interface import ServerFlaskInterface
from .model_weights_io import ModelWeightsFlaskHandler, ModelWeightsIoInterface, weights_filename_pattern

class Client(FlaskNode):
    """a client node implementation based on FlaskNode."""

    MAX_DATASET_SIZE_KEPT = 6000

    def _init_control_states(self):
        ''' allocate cid for the clients hold by this container and
        initiate round counters.
        
        client_cids allocation examples:
        # case 0: Given: container_num: 2, client_num: 13
        Thus:
        num_clients_in_each_container -> 6
        num_clients % num_containers -> 1
        # container_0
        client_cids: [0..=6]
        # container_1
        client_cids: [7..=12]

        # case 1: Given: container_num: 3, client_num: 13
        Thus:
        num_clients_in_each_container -> 4
        num_clients % num_containers -> 1
        # container_0
        client_cids: [0..=4]
        # container_1
        client_cids: [5..=8]
        # container_2
        client_cids: [9..=12]
        '''
        num_containers = self.runtime_config['docker']['num_containers']
        num_clients = self.runtime_config['server']['num_clients']
        num_clients_in_this_container = num_clients // num_containers
        cid_start = int(self._container_id) * num_clients_in_this_container

        num_clients_remainder = num_clients % num_containers
        if num_clients_remainder != 0:
            if num_clients_remainder > int(self._container_id):
                num_clients_in_this_container += 1
                cid_start += int(self._container_id)
            else:
                cid_start += num_clients_remainder

        self._cid_list: List[ContainerId] = list(range(cid_start, cid_start + num_clients_in_this_container))
        self._local_train_round: Dict[ContainerId, int] = { cid: 0 for cid in self._cid_list }
        self._host_params_round: Dict[ContainerId, int] = { cid: -1 for cid in self._cid_list }

    def __init__(self, data_config, model_config, runtime_config):
        super().__init__('client', data_config, model_config, runtime_config, role=Role.Client)
        self._cid = os.environ.get('CLIENT_ID', '0') # TODO remove self._cid
        self._container_id = os.environ.get('CONTAINER_ID', '0')
        self._init_control_states()
        self._init_container()
        self._init_no_use()

        self._init_logger()
        self.start()

    def _init_no_use(self):
        self.global_weights = None

    def start(self):
        self._register_handles()
        self.sio.emit('client_wake_up')
        self.logger.info("sent wakeup")
        self.sio.wait()

    def _init_flask_service(self):
        server_host, server_port = self.fed_model.param_parser.parse_server_addr(self.name)
        weights_download_url = f'http://{server_host}:{server_port}{ServerFlaskInterface.DownloadPattern}'
        self._model_weights_io_handler: ModelWeightsIoInterface = ModelWeightsFlaskHandler(
            weights_download_url)
        self.sio = SocketIO(server_host, server_port)
        self._register_handles()

    def _init_logger(self):
        super()._init_logger('container', 'Container' + self._container_id)

    def _init_container(self):
        # Initialize the fed model for all the clients in this container
        # This method should be called after self._init_control_states()
        fed_model_class: type = eval(self.model_config['FedModel']['name'])
        self.client_fed_model_fname = os.path.join(self.log_dir, 'client_%s_fed_model')
        biggest_cid_in_this_container = max(self._cid_list)
        for cid in self._cid_list:
            start = time.time()
            client_fed_model_fpath = self.client_fed_model_fname % cid
            self.fed_model = fed_model_class(
                role=Role.Client, data_config=self.data_config, model_config=self.model_config, runtime_config=self.runtime_config
            )
            self.fed_model = save_fed_model(self.fed_model, client_fed_model_fpath)
            if cid != biggest_cid_in_this_container:
                del self.fed_model  # Q(fgh): 这里已经留下了最后一个初始化的fed_model了，为什么还要在后面再加载一个fed_model呢？
                # self.curr_online_client = biggest_cid_in_this_container
            self.logger.info('Saving model costs %s(s)' % (time.time() - start))

        # Load arbitrary fed model
        start = time.time()
        self.curr_online_client = self._cid_list[0]
        self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % self.curr_online_client)
        print('#' * 30)
        print('Loading federated model costs %s(s)' % (time.time() - start))

    def _register_handles(self):
        from . import ClientSocketIOEvents

        def on_connect():
            print('connect')
            self.logger.info('connect')

        def on_disconnect():
            print('disconnect')
            self.logger.info('disconnect')

        def on_reconnect():
            print('reconnect')
            self.logger.info('reconnect')

        def on_init():
            self.logger.info('on init')
            self.logger.info("local model initialized done.")
            self.sio.emit('client_ready', [self._container_id] + self._cid_list)

        def on_request_update(*args):

            # Mark the receive time
            time_receive_request = time.time()

            data_from_server, response_with = args[0], args[1]

            # Call backs for debug, could be removed in the future
            response_with('Train Received by', self._container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_update = time.time()
                self.logger.info("### Round {}, Cid {} ###".format(current_round, cid))
                if self.curr_online_client != cid:
                    start = time.time()
                    self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % cid)
                    self.curr_online_client = cid
                    self.logger.info("Loaded fed model of client %s using %s" % (cid, time.time()-start))
                self._local_train_round[cid] += 1
                # Download the parameter if the local model is not the latest
                if (current_round - self._host_params_round[cid]) > 1:
                    self._host_params_round[cid] = current_round - 1
                    weights_file_name = weights_filename_pattern.format(self._host_params_round[cid])
                    weights = self._model_weights_io_handler.fetch_params(weights_file_name)
                    self.fed_model.set_host_params_to_local(weights, current_round=current_round)
                    self.logger.info("train received model: %s" % weights_file_name)

                # fit on local and retrieve new uploading params
                train_loss, train_data_size = self.fed_model.fit_on_local_data()
                upload_data = self.fed_model.retrieve_local_upload_info()

                # Save current client
                # self.fed_model = save_fed_model(
                #     self.fed_model, self.client_fed_model_fname % self.curr_online_client
                # )

                # Mark the update finish time
                time_finish_update = time.time()

                response = {
                    'cid': self._cid,
                    'round_number': current_round,
                    'local_round_number': self.local_train_round[cid],
                    'weights': obj_to_pickle_string(upload_data),
                    'train_size': train_data_size,
                    'train_loss': train_loss,
                    'time_start_update': time_start_update,
                    'time_finish_update': time_finish_update,
                    'time_receive_request': time_receive_request,
                }

                # TMP
                self.logger.info("Local train time: %s" % (time_finish_update - time_start_update))

                self.logger.info("Emit client_update")
                try:
                    self.sio.emit('client_update', response)
                    self.logger.info("sent trained model to server")
                except Exception as e:
                    self.logger.error(e)
                self.logger.info("Client %s Emited update" % cid)

        def on_request_evaluate(*args):

            time_receive_evaluate = time.time()

            data_from_server, response_with = args[0], args[1]

            # Call backs
            response_with('Evaluate Received by', self._container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_evaluate = time.time()
                if self.curr_online_client != cid:
                    start = time.time()
                    self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % cid)
                    self.curr_online_client = cid
                    self.logger.info("Loaded fed model of client %s using %s" % (cid, time.time()-start))
                # Download the latest weights
                weights_file_name = data_from_server.get('weights_file_name')
                if weights_file_name is None and current_round > self._host_params_round[cid]:
                    weights_file_name = weights_filename_pattern.format(current_round)

                if weights_file_name:
                    self._host_params_round[cid] = current_round
                    weights = self._model_weights_io_handler.fetch_params(weights_file_name)
                    self.fed_model.set_host_params_to_local(weights, current_round=current_round)
                    self.logger.info("eval received model: %s" % weights_file_name)

                evaluate = self.fed_model.local_evaluate()

                # Save current client
                # self.fed_model = save_fed_model(
                #     self.fed_model, self.client_fed_model_fname % self.curr_online_client
                # )

                time_finish_evaluate = time.time()

                response = {
                    'cid': self._cid,
                    'round_number': current_round,
                    'local_round_number': self.local_train_round,
                    'time_start_evaluate': time_start_evaluate,
                    'time_finish_evaluate': time_finish_evaluate,
                    'time_receive_request': time_receive_evaluate,
                    'evaluate': evaluate
                }

                self.logger.info("Emit client evaluate")
                try:
                    self.sio.emit('client_evaluate', response)
                    self.logger.info("sent evaluation results to server")
                except Exception as e:
                    self.logger.error(e)
                self.logger.info("Client %s Emited evaluate" % cid)

        def on_stop():
            self.fed_model.client_exit_job(self)
            print("Federated training finished ...")
            exit(0)

        self.sio.on(ClientSocketIOEvents.Connect.value, on_connect)
        self.sio.on(ClientSocketIOEvents.Disconnect.value, on_disconnect)
        self.sio.on(ClientSocketIOEvents.Reconnect.value, on_reconnect)
        self.sio.on(ClientSocketIOEvents.Init.value, on_init)
        self.sio.on(ClientSocketIOEvents.RequestUpdate.value, on_request_update)
        self.sio.on(ClientSocketIOEvents.RequestEvaluate.value, on_request_evaluate)
        self.sio.on(ClientSocketIOEvents.Stop.value, on_stop)
