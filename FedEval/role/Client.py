import os
import time
from typing import Any, Callable, Dict, List, Mapping

from ..config.configuration import ConfigurationManager
from ..strategy import *
from ..utils.utils import obj_to_pickle_string
from .container import ContainerId
from .flask_node import FlaskNode, ServerSocketIOEvent
from .model_weights_io import weights_filename_pattern
from .role import Role


class Client(FlaskNode):
    """a client node implementation based on FlaskNode."""

    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self):
        ConfigurationManager().role = Role.Client
        super().__init__('client')
        self._cid = os.environ.get('CLIENT_ID', '0') # TODO(fgh) remove self._cid
        self._container_id = os.environ.get('CONTAINER_ID', '0')
        self._init_control_states()
        self._init_container()

        self._init_logger()
        self._register_handles()
        self.start()

    def _init_control_states(self):
        ''' allocate cid for the clients hold by this container and
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

    def _init_logger(self):
        super()._init_logger('container', 'Container' + self._container_id)

    def _init_container(self):
        # Initialize the fed model for all the clients in this container
        # This method should be called after self._init_control_states()
        mdl_cfg = ConfigurationManager().model_config
        fed_model_type: type = eval(mdl_cfg.strategy_name)
        self.client_fed_model_fname = os.path.join(self.log_dir, 'client_%s_fed_model')
        biggest_cid_in_this_container = max(self._cid_list)
        for cid in self._cid_list:
            start = time.time()
            client_fed_model_fpath = self.client_fed_model_fname % cid
            self.fed_model = fed_model_type()
            self.fed_model.set_client_id(cid)
            self.fed_model = save_fed_model(self.fed_model, client_fed_model_fpath)
            if cid != biggest_cid_in_this_container:
                del self.fed_model
            self.logger.info('Saving model costs %s(s)' % (time.time() - start))
        self.curr_online_client = biggest_cid_in_this_container

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
            self.invoke(ClientSocketIOEvent.Ready, [self._container_id] + self._cid_list)

        @self.on(ClientSocketIOEvent.RequestUpdate)
        def on_request_update(data_from_server: Mapping[str, Any], response_with: Callable):

            # Mark the receive time
            time_receive_request = time.time()

            # Call backs for debug, could be removed in the future
            response_with('Train Received by', self._container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            self.logger.info('Debug: selected clients' + str(selected_clients))

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_update = time.time()
                self.logger.info("### Round {}, Cid {} ###".format(current_round, cid))
                if self.curr_online_client != cid:
                    start = time.time()
                    # Save before load
                    self.fed_model = save_fed_model(
                        self.fed_model, self.client_fed_model_fname % self.curr_online_client
                    )
                    self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % cid)
                    self.curr_online_client = cid
                    self.logger.info("Loaded fed model of client %s using %s" % (cid, time.time()-start))
                self._local_train_round[cid] += 1
                # Download the parameter if the local model is not the latest
                self.logger.debug('Debug: %s' % (current_round - self._host_params_round[cid]))
                if (current_round - self._host_params_round[cid]) > 1:
                    self._host_params_round[cid] = current_round - 1
                    weights_file_name = weights_filename_pattern.format(self._host_params_round[cid])
                    weights = self._model_weights_io_handler.fetch_params(weights_file_name)
                    self.fed_model.set_host_params_to_local(weights, current_round=current_round)
                    self.logger.info("train received model: %s" % weights_file_name)

                # fit on local and retrieve new uploading params
                train_loss, train_data_size = self.fed_model.fit_on_local_data()
                upload_data = self.fed_model.retrieve_local_upload_info()

                self.logger.info("Local train loss %s" % train_loss)

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
                    self.invoke(ClientSocketIOEvent.ResponseUpdate, response)
                    self.logger.info("sent trained model to server")
                except Exception as e:
                    self.logger.error(e)
                self.logger.info("Client %s Emited update" % cid)

        @self.on(ClientSocketIOEvent.RequestEvaluate)
        def on_request_evaluate(data_from_server: Mapping[str, Any], response_with: Callable):

            time_receive_evaluate = time.time()

            # Call backs
            response_with('Evaluate Received by', self._container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_evaluate = time.time()
                if self.curr_online_client != cid:
                    # Save before load
                    self.fed_model = save_fed_model(
                        self.fed_model, self.client_fed_model_fname % self.curr_online_client
                    )
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

                self.logger.info("Local Evaluate" + str(evaluate))

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
                    self.invoke(ClientSocketIOEvent.ResponseEvaluate, response)
                    self.logger.info("sent evaluation results to server")
                except Exception as e:
                    self.logger.error(e)
                self.logger.info("Client %s Emited evaluate" % cid)

        @self.on(ClientSocketIOEvent.Stop)
        def on_stop():
            self.fed_model.client_exit_job(self)
            print("Federated training finished ...")
            exit(0)

    def start(self):
        self.invoke(ServerSocketIOEvent.WakeUp)
        self.logger.info("sent wakeup")
        self.wait()
