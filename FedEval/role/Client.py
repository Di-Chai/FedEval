import os
import gc
import time
import json
import pickle
import requests
import logging
import numpy as np
import joblib

from socketIO_client import SocketIO
from ..utils import obj_to_pickle_string
from ..strategy import *


class Client(object):

    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self, data_config, model_config, runtime_config):

        # (1) Name
        self.name = 'client'
        # self.cid = os.environ.get('CLIENT_ID', '0')
        self.container_id = os.environ.get('CONTAINER_ID', '0')
        # (2) Logger
        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        self.logger = logging.getLogger("container")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(runtime_config.get('log_dir', 'log'), 'Container' + self.container_id, time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        self.fh = logging.FileHandler(os.path.join(self.log_dir, 'train.log'))
        self.fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)
        self.ch.setFormatter(self.formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        
        # Initialize the fed model for all the clients in this container
        num_containers = runtime_config['docker']['num_containers']
        num_clients = runtime_config['server']['num_clients']
        num_clients_in_this_container = int(num_clients / num_containers)
        cid_start = int(self.container_id) * num_clients_in_this_container
        if num_clients % num_containers != 0:
            if num_clients % num_containers > int(self.container_id):
                num_clients_in_this_container += 1
                cid_start += int(self.container_id)
            else:
                cid_start += num_clients % num_containers
        
        fed_model_class = eval(model_config['FedModel']['name'])
        self.client_fed_model_fname = os.path.join(self.log_dir, 'client_%s_fed_model')
        self.local_train_round = {}
        self.host_params_round = {}
        self.cid_list = []
        for cid in range(cid_start, cid_start + num_clients_in_this_container):
            start = time.time()
            if os.path.isdir(self.client_fed_model_fname % cid) is False:
                os.makedirs(self.client_fed_model_fname % cid, exist_ok=True)
            self.fed_model = fed_model_class(
                role='client_%s' % cid, data_config=data_config, model_config=model_config, runtime_config=runtime_config
            )
            self.fed_model = save_fed_model(self.fed_model, self.client_fed_model_fname % cid)
            if cid < (cid_start + num_clients_in_this_container - 1):
                del self.fed_model
            self.local_train_round[cid] = 0
            self.host_params_round[cid] = -1
            self.cid_list.append(cid)
            self.logger.info('Save model using %s' % (time.time() - start))

        # Load arbitrary fed model
        start = time.time()
        self.curr_online_client = self.cid_list[0]
        self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % self.curr_online_client)
        print('#' * 30)
        print('Load fed model costing', time.time() - start)
        server_host, server_port = self.fed_model.param_parser.parse_server_addr(self.name)

        self.weights_download_url = 'http://' + server_host + ':%s' % server_port + '/download/'

        # Start the connection
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        self.logger.info("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def register_handles(self):

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
            self.sio.emit('client_ready', [self.container_id] + self.cid_list)
        
        def on_request_update(*args):
            
            # Mark the receive time
            time_receive_request = time.time()

            data_from_server, callback = args[0], args[1]

            # Call backs for debug, could be removed in the future
            callback('Train Received by', self.container_id)

            # Get the selected clients
            selected_clients = data_from_server['selected_clients']

            self.logger.info('Debug: selected clients' + str(selected_clients))

            current_round = data_from_server['round_number']

            for cid in selected_clients:
                time_start_update = time.time()
                self.logger.info("### Round {}, Cid {} ###".format(current_round, cid))
                if self.curr_online_client != cid:
                    start = time.time()
                    self.fed_model = load_fed_model(self.fed_model, self.client_fed_model_fname % cid)
                    self.curr_online_client = cid
                    self.logger.info("Loaded fed model of client %s using %s" % (cid, time.time()-start))
                self.local_train_round[cid] += 1
                # Download the parameter if the local model is not the latest
                self.logger.info('Debug: %s' % (current_round - self.host_params_round[cid]))
                if (current_round - self.host_params_round[cid]) > 1:
                    weights_file_name = 'model_{}.pkl'.format(current_round - 1)
                    self.host_params_round[cid] = current_round - 1
                    response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
                    self.fed_model.set_host_params_to_local(pickle.loads(response.content), current_round=current_round)
                    self.logger.info("train received model: %s" % weights_file_name)

                # fit on local and retrieve new uploading params
                train_loss, train_data_size = self.fed_model.fit_on_local_data()
                upload_data = self.fed_model.retrieve_local_upload_info()

                self.logger.info("Local train loss %s" % train_loss)

                # Save current client
                # self.fed_model = save_fed_model(
                #     self.fed_model, self.client_fed_model_fname % self.curr_online_client
                # )

                # Mark the update finish time
                time_finish_update = time.time()

                response = {
                    'cid': os.environ.get('CLIENT_ID'),
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

            data_from_server, callback = args[0], args[1]

            # Call backs
            callback('Evaluate Received by', self.container_id)

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
                if data_from_server.get('weights_file_name'):
                    weights_file_name = data_from_server['weights_file_name']
                elif (current_round - self.host_params_round[cid]) > 0:
                    weights_file_name = 'model_{}.pkl'.format(current_round)
                else:
                    weights_file_name = None

                if weights_file_name:
                    self.host_params_round[cid] = current_round
                    response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
                    self.fed_model.set_host_params_to_local(pickle.loads(response.content), current_round=current_round)
                    self.logger.info("eval received model: %s" % weights_file_name)

                evaluate = self.fed_model.local_evaluate()

                self.logger.info("Local Evaluate" + str(evaluate))

                # Save current client
                # self.fed_model = save_fed_model(
                #     self.fed_model, self.client_fed_model_fname % self.curr_online_client
                # )

                time_finish_evaluate = time.time()

                response = {
                    'cid': os.environ.get('CLIENT_ID'),
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

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('request_evaluate', on_request_evaluate)
        self.sio.on('stop', on_stop)
