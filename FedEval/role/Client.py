import os
import gc
import time
import json
import pickle
import requests
import logging
import numpy as np

from socketIO_client import SocketIO
from ..utils import pickle_string_to_obj, obj_to_pickle_string
from ..strategy import *


class Client(object):

    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self, data_config, model_config, runtime_config):

        # (1) Name
        self.name = 'client'
        self.cid = os.environ.get('CLIENT_ID', '0')
        # (2) Logger
        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        self.logger = logging.getLogger("client")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(runtime_config.get('log_dir', 'log'), 'Client' + self.cid, time_str)
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
        
        fed_model = eval(model_config['FedModel']['name'])
        self.fed_model = fed_model(
            role=self, data_config=data_config, model_config=model_config, runtime_config=runtime_config
        )

        server_host, server_port = self.fed_model.param_parser.parse_server_addr(self.name)

        self.weights_download_url = 'http://' + server_host + ':%s' % server_port + '/download/'
        self.local_train_round = 0
        self.host_params_round = -1
        self.global_weights = None

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
            self.sio.emit('client_ready')

        def on_request_update(*args):

            time_receive_request = time.time()

            data_from_server, callback = args[0], args[1]

            # Call backs
            callback('Train Received by', os.environ.get('CLIENT_ID'))

            current_round = data_from_server['round_number']
            self.logger.info("### Round {} ###".format(current_round))
            self.local_train_round += 1

            if (current_round - self.host_params_round) > 1:
                weights_file_name = 'model_{}.pkl'.format(current_round - 1)
                self.host_params_round = current_round - 1
                response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
                self.fed_model.set_host_params_to_local(pickle.loads(response.content), current_round=current_round)
                self.logger.info("train received model: %s" % weights_file_name)

            # 2 fit on local and retrieve new uploading params
            train_loss, train_data_size = self.fed_model.fit_on_local_data()

            upload_data = self.fed_model.retrieve_local_upload_info()

            time_finish_update = time.time()

            response = {
                'cid': os.environ.get('CLIENT_ID'),
                'round_number': current_round,
                'local_round_number': self.local_train_round,
                'weights': obj_to_pickle_string(upload_data),
                'train_size': train_data_size,
                'train_loss': train_loss,
                'time_finish_update': time_finish_update,
                'time_receive_request': time_receive_request,
            }

            self.logger.info("Emit client_update")
            try:
                self.sio.emit('client_update', response)
                self.logger.info("sent trained model to server")
            except Exception as e:
                self.logger.error(e)
            self.logger.info("Emited...")

        def on_request_evaluate(*args):

            data_from_server, callback = args[0], args[1]

            # Call backs
            callback('Evaluate Received by', os.environ.get('CLIENT_ID'))

            current_round = data_from_server['round_number']

            if data_from_server.get('weights_file_name'):
                weights_file_name = data_from_server['weights_file_name']
            elif (current_round - self.host_params_round) > 0:
                weights_file_name = 'model_{}.pkl'.format(current_round)
            else:
                weights_file_name = None

            if weights_file_name:
                self.host_params_round = current_round
                response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
                self.fed_model.set_host_params_to_local(pickle.loads(response.content), current_round=current_round)
                self.logger.info("eval received model: %s" % weights_file_name)

            time_receive_evaluate = time.time()

            evaluate = self.fed_model.local_evaluate()

            time_finish_evaluate = time.time()

            response = {
                'cid': os.environ.get('CLIENT_ID'),
                'round_number': current_round,
                'local_round_number': self.local_train_round,
                'time_finish_update': time_finish_evaluate,
                'time_receive_request': time_receive_evaluate,
                'evaluate': evaluate
            }

            self.logger.info("Emit client evaluate")
            try:
                self.sio.emit('client_evaluate', response)
                self.logger.info("sent evaluation results to server")
            except Exception as e:
                self.logger.error(e)
            self.logger.info("Emited...")

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
