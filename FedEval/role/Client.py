import os
import gc
import time
import json
import pickle
import requests
import logging
import numpy as np

from ..model.BaseModel import parse_strategy_and_compress, recover_to_weights
from socketIO_client import SocketIO
from ..utils import pickle_string_to_obj, obj_to_pickle_string
from ..strategy import *


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class Client(object):
    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self,
                 server_host,
                 server_port,
                 model,
                 train_data,
                 val_data,
                 test_data,
                 client_name,
                 fed_model_name,
                 train_strategy,
                 upload_strategy,
                 ignore_load=True, ):

        self.ignore_load = ignore_load

        self.local_model = None
        self.dataset = None

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.name = client_name

        fed_model = parse_strategy_name(fed_model_name)

        self.fed_model = fed_model(
            role='client', model=model, upload_strategy=upload_strategy,
            train_strategy=train_strategy,
            train_data=train_data, val_data=val_data, test_data=test_data
        )

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())

        self.weights_download_url = 'http://' + server_host + ':%s' % server_port + '/download/'
        self.local_train_round = 0
        self.host_params_round = -1
        self.global_weights = None

        # logger
        self.logger = logging.getLogger("client")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(model.model_dir, self.name, time_str)
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
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        self.logger.info("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def load_stat(self):
        loadavg = {}
        with open("/proc/loadavg") as fin:
            con = fin.read().split()
            loadavg['lavg_1'] = con[0]
            loadavg['lavg_5'] = con[1]
            loadavg['lavg_15'] = con[2]
            loadavg['nr'] = con[3]
            loadavg['last_pid'] = con[4]
        return loadavg['lavg_15']

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
            train_loss, train_size = self.fed_model.fit_on_local_data(
                train_data=self.train_data,
                epochs=self.fed_model.train_strategy['E'],
                batch_size=self.fed_model.train_strategy['B']
            )

            upload_data = self.fed_model.retrieve_local_upload_info()

            time_finish_update = time.time()

            response = {
                'cid': os.environ.get('CLIENT_ID'),
                'round_number': current_round,
                'local_round_number': self.local_train_round,
                'weights': obj_to_pickle_string(upload_data),
                'train_size': train_size,
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

        def on_check_client_resource(*args):
            req = args[0]
            args[1]('Check Res', req['rid'])
            print("check client resource.")
            if self.ignore_load:
                load_average = 0.15
                print("Ignore load average")
            else:
                load_average = self.load_stat()
                print("Load average:", load_average)

            resp = {
                'round_number': req['round_number'],
                'load_rate': load_average
            }
            self.sio.emit('check_client_resource_done', resp)

        def on_stop():
            print("Federated training finished ...")
            exit(0)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('request_evaluate', on_request_evaluate)
        self.sio.on('stop', on_stop)
        self.sio.on('check_client_resource', on_check_client_resource)
