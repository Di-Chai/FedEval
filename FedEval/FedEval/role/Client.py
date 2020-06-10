import os
import time
import json
import gc
import numpy as np
import pickle
import requests
import logging

from ..model.BaseModel import parse_strategy_and_compress, recover_to_weights
from socketIO_client import SocketIO
from ..utils import pickle_string_to_obj, obj_to_pickle_string


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
                 local_batch_size,
                 local_num_rounds,
                 upload_name_filter,
                 upload_sparse,
                 upload_strategy,
                 ignore_load=True,):

        self.ignore_load = ignore_load

        self.local_model = None
        self.dataset = None

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.local_batch_size = local_batch_size
        self.local_num_rounds = local_num_rounds

        self.upload_name_filter = upload_name_filter
        self.upload_sparse = upload_sparse
        self.upload_strategy = upload_strategy

        self.model = model
        self.name = client_name

        date_str = time.strftime('%m%d')
        time_str = time.strftime('%m%d%H%M')

        self.weights_download_url = 'http://' + server_host + ':%s' % server_port + '/download/'
        self.local_train_round = None
        self.global_weights = None

        # logger
        self.logger = logging.getLogger("client")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(model.model_dir, model.code_version, self.name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.fh = logging.FileHandler(os.path.join(self.log_dir, date_str + '.log'))
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
            self.model.build()
            self.logger.info("local model initialized done.")
            # ready to be dispatched for training
            self.sio.emit('client_ready')

        def on_request_update(*args):

            receive_update_time = time.time()

            # Call backs
            args[1]('Train Received by', os.environ.get('CLIENT_ID'))

            info_from_server = args[0]
            cur_round = info_from_server['round_number']
            self.logger.info("### Round {} ###".format(cur_round))

            self.local_train_round = cur_round

            if cur_round == 0:
                weights_file_name = info_from_server['current_weights_file']
                response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
                weights = pickle.loads(response.content)
                self.model.set_weights(weights)
                self.logger.info("received initial model")

            old_weights = self.model.get_weights(upload_name_filter=self.upload_name_filter)

            del self.global_weights
            self.global_weights = old_weights

            train_loss, train_size = self.model.train_one_round(
                self.train_data,
                epoch=self.local_num_rounds,
                batch_size=self.local_batch_size,
            )

            new_weights = self.model.get_weights(upload_name_filter=self.upload_name_filter)

            upload_data = parse_strategy_and_compress(old_weights, new_weights, upload_sparse=self.upload_sparse,
                                                      upload_strategy=self.upload_strategy)

            finish_update_time = time.time()

            resp = {
                'cid': os.environ.get('CLIENT_ID'),
                'round_number': cur_round,
                'weights': obj_to_pickle_string(upload_data),
                'train_size': train_size,
                'train_loss': train_loss,
                'receive_update_time': receive_update_time,
                'finish_update_time': finish_update_time
            }

            del upload_data
            del new_weights
            gc.collect()

            self.logger.info("Emit client_update")
            try:
                self.sio.emit('client_update', resp)
                self.logger.info("sent trained model to server")
            except Exception as e:
                self.logger.error(e)
            self.logger.info("Emited...")

        def on_evaluate(*args):

            self.logger.info('Received Eval from server')

            # Callback to response
            args[1]('Eval Received by', os.environ.get('CLIENT_ID'))

            info_from_server = args[0]

            # Update local weight
            weights_file_name = info_from_server['current_weights_file']
            response = requests.get(self.weights_download_url + weights_file_name, timeout=600)
            weights = pickle.loads(response.content)

            current_round = info_from_server['round_number']

            if current_round == self.local_train_round:
                old_weights = self.global_weights
            else:
                old_weights = self.model.get_weights()

            if current_round == 0:
                if len(old_weights['optimizer']) == 0:
                    old_weights['optimizer'] = self.model.get_weights()['optimizer']

            receive_time = time.time()

            weights = recover_to_weights(old_weights, weights,
                                         upload_strategy=self.upload_strategy)

            self.model.set_weights(weights)

            self.logger.info('Weight updated')

            # val
            val_result = self.model.evaluate(self.val_data)
            # test
            test_result = self.model.evaluate(self.test_data)

            eval_result_to_server = {
                'receive_time': receive_time,
                'finish_eval_time': time.time(),
                'cid': os.environ.get('CLIENT_ID')
            }

            eval_result_to_server.update({'val_' + key: value for key, value in val_result.items()})
            eval_result_to_server.update({'test_' + key: value for key, value in test_result.items()})

            self.logger.info('Eval finished')
            try:
                self.sio.emit('client_eval', eval_result_to_server)
                self.logger.info("sent eval result to server")
            except Exception as e:
                self.logger.error(e)
            self.logger.info("Emited...")

            del weights
            del old_weights
            del eval_result_to_server
            gc.collect()

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

        def on_stop(*args):
            print("Federated training finished ...")
            exit(0)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('evaluate', on_evaluate)
        self.sio.on('stop', on_stop)
        self.sio.on('check_client_resource', on_check_client_resource)