import os
import time
import json
import logging

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
                 local_batch_size=None,
                 local_num_rounds=1,
                 ignore_load=True,):

        self.ignore_load = ignore_load

        self.local_model = None
        self.dataset = None

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.local_batch_size = local_batch_size
        self.local_num_rounds = local_num_rounds

        self.model = model
        self.name = client_name

        date_str = time.strftime('%m%d')
        time_str = time.strftime('%m%d%H%M')

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

        def on_disconnect():
            print('disconnect')
            # self.sio.emit('client_wake_up')

        def on_reconnect():
            print('reconnect')

        def on_init():
            self.logger.info('on init')
            if not self.model.is_build:
                self.model.build()
                self.logger.info("local model initialized done.")
            # ready to be dispatched for training
            self.sio.emit('client_ready')

        def on_request_update(*args):

            receive_update_time = time.time()

            info_from_server = args[0]

            # call back function
            args[1]('Train Received', os.environ.get('CLIENT_ID'))

            cur_round = info_from_server['round_number']
            self.logger.info("### Round {} ###".format(cur_round))

            if cur_round == 0:
                self.logger.info("received initial model")
                weights = pickle_string_to_obj(info_from_server['current_weights'])
                self.model.set_weights(weights)

            my_weights, train_loss, train_size = self.model.train_one_round(
                self.train_data,
                epoch=self.local_num_rounds,
                batch_size=self.local_batch_size,
            )

            pickle_string_weights = obj_to_pickle_string(my_weights)
            resp = {'cid': os.environ.get('CLIENT_ID'), 'round_number': cur_round, 'weights': pickle_string_weights, 'train_size': train_size,
                    'train_loss': train_loss, 'receive_update_time': receive_update_time,
                    'finish_update_time': time.time()}

            print("Emit client_update")
            self.sio.emit('client_update', resp)
            self.logger.info("sent trained model to server")
            print("Emited...")

        def on_evaluate(*args):

            receive_time = time.time()

            args[1]('Eval Received', os.environ.get('CLIENT_ID'))

            self.logger.info('Received Eval from server')

            # Update local weight
            info_from_server = args[0]
            weights = pickle_string_to_obj(info_from_server['current_weights'])
            self.model.set_weights(weights)

            self.logger.info('Weight updated')

            # val
            val_result = self.model.evaluate(self.val_data)
            # test
            test_result = self.model.evaluate(self.test_data)

            eval_result_to_server = {
                'receive_time': receive_time,
                'finish_eval_time': time.time()
            }

            eval_result_to_server.update({'val_' + key: value for key, value in val_result.items()})
            eval_result_to_server.update({'test_' + key: value for key, value in test_result.items()})

            self.logger.info('Eval finished')

            print("Emit client_eval")
            self.sio.emit('client_eval', eval_result_to_server)

        def on_check_client_resource(*args):
            req = args[0]
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