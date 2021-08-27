import os
import re
import datetime
import json
import logging
import random
import threading
import time

import numpy as np
import psutil
from flask import request, Flask, render_template, send_file
from flask_socketio import SocketIO, emit

from ..strategy import *
from ..utils import pickle_string_to_obj, obj_to_pickle_string
from ..run_util import save_config


class Server(object):

    def __init__(self, data_config, model_config, runtime_config):

        # (1) Name
        self.name = 'server'
        # (2) Logger
        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        self.logger = logging.getLogger("Server")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(runtime_config.get('log_dir', 'log'), 'Server', time_str)
        self.log_file = os.path.join(self.log_dir, 'train.log')
        os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(self.log_file, encoding='utf8')
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        fed_model = eval(model_config['FedModel']['name'])
        self.fed_model = fed_model(
            role='server', data_config=data_config, model_config=model_config, runtime_config=runtime_config,
            logger=self.logger
        )

        self.run_config = self.fed_model.param_parser.parse_run_config()
        self.num_clients = self.run_config['num_clients']
        self.max_num_rounds = self.run_config['max_num_rounds']
        self.num_tolerance = self.run_config['num_tolerance']
        self.num_clients_contacted_per_round = self.run_config['num_clients_contacted_per_round']
        self.rounds_between_val = self.run_config['rounds_between_val']
        self.lazy_update = False
        
        self.ready_container_sid_dict = {}
        self.ready_container_id_dict = {}
        self.ready_clients = []

        self.host, self.port = self.fed_model.param_parser.parse_server_addr(self.name)
        self.client_resource = {}

        self.time_send_train = None
        self.time_agg_train_start = None
        self.time_agg_train_end = None
        self.time_agg_eval_start = None
        self.time_agg_eval_end = None
        self.time_record = []
        self.server_send_bytes = 0
        self.server_receive_bytes = 0
        
        self.thread_lock = threading.Lock()

        self.STOP = False
        self.server_job_finish = False

        self.logger.info(self.run_config)
        self.logger.info(self.get_model_description())

        self.current_params = self.fed_model.host_get_init_params()

        # weights should be a ordered list of parameter
        # for stats
        self.avg_val_metrics = []
        self.avg_test_metrics = []

        # for convergence check
        self.best_val_metric = None
        self.best_test_metric = {}
        self.best_test_metric_full = None
        self.best_weight = None
        self.best_round = -1

        self.training_start_time = time.time()
        self.training_stop_time = None

        self.model_path = os.path.abspath(self.log_dir)
        self.weight_filename = 'model_{}.pkl'
        self.best_weight_filename = 'best_model.pkl'

        # training states
        self.current_round = 0
        self.c_up = []
        self.c_eval = []
        self.check_list = []
        self.info_each_round = {}

        # save the init weights
        self.save_weight()

        current_path = os.path.dirname(os.path.abspath(__file__))
        self.app = Flask(__name__, template_folder=os.path.join(current_path, 'templates'),
                         static_folder=os.path.join(current_path, 'static'))
        self.app.config['SECRET_KEY'] = 'secret!'
        self.socketio = SocketIO(self.app, max_http_buffer_size=10 ** 20, async_handlers=True,
                                 ping_timeout=3600, ping_interval=1800, cors_allowed_origins='*')

        # socket io messages
        self.register_handles()
        self.invalid_tolerate = 0

        self.client_sids_selected = None

        @self.app.route('/dashboard')
        def dashboard():

            if len(self.avg_test_metrics) > 0:
                avg_test_metric_keys = [e for e in list(self.avg_test_metrics[0].keys()) if e != 'time']
            else:
                avg_test_metric_keys = []

            if len(self.avg_val_metrics) > 0:
                avg_val_metric_keys = [e for e in list(self.avg_val_metrics[0].keys()) if e != 'time']
            else:
                avg_val_metric_keys = []

            time_record = [e for e in self.time_record if len(e.keys()) >= 6]
            if len(time_record) > 0:
                time_record.append({'round': 'Average'})
                for key in time_record[0]:
                    if key not in ['round', 'eval_receive_time']:
                        time_record[-1][key] = np.mean([e[key] for e in time_record[:-1]])

            time_record = [time_record[i] for i in range(len(time_record)) if (len(time_record) - i) <= 6]

            if self.STOP and self.training_stop_time is not None:
                current_used_time = self.training_stop_time - self.training_start_time
            else:
                current_used_time = int(round(time.time())) - self.training_start_time
            m, s = divmod(current_used_time, 60)
            h, m = divmod(m, 60)

            metrics = self.fed_model.model_config['MLModel'].get('metrics')
            if metrics is not None and len(metrics) > 0:
                test_accuracy_key = 'test_' + metrics[0]
            else:
                test_accuracy_key = 'test_accuracy'

            return render_template(
                'dashboard.html',
                status='Finish' if self.STOP else 'Running',
                rounds="%s / %s" % (self.current_round, self.max_num_rounds),
                num_online_clients="%s / %s / %s" % (self.num_clients_contacted_per_round,
                                                     len(self.ready_clients), self.num_clients),
                avg_test_metric=self.avg_test_metrics,
                avg_test_metric_keys=avg_test_metric_keys,
                avg_val_metric=self.avg_val_metrics,
                avg_val_metric_keys=avg_val_metric_keys,
                time_record=time_record,
                current_used_time="%02d:%02d:%02d" % (h, m, s),
                test_accuracy=self.best_test_metric.get(test_accuracy_key, 0),
                test_loss=self.best_test_metric.get('test_loss', 0),
                server_send=self.server_send_bytes / (2 ** 30),
                server_receive=self.server_receive_bytes/(2**30)
            )

        # TMP use
        @self.app.route('/status')
        def status_page():
            return json.dumps({
                'finished': self.server_job_finish,
                'rounds': self.current_round,
                'results': [
                    None if len(self.avg_val_metrics) == 0 else self.avg_val_metrics[-1], 
                    None if len(self.avg_test_metrics) == 0 else self.avg_test_metrics[-1]
                    ],
                'log_dir': self.log_dir,
            })
        
        @self.app.route("/download/<filename>", methods=['GET'])
        def download_file(filename):
            if os.path.isfile(os.path.join(self.model_path, filename)):
                return send_file(os.path.join(self.model_path, filename), as_attachment=True)
            else:
                return json.dumps({'status': 404, 'msg': 'file not found'})

    # cur_round could None
    def aggregate_train_loss(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        return aggr_loss

    def get_model_description(self):
        return_value = """\nmodel parameters:\n"""
        for attr in dir(self.fed_model):
            attr_value = getattr(self.fed_model, attr)
            if type(attr_value) in [str, int, float] and attr.startswith('_') is False:
                return_value += "{}={}\n".format(attr, attr_value)
        return return_value

    def save_weight(self):
        obj_to_pickle_string(
            self.current_params,
            os.path.join(self.model_path, self.weight_filename.format(self.current_round))
        )
        # Keep the latest 5 weights
        all_files_in_model_dir = os.listdir(self.model_path)
        matched_model_files = [re.match(r'model_([0-9]+).pkl', e) for e in all_files_in_model_dir]
        matched_model_files = [e for e in matched_model_files if e is not None]
        for matched_model in matched_model_files:
            if self.current_round - int(matched_model.group(1)) >= 5:
                os.remove(os.path.join(self.model_path, matched_model.group(0)))

    def save_result_to_json(self):
        m, s = divmod(int(round(time.time())) - self.training_start_time, 60)
        h, m = divmod(m, 60)
        avg_time_records = []
        keys = ['update_send', 'update_run', 'update_receive', 'agg_server',
                'eval_send', 'eval_run', 'eval_receive', 'server_eval']
        for key in keys:
            avg_time_records.append(np.mean([e.get(key, 0) for e in self.time_record]))
        self.result_json = {
            'best_metric': self.best_test_metric,
            'best_metric_full': self.best_test_metric_full,
            'total_time': '{}:{}:{}'.format(h, m, s),
            'time_detail': str(avg_time_records),
            'total_rounds': self.current_round,
            'server_send': self.server_send_bytes / (2 ** 30),
            'server_receive': self.server_receive_bytes / (2 ** 30),
            'info_each_round': self.info_each_round
        }
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(self.result_json, f)
        save_config(self.data_config, self.model_config, self.runtime_config, self.log_dir)

    @staticmethod
    def get_comm_in_and_out():
        eth0_info = psutil.net_io_counters(pernic=True).get('eth0')
        if eth0_info is None:
            return 0, 0
        else:
            bytes_recv = eth0_info.bytes_recv
            bytes_sent = eth0_info.bytes_sent
            return bytes_recv, bytes_sent

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")
            self.logger.info('%s connected' % request.sid)

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")
            self.logger.info('%s reconnected' % request.sid)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, "disconnected")
            self.logger.info('%s disconnected' % request.sid)
            for container_id in self.ready_container_sid_dict:
                if container_id == request.sid:
                    self.ready_container_sid_dict.pop(container_id)
                    offline_client_list = self.ready_container_id_dict.pop(container_id)
                    self.ready_clients = list(set(self.ready_clients) - set(offline_client_list))

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init')

        @self.socketio.on('client_ready')
        def handle_client_ready(container_clients):

            container_id = container_clients[0]
            client_ids = container_clients[1:]

            self.logger.info('Container %s, with clients %s are ready for training' % (container_id, str(client_ids)))
            
            self.ready_clients += client_ids
            self.ready_container_id_dict[container_id] = client_ids
            self.ready_container_sid_dict[container_id] = request.sid
            
            if len(self.ready_clients) >= self.num_clients and self.current_round == 0:
                print("start to federated learning.....")
                self.training_start_time = int(round(time.time()))
                self.train_next_round()
            elif len(self.ready_clients) < self.num_clients:
                print("not enough client worker running.....")
            else:
                print("current_round is not equal to 0")

        @self.socketio.on('client_update')
        def handle_client_update(data):

            if data['round_number'] == self.current_round:

                self.thread_lock.acquire()
                data['weights'] = pickle_string_to_obj(data['weights'])
                data['time_receive_update'] = time.time()
                self.c_up.append(data)
                receive_all = len(self.c_up) == self.num_clients_contacted_per_round
                self.thread_lock.release()

                if receive_all:

                    self.logger.info("Received update from all clients")

                    receive_update_time = [e['time_receive_request'] - self.time_send_train for e in self.c_up]
                    finish_update_time = [e['time_finish_update'] - e['time_start_update'] for e in self.c_up]
                    update_receive_time = [e['time_receive_update'] - e['time_finish_update'] for e in self.c_up]

                    self.time_record[-1]['update_send'] = np.mean(receive_update_time)
                    self.time_record[-1]['update_run'] = np.mean(finish_update_time)
                    self.time_record[-1]['update_receive'] = np.mean(update_receive_time)

                    # From request update, until receives all clients' update
                    self.time_agg_train_start = time.time()

                    # current train
                    client_params = [x['weights'] for x in self.c_up]
                    aggregate_weights = np.array([x['train_size'] for x in self.c_up])

                    self.current_params = self.fed_model.update_host_params(
                        client_params, aggregate_weights / np.sum(aggregate_weights)
                    )

                    self.save_weight()

                    aggr_train_loss = self.aggregate_train_loss(
                        [x['train_loss'] for x in self.c_up],
                        [x['train_size'] for x in self.c_up],
                        self.current_round
                    )
                    self.info_each_round[self.current_round]['train_loss'] = aggr_train_loss

                    self.logger.info("=== Train ===")
                    self.logger.info('Receive update result form %s clients' % len(self.c_up))
                    self.logger.info("aggr_train_loss {}".format(aggr_train_loss))

                    # Fed Aggregate : computation time
                    self.time_agg_train_end = time.time()
                    self.time_record[-1]['agg_server'] = self.time_agg_train_end - self.time_agg_train_start

                    self.info_each_round[self.current_round]['time_train_send'] = self.time_record[-1]['update_send']
                    self.info_each_round[self.current_round]['time_train_run'] = self.time_record[-1]['update_send']
                    self.info_each_round[self.current_round]['time_train_receive'] = self.time_record[-1][
                        'update_receive']
                    self.info_each_round[self.current_round]['time_train_agg'] = self.time_record[-1]['agg_server']

                    # Collect the send and received bytes
                    self.server_receive_bytes, self.server_send_bytes = self.get_comm_in_and_out()

                    if self.current_round % self.rounds_between_val == 0:
                        self.evaluate()
                    else:
                        self.train_next_round()

                    self.info_each_round[self.current_round]['round_finish_time'] = time.time()

        @self.socketio.on('client_evaluate')
        def handle_client_evaluate(data):

            if data['round_number'] == self.current_round:

                self.thread_lock.acquire()
                data['time_receive_evaluate'] = time.time()
                self.c_eval.append(data)
                if self.lazy_update and not self.STOP:
                    receive_all = len(self.c_eval) == self.num_clients_contacted_per_round
                else:
                    receive_all = len(self.c_eval) == self.num_clients
                # self.logger.info('Receive evaluate result form %s' % request.sid)
                self.thread_lock.release()

                if receive_all:
                    # sort according to the client id
                    try:
                        self.c_eval = sorted(self.c_eval, key=lambda x: int(x['cid']))
                    except TypeError as error:
                        print('Debug Mode', error)

                    self.logger.info("=== Evaluate ===")
                    self.logger.info('Receive evaluate result form %s clients' % len(self.c_eval))

                    receive_eval_time = [e['time_receive_request'] - self.time_agg_train_end for e in self.c_eval]
                    finish_eval_time = [e['time_finish_evaluate'] - e['time_start_evaluate'] for e in self.c_eval]
                    eval_receive_time = [e['time_receive_evaluate'] - e['time_finish_evaluate'] for e in self.c_eval]

                    self.logger.info(
                        'Update Run min %s max %s mean %s'
                        % (min(finish_eval_time), max(finish_eval_time), np.mean(finish_eval_time))
                    )

                    self.time_agg_eval_start = time.time()

                    avg_val_metrics = {}
                    avg_test_metrics = {}
                    full_test_metric = {}
                    for key in self.c_eval[0]['evaluate']:
                        if key == 'val_size':
                            continue
                        if key == 'test_size':
                            continue
                            # full_test_metric['test_size'] = [
                            #     float(update['evaluate']['test_size']) for update in self.c_eval]
                        if key.startswith('val_'):
                            avg_val_metrics[key] = np.average(
                                [float(update['evaluate'][key]) for update in self.c_eval],
                                weights=[float(update['evaluate']['val_size']) for update in self.c_eval]
                            )
                            self.logger.info('Val %s : %s' % (key, avg_val_metrics[key]))
                        if key.startswith('test_'):
                            full_test_metric[key] = [float(update['evaluate'][key]) for update in self.c_eval]
                            avg_test_metrics[key] = np.average(
                                full_test_metric[key],
                                weights=[float(update['evaluate']['test_size']) for update in self.c_eval]
                            )
                            self.logger.info('Test %s : %s' % (key, avg_test_metrics[key]))

                    self.info_each_round[self.current_round].update(avg_val_metrics)
                    self.info_each_round[self.current_round].update(avg_test_metrics)

                    avg_test_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    avg_val_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    self.time_agg_eval_end = time.time()
                    self.time_record[-1]['server_eval'] = self.time_agg_eval_end - self.time_agg_eval_start

                    self.time_record[-1]['eval_send'] = np.mean(receive_eval_time)
                    self.time_record[-1]['eval_run'] = np.mean(finish_eval_time)
                    self.time_record[-1]['eval_receive'] = np.mean(eval_receive_time)

                    self.avg_test_metrics.append(avg_test_metrics)
                    self.avg_val_metrics.append(avg_val_metrics)

                    current_metric = avg_val_metrics.get('val_loss')
                    self.logger.info('val loss %s' % current_metric)

                    self.info_each_round[self.current_round]['time_eval_send'] = self.time_record[-1]['eval_send']
                    self.info_each_round[self.current_round]['time_eval_run'] = self.time_record[-1]['eval_run']
                    self.info_each_round[self.current_round]['time_eval_receive'] = self.time_record[-1]['eval_receive']
                    self.info_each_round[self.current_round]['time_eval_agg'] = self.time_record[-1]['server_eval']

                    if self.STOP:
                        # Another round of testing after the training is finished
                        self.best_test_metric_full = full_test_metric
                        self.best_test_metric.update(avg_test_metrics)
                    else:
                        if self.best_val_metric is None or self.best_val_metric > current_metric:
                            self.best_val_metric = current_metric
                            self.best_round = self.current_round
                            self.invalid_tolerate = 0
                            self.best_test_metric.update(avg_test_metrics)
                            obj_to_pickle_string(self.current_params,
                                                 os.path.join(self.model_path, self.best_weight_filename))
                            self.logger.info(str(self.best_test_metric))
                            if not self.lazy_update:
                                self.best_test_metric_full = full_test_metric
                        else:
                            self.invalid_tolerate += 1

                        if self.invalid_tolerate > self.num_tolerance > 0:
                            self.logger.info("converges! starting test phase..")
                            self.STOP = True

                        if self.current_round >= self.max_num_rounds:
                            self.logger.info("get to maximum step, stop...")
                            self.STOP = True

                    # Collect the send and received bytes
                    self.server_receive_bytes, self.server_send_bytes = self.get_comm_in_and_out()

                    if self.STOP:
                        # Another round of testing after the training is finished
                        if self.lazy_update and self.best_test_metric_full is None:
                            self.evaluate(self.ready_clients, self.best_weight_filename)
                        else:
                            self.logger.info("== done ==")
                            self.logger.info("Federated training finished ... ")
                            self.logger.info("best full test metric: " +
                                             json.dumps(self.best_test_metric_full))
                            self.logger.info("best model at round {}".format(self.best_round))
                            for key in self.best_test_metric:
                                self.logger.info(
                                    "get best test {} {}".format(key, self.best_test_metric[key])
                                )
                            self.training_stop_time = int(round(time.time()))
                            # Time
                            m, s = divmod(self.training_stop_time - self.training_start_time, 60)
                            h, m = divmod(m, 60)
                            self.logger.info('Total time: {}:{}:{}'.format(h, m, s))

                            avg_time_records = []
                            keys = ['update_send', 'update_run', 'update_receive', 'agg_server',
                                    'eval_send', 'eval_run', 'eval_receive', 'server_eval']
                            for key in keys:
                                avg_time_records.append(np.mean([e.get(key, 0) for e in self.time_record]))
                            self.logger.info('Time Detail: ' + str(avg_time_records))
                            self.logger.info('Total Rounds: %s' % self.current_round)
                            self.logger.info('Server Send(GB): %s' % (self.server_send_bytes / (2 ** 30)))
                            self.logger.info('Server Receive(GB): %s' % (self.server_receive_bytes / (2 ** 30)))
                            # save data to file
                            self.result_json = {
                                'best_metric': self.best_test_metric,
                                'best_metric_full': self.best_test_metric_full,
                                'total_time': '{}:{}:{}'.format(h, m, s),
                                'time_detail': str(avg_time_records),
                                'total_rounds': self.current_round,
                                'server_send': self.server_send_bytes / (2 ** 30),
                                'server_receive': self.server_receive_bytes / (2 ** 30),
                                'info_each_round': self.info_each_round
                            }
                            with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
                                json.dump(self.result_json, f)
                            save_config(self.data_config, self.model_config, self.runtime_config, self.log_dir)
                            # Stop all the clients
                            emit('stop', broadcast=True)
                            # Call the server exit job
                            self.fed_model.host_exit_job(self)
                            # Server job finish
                            self.server_job_finish = True
                    else:
                        self.save_result_to_json()
                        self.logger.info("start to next round...")
                        self.train_next_round()

    def response(self, mode, cid):
        self.check_list.append(cid)
        # self.logger.info('Response: ' + mode + ' %s' % cid)

    def retrieval_session_information(self, selected_clients):
        send_target = {}
        for container in self.ready_container_id_dict:
            for cid in self.ready_container_id_dict[container]:
                if cid in selected_clients:
                    send_target[container] = send_target.get(container, []) + [cid]
        return send_target

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):

        selected_clients = self.fed_model.host_select_train_clients(self.ready_clients)

        self.current_round += 1

        self.info_each_round[self.current_round] = {'timestamp': time.time()}

        # Record the time
        self.time_send_train = time.time()
        self.time_record.append({'round': self.current_round})
        self.logger.info("##### Round {} #####".format(self.current_round))

        # buffers all client updates
        self.c_up = []

        # get the session information
        actual_send = self.retrieval_session_information(selected_clients)
        
        # Start the update
        data_send = {'round_number': self.current_round, 'selected_clients': None}

        for container_id, target_clients in actual_send.items():
            self.logger.info('Sending train requests to container %s targeting clients %s' % (
                container_id, str(target_clients)))
            data_send['selected_clients'] = target_clients
            emit('request_update', data_send, room=self.ready_container_sid_dict[container_id], callback=self.response)

        self.logger.info('Finished sending update requests, waiting resp from clients')

    def evaluate(self, selected_clients=None, specified_model_file=None):
        self.logger.info('Starting eval')
        self.c_eval = []
        if specified_model_file is not None and os.path.isfile(os.path.join(self.model_path, specified_model_file)):
            data_send = {'round_number': self.current_round, 'weights_file_name': specified_model_file}
        else:
            data_send = {'round_number': self.current_round}

        # retrieval send information
        if selected_clients is None:
            # selected_clients = self.fed_model.host_select_evaluate_clients(self.ready_clients)
            selected_clients = self.ready_clients
        actual_send = self.retrieval_session_information(selected_clients)

        self.logger.info('Sending eval requests to %s clients' % len(selected_clients))
        for container_id, target_clients in actual_send.items():
            data_send['selected_clients'] = target_clients
            emit('request_evaluate', data_send, room=self.ready_container_sid_dict[container_id], callback=self.response)
        self.logger.info('Waiting resp from clients')

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)
