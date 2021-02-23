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


class Aggregator(object):

    def __init__(self, model, logger, fed_model_name, train_strategy, upload_strategy):

        fed_model = parse_strategy_name(fed_model_name)

        self.fed_model = fed_model(
            role='server', model=model, upload_strategy=upload_strategy,
            train_strategy=train_strategy,
        )

        self.logger = logger
        self.logger.info(self.get_model_description())

        self.current_params = self.fed_model.host_get_init_params()
        self.model_path = os.path.join(self.fed_model.model.model_dir, self.fed_model.model.code_version + '.pkl')

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

        self.training_start_time = int(round(time.time()))
        self.training_stop_time = None

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


class Server(object):
    def __init__(self, server_config, model, train_strategy, upload_strategy, fed_model_name):
        self.server_config = server_config
        self.ready_client_sids = set()
        
        self.host = self.server_config['listen']
        self.port = self.server_config['port']
        self.client_resource = {}

        self.num_clients = self.server_config["num_clients"]
        self.max_num_rounds = train_strategy["max_num_rounds"]
        self.num_tolerance = train_strategy["num_tolerance"]
        self.num_clients_contacted_per_round = int(self.num_clients * train_strategy['C'])
        print(self.num_clients_contacted_per_round)
        self.rounds_between_val = train_strategy["rounds_between_val"]
        self.lazy_update = True if train_strategy['lazy_update'] == 'True' else False

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())

        self.logger = logging.getLogger("Server")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(model.model_dir, "Server", time_str)
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

        self.time_check_res = None
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

        self.wait_time = 0
        self.logger.info(self.server_config)

        self.aggregator = Aggregator(model, self.logger,
                                     fed_model_name=fed_model_name,
                                     train_strategy=train_strategy,
                                     upload_strategy=upload_strategy)

        self.model_path = os.path.abspath(self.log_dir)
        self.weight_filename = 'model_{}.pkl'
        self.best_weight_filename = 'best_model.pkl'

        #####
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
                                 ping_timeout=3600, ping_interval=1800)

        # socket io messages
        self.register_handles()
        self.invalid_tolerate = 0

        self.client_sids_selected = None

        @self.app.route('/dashboard')
        def dashboard():

            if len(self.aggregator.avg_test_metrics) > 0:
                avg_test_metric_keys = [e for e in list(self.aggregator.avg_test_metrics[0].keys()) if e != 'time']
            else:
                avg_test_metric_keys = []

            if len(self.aggregator.avg_val_metrics) > 0:
                avg_val_metric_keys = [e for e in list(self.aggregator.avg_val_metrics[0].keys()) if e != 'time']
            else:
                avg_val_metric_keys = []

            time_record = [e for e in self.time_record if len(e.keys()) >= 6]
            if len(time_record) > 0:
                time_record.append({'round': 'Average'})
                for key in time_record[0]:
                    if key not in ['round', 'eval_receive_time']:
                        time_record[-1][key] = np.mean([e[key] for e in time_record[:-1]])

            time_record = [time_record[i] for i in range(len(time_record)) if (len(time_record) - i) <= 6]

            if self.STOP and self.aggregator.training_stop_time is not None:
                current_used_time = self.aggregator.training_stop_time - self.aggregator.training_start_time
            else:
                current_used_time = int(round(time.time())) - self.aggregator.training_start_time
            m, s = divmod(current_used_time, 60)
            h, m = divmod(m, 60)

            return render_template(
                'dashboard.html',
                status='Finish' if self.STOP else 'Running',
                rounds="%s / %s" % (self.current_round, self.max_num_rounds),
                num_online_clients="%s / %s / %s" % (self.num_clients_contacted_per_round,
                                                     len(self.ready_client_sids), self.num_clients),
                avg_test_metric=self.aggregator.avg_test_metrics,
                avg_test_metric_keys=avg_test_metric_keys,
                avg_val_metric=self.aggregator.avg_val_metrics,
                avg_val_metric_keys=avg_val_metric_keys,
                time_record=time_record,
                current_used_time="%02d:%02d:%02d" % (h, m, s),
                test_accuracy=self.aggregator.best_test_metric.get('test_accuracy', 0),
                test_loss=self.aggregator.best_test_metric.get('test_loss', 0),
                server_send=self.server_send_bytes / (2 ** 30),
                server_receive=self.server_receive_bytes/(2**30)
            )

        # TMP use
        @self.app.route('/status')
        def status_page():
            return json.dumps({
                'status': self.server_job_finish,
                'rounds': self.current_round,
                'log_dir': self.log_dir,
            })

        @self.app.route("/download/<filename>", methods=['GET'])
        def download_file(filename):
            if os.path.isfile(os.path.join(self.model_path, filename)):
                return send_file(os.path.join(self.model_path, filename), as_attachment=True)
            else:
                return json.dumps({'status': 404, 'msg': 'file not found'})

    def save_weight(self):
        obj_to_pickle_string(
            self.aggregator.current_params,
            os.path.join(self.model_path, self.weight_filename.format(self.current_round))
        )
        # Keep the latest 5 weights
        all_files_in_model_dir = os.listdir(self.model_path)
        matched_model_files = [re.match(r'model_([0-9]+).pkl', e) for e in all_files_in_model_dir]
        matched_model_files = [e for e in matched_model_files if e is not None]
        for matched_model in matched_model_files:
            if self.current_round - int(matched_model.group(1)) >= 5:
                os.remove(os.path.join(self.model_path, matched_model.group(0)))

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
            # self.logger.info('%s connected' % request.sid)

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")
            # self.logger.info('%s reconnected' % request.sid)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, "disconnected")
            # self.logger.info('%s disconnected' % request.sid)
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init')

        @self.socketio.on('client_ready')
        def handle_client_ready():
            print("client ready for training", request.sid)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= self.num_clients and self.current_round == 0:
                print("start to federated learning.....")
                self.aggregator.training_start_time = int(round(time.time()))
                self.check_client_resource()
            elif len(self.ready_client_sids) < self.num_clients:
                print("not enough client worker running.....")
            else:
                print("current_round is not equal to 0")

        @self.socketio.on('check_client_resource_done')
        def handle_check_client_resource_done(data):
            # self.logger.info('Check Res done')
            if data['round_number'] == self.current_round:
                self.thread_lock.acquire()
                self.client_resource[request.sid] = data['load_rate']
                res_check = len(self.client_resource) == self.num_clients_contacted_per_round
                self.thread_lock.release()
                if res_check:
                    satisfy = 0
                    client_sids_selected = []
                    for client_id, val in self.client_resource.items():
                        # self.logger.info(str(client_id) + "cpu rate: " + str(val))
                        if float(val) < 0.4:
                            client_sids_selected.append(client_id)
                            satisfy = satisfy + 1
                    if satisfy == self.num_clients_contacted_per_round:
                        self.train_next_round(client_sids_selected)
                    else:
                        self.check_client_resource()

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

                    receive_update_time = [e['time_receive_request'] - self.time_send_train for e in self.c_up]
                    finish_update_time = [e['time_finish_update'] - e['time_receive_request'] for e in self.c_up]
                    update_receive_time = [e['time_receive_update'] - e['time_finish_update'] for e in self.c_up]

                    self.time_record[-1]['update_send'] = np.mean(receive_update_time)
                    self.time_record[-1]['update_run'] = np.mean(finish_update_time)
                    self.time_record[-1]['update_receive'] = np.mean(update_receive_time)

                    # From request update, until receives all clients' update
                    self.time_agg_train_start = time.time()

                    # current train
                    client_params = [x['weights'] for x in self.c_up]
                    aggregate_weights = np.array([x['train_size'] for x in self.c_up])

                    self.aggregator.current_params = self.aggregator.fed_model.update_host_params(
                        client_params, aggregate_weights / np.sum(aggregate_weights)
                    )

                    self.save_weight()

                    aggr_train_loss = self.aggregator.aggregate_train_loss(
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

                    # Prepare to the next round or evaluate
                    self.client_sids_selected =\
                        random.sample(list(self.ready_client_sids), self.num_clients_contacted_per_round)

                    if self.current_round % self.rounds_between_val == 0:
                        # Evaluate on the selected or all the clients
                        if self.lazy_update:
                            self.evaluate(self.client_sids_selected)
                        else:
                            self.evaluate(self.ready_client_sids)
                    else:
                        self.check_client_resource()

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
                    self.c_eval = sorted(self.c_eval, key=lambda x: int(x['cid']))

                    self.logger.info("=== Evaluate ===")
                    self.logger.info('Receive evaluate result form %s clients' % len(self.c_eval))

                    receive_eval_time = [e['time_receive_request'] - self.time_agg_train_end for e in self.c_eval]
                    finish_eval_time = [e['time_finish_update'] - e['time_receive_request'] for e in self.c_eval]
                    eval_receive_time = [e['time_receive_evaluate'] - e['time_finish_update'] for e in self.c_eval]

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
                            full_test_metric['test_size'] = [
                                float(update['evaluate']['test_size']) for update in self.c_eval]
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

                    self.aggregator.avg_test_metrics.append(avg_test_metrics)
                    self.aggregator.avg_val_metrics.append(avg_val_metrics)

                    current_metric = avg_val_metrics.get('val_default')
                    self.logger.info('val default %s' % current_metric)

                    self.info_each_round[self.current_round]['time_eval_send'] = self.time_record[-1]['eval_send']
                    self.info_each_round[self.current_round]['time_eval_run'] = self.time_record[-1]['eval_run']
                    self.info_each_round[self.current_round]['time_eval_receive'] = self.time_record[-1]['eval_receive']
                    self.info_each_round[self.current_round]['time_eval_agg'] = self.time_record[-1]['server_eval']

                    if self.STOP:
                        # Another round of testing after the training is finished
                        self.aggregator.best_test_metric_full = full_test_metric
                        self.aggregator.best_test_metric.update(avg_test_metrics)
                    else:
                        if self.aggregator.best_val_metric is None or self.aggregator.best_val_metric > current_metric:
                            self.aggregator.best_val_metric = current_metric
                            self.aggregator.best_round = self.current_round
                            self.invalid_tolerate = 0
                            self.aggregator.best_test_metric.update(avg_test_metrics)
                            obj_to_pickle_string(self.aggregator.current_params,
                                                 os.path.join(self.model_path, self.best_weight_filename))
                            if not self.lazy_update:
                                self.aggregator.best_test_metric_full = full_test_metric
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
                        if self.lazy_update and self.aggregator.best_test_metric_full is None:
                            self.evaluate(self.ready_client_sids, self.best_weight_filename)
                        else:
                            self.logger.info("== done ==")
                            self.logger.info("Federated training finished ... ")
                            self.logger.info("best full test metric: " +
                                             json.dumps(self.aggregator.best_test_metric_full))
                            self.logger.info("best model at round {}".format(self.aggregator.best_round))
                            for key in self.aggregator.best_test_metric:
                                self.logger.info(
                                    "get best test {} {}".format(key, self.aggregator.best_test_metric[key])
                                )
                            self.aggregator.training_stop_time = int(round(time.time()))
                            # Time
                            m, s = divmod(self.aggregator.training_stop_time - self.aggregator.training_start_time, 60)
                            h, m = divmod(m, 60)
                            self.logger.info('Total time: {}:{}:{}'.format(h, m, s))

                            avg_time_records = []
                            keys = ['check_res', 'update_send', 'update_run', 'update_receive', 'agg_server',
                                    'eval_send', 'eval_run', 'eval_receive', 'server_eval']
                            for key in keys:
                                avg_time_records.append(np.mean([e.get(key, 0) for e in self.time_record]))
                            self.logger.info('Time Detail: ' + str(avg_time_records))
                            self.logger.info('Total Rounds: %s' % self.current_round)
                            self.logger.info('Server Send(GB): %s' % (self.server_send_bytes / (2 ** 30)))
                            self.logger.info('Server Receive(GB): %s' % (self.server_receive_bytes / (2 ** 30)))
                            # save data to file
                            result_json = {
                                'best_metric': self.aggregator.best_test_metric,
                                'best_metric_full': self.aggregator.best_test_metric_full,
                                'total_time': '{}:{}:{}'.format(h, m, s),
                                'time_detail': str(avg_time_records),
                                'total_rounds': self.current_round,
                                'server_send': self.server_send_bytes / (2 ** 30),
                                'server_receive': self.server_receive_bytes / (2 ** 30),
                                'info_each_round': self.info_each_round
                            }
                            with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
                                json.dump(result_json, f)
                            # Server job finish
                            self.server_job_finish = True
                            # Stop all the clients
                            emit('stop', broadcast=True)
                    else:
                        self.logger.info("start to next round...")
                        self.check_client_resource()

    def check_client_resource(self):
        self.time_check_res = time.time()
        self.client_resource = {}
        self.check_list = []
        if self.client_sids_selected is None:
            self.client_sids_selected = \
                random.sample(list(self.ready_client_sids), self.num_clients_contacted_per_round)
        for rid in self.client_sids_selected:
            emit('check_client_resource', {
                'round_number': self.current_round,
                'rid': rid
            }, room=rid, callback=self.response)

    def response(self, mode, cid):
        self.check_list.append(cid)
        # self.logger.info('Response: ' + mode + ' %s' % cid)

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self, client_sids_selected):

        self.current_round += 1

        self.info_each_round[self.current_round] = {}

        # Record the time
        self.time_send_train = time.time()
        self.time_record.append({'round': self.current_round})
        self.time_record[-1]['check_res'] = self.time_send_train - self.time_check_res
        self.logger.info("##### Round {} #####".format(self.current_round))

        self.info_each_round[self.current_round]['time_init'] = self.time_send_train - self.time_check_res

        # buffers all client updates
        self.c_up = []
        
        # Start the update
        data_send = {'round_number': self.current_round}

        self.logger.info('Sending train requests to %s clients' % len(client_sids_selected))
        for rid in client_sids_selected:
            emit('request_update', data_send, room=rid, callback=self.response)
        self.logger.info('Waiting resp from clients')

    def evaluate(self, client_sids_selected, specified_model_file=None):
        self.logger.info('Starting eval')
        self.c_eval = []
        if specified_model_file is not None and os.path.isfile(os.path.join(self.model_path, specified_model_file)):
            data_send = {'round_number': self.current_round, 'weights_file_name': specified_model_file}
        else:
            data_send = {'round_number': self.current_round}

        self.logger.info('Sending eval requests to %s clients' % len(self.ready_client_sids))
        # TODO: lazy update
        # for c_sid in self.ready_client_sids:
        for rid in client_sids_selected:
            emit('request_evaluate', data_send, room=rid, callback=self.response)
        self.logger.info('Waiting resp from clients')

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)
