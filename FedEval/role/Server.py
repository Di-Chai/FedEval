import datetime
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from flask import render_template, request, send_file

from ..run_util import save_config
from ..utils import obj_to_pickle_string, pickle_string_to_obj
from .container import ContainerId
from .flask_node import ClientSocketIOEvent, FlaskNode, Sid
from .model_weights_io import server_best_weight_filename, weights_filename_pattern
from .role import Role
from .service_interface import ServerFlaskInterface


class Server(FlaskNode):
    """a central server implementation based on FlaskNode."""

    def __init__(self, data_config, model_config, runtime_config):
        super().__init__('server', data_config, model_config, runtime_config, role=Role.Server)
        self._init_states()
        self._init_logger()
        self._register_handles()
        self._register_services()
        self.logger.info(self._run_config)
        self.logger.info(self.get_model_description())
        self.save_weight(weights_filename_pattern)   # save the init weights

    def _init_logger(self):
        super()._init_logger('Server', 'Server')

    def _bind_run_configs(self):
        self._run_config = self.fed_model.param_parser.parse_run_config()
        self._num_clients = self._run_config['num_clients']
        self._max_num_rounds = self._run_config['max_num_rounds']
        self._num_tolerance = self._run_config['num_tolerance']
        self._num_clients_contacted_per_round = self._run_config['num_clients_contacted_per_round']
        self._rounds_between_val = self._run_config['rounds_between_val']

    def _init_metric_states(self):
        # weights should be an ordered list of parameter
        # for stats
        self._avg_val_metrics: List = []
        self._avg_test_metrics: List = []

        # for convergence check
        self._best_val_metric = None
        self._best_test_metric = {}
        self._best_test_metric_full = None
        self._best_weight = None
        self._best_round = -1

    def _init_statistical_states(self):
        """initialize statistics."""
        # time & moments
        self._time_send_train: Optional[float] = None
        self._time_agg_train_start: Optional[float] = None
        self._time_agg_train_end: Optional[float] = None
        self._time_agg_eval_start: Optional[float] = None
        self._time_agg_eval_end: Optional[float] = None
        self._time_record: List[Dict[str, Any]] = []
        self._training_start_time: int = int(time.time())    # seconds
        self._training_stop_time: Optional[int] = None       # seconds

        # network traffic
        self._server_send_bytes: int = 0
        self._server_receive_bytes: int = 0

        # rounds during training
        self._current_round: int = 0
        self._c_up = []                                      # clients' updates of this round
        self._c_eval = []                                    # clients' evaluations of this round
        self._check_list = []
        self._info_each_round = {}

    def _init_control_states(self):
        """initilize attributes for controling."""
        self._thread_lock = threading.Lock()
        self._STOP = False
        self._server_job_finish = False
        self._client_sids_selected: Optional[List[Any]] = None
        self._invalid_tolerate: int = 0  # for client-side evaluation
        self._ready_container_sid_dict: Dict[ContainerId, Sid] = {}
        self._ready_container_id_dict: Dict[ContainerId, List[ContainerId]] = {}
        self._ready_clients: List[ContainerId] = []
        self._client_resource = {}

    def _init_model_io_configs(self):
        self._model_path = os.path.abspath(self.log_dir) # TODO(fgh): seperate model_path from log_dir

    def _init_states(self):
        self._bind_run_configs()
        self._lazy_update = False

        self._init_statistical_states()
        self._init_control_states()
        self._current_params = self.fed_model.host_get_init_params()
        self._init_metric_states()
        self._init_model_io_configs()

    def _register_services(self):
        @self.route(ServerFlaskInterface.Dashboard.value)
        def dashboard():
            """for performance illustration and monitoring.

            Returns:
                the rendered dashboard web page.
            """

            avg_test_metric = self._avg_test_metrics[0] if len(self._avg_test_metrics) > 0 else {}
            avg_test_metric_keys = [e for e in avg_test_metric.keys() if e != 'time']

            avg_val_metric = self._avg_val_metrics[0] if len(self._avg_val_metrics) > 0 else {}
            avg_val_metric_keys = [e for e in avg_val_metric.keys() if e != 'time']

            time_record = [e for e in self._time_record if len(e.keys()) >= 6]
            if len(time_record) > 0:
                time_record.append({'round': 'Average'})
                for key in time_record[0]:
                    if key not in ['round', 'eval_receive_time']:
                        time_record[-1][key] = np.mean([e[key] for e in time_record[:-1]])

                time_record = time_record[-6:]
                # time_record = [time_record[i] for i in range(len(time_record)) if (len(time_record) - 6) <= i]

            train_stop_time = self._training_stop_time if self._STOP and self._training_stop_time is not None else round(time.time())
            current_used_time = int(train_stop_time - self._training_start_time)
            m, s = divmod(current_used_time, 60)
            h, m = divmod(m, 60)

            metrics = self.fed_model.model_config['MLModel'].get('metrics', [])
            test_accuracy_key = 'accuracy' if len(metrics) == 0 else metrics[0]
            test_accuracy_key = 'test_' + test_accuracy_key

            return render_template(
                'dashboard.html',
                status='Finish' if self._STOP else 'Running',
                rounds="%s / %s" % (self._current_round, self._max_num_rounds),
                num_online_clients="%s / %s / %s" % (self._num_clients_contacted_per_round,
                                                     len(self._ready_clients), self._num_clients),
                avg_test_metric=self._avg_test_metrics,
                avg_test_metric_keys=avg_test_metric_keys,
                avg_val_metric=self._avg_val_metrics,
                avg_val_metric_keys=avg_val_metric_keys,
                time_record=time_record,
                current_used_time="%02d:%02d:%02d" % (h, m, s),
                test_accuracy=self._best_test_metric.get(test_accuracy_key, 0),
                test_loss=self._best_test_metric.get('test_loss', 0),
                server_send=self._server_send_bytes / (2**30),
                server_receive=self._server_receive_bytes / (2**30),
            )

        # TMP use
        @self.route(ServerFlaskInterface.Status.value)
        def status_page():
            return json.dumps({
                'finished': self._server_job_finish,
                'rounds': self._current_round,
                'results': [
                    None if len(self._avg_val_metrics) == 0 else self._avg_val_metrics[-1], 
                    None if len(self._avg_test_metrics) == 0 else self._avg_test_metrics[-1]
                    ],
                'log_dir': self.log_dir,
            })

        @self.route(ServerFlaskInterface.DownloadPattern.value.format('<filename>'), methods=['GET'])
        def download_file(filename):
            if os.path.isfile(os.path.join(self._model_path, filename)):
                return send_file(os.path.join(self._model_path, filename), as_attachment=True)
            else:
                return json.dumps({'status': 404, 'msg': 'file not found'})

    # cur_round could None
    def aggregate_train_loss(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time()) - self._training_start_time)
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

    def save_weight(self, weight_filename_pattern):
        obj_to_pickle_string(
            self._current_params,
            os.path.join(self._model_path, weight_filename_pattern.format(self._current_round))
        )
        # Keep the latest 5 weights
        all_files_in_model_dir = os.listdir(self._model_path)
        matched_model_files = [re.match(r'model_([0-9]+).pkl', e) for e in all_files_in_model_dir]
        matched_model_files = [e for e in matched_model_files if e is not None]
        for matched_model in matched_model_files:
            if self._current_round - int(matched_model.group(1)) >= 5:
                os.remove(os.path.join(self._model_path, matched_model.group(0)))

    def save_result_to_json(self):
        m, s = divmod(int(round(time.time()) - self._training_start_time), 60)
        h, m = divmod(m, 60)
        avg_time_records = []
        keys = ['update_send', 'update_run', 'update_receive', 'agg_server',
                'eval_send', 'eval_run', 'eval_receive', 'server_eval']
        for key in keys:
            avg_time_records.append(np.mean([e.get(key, 0) for e in self._time_record]))
        self.result_json = {
            'best_metric': self._best_test_metric,
            'best_metric_full': self._best_test_metric_full,
            'total_time': '{}:{}:{}'.format(h, m, s),
            'time_detail': str(avg_time_records),
            'total_rounds': self._current_round,
            'server_send': self._server_send_bytes / (2 ** 30),
            'server_receive': self._server_receive_bytes / (2 ** 30),
            'info_each_round': self._info_each_round
        }
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(self.result_json, f)
        save_config(self.data_config, self.model_config, self.runtime_config, self.log_dir)

    def _register_handles(self):
        from . import ServerSocketIOEvent

        # single-threaded async, no need to lock

        @self.on(ServerSocketIOEvent.Connect)
        def handle_connect():
            print(request.sid, "connected")
            self.logger.info('%s connected' % request.sid)

        @self.on(ServerSocketIOEvent.Reconnect)
        def handle_reconnect():
            print(request.sid, "reconnected")
            self.logger.info('%s reconnected' % request.sid)

        @self.on(ServerSocketIOEvent.Disconnect)
        def handle_disconnect():
            print(request.sid, "disconnected")
            self.logger.info('%s disconnected' % request.sid)
            if request.sid in self._ready_container_sid_dict:
                self._ready_container_sid_dict.pop(request.sid)
                offline_clients: List[ContainerId] = self._ready_container_id_dict.pop(request.sid)
                ready_clients = set(self._ready_clients) - set(offline_clients)
                self._ready_clients = list(ready_clients)

        @self.on(ServerSocketIOEvent.WakeUp)
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            self.invoke(ClientSocketIOEvent.Init)

        @self.on(ServerSocketIOEvent.Ready)
        def handle_client_ready(container_clients):

            container_id = container_clients[0]
            client_ids = container_clients[1:]

            self.logger.info('Container %s, with clients %s are ready for training' % (container_id, str(client_ids)))
            
            self._ready_clients += client_ids
            self._ready_container_id_dict[container_id] = client_ids
            self._ready_container_sid_dict[container_id] = request.sid

            if len(self._ready_clients) >= self._num_clients and self._current_round == 0:
                print("start to federated learning.....")
                self._training_start_time = int(round(time.time()))
                self.train_next_round()
            elif len(self._ready_clients) < self._num_clients:
                print("not enough client worker running.....")
            else:
                print("current_round is not equal to 0")

        @self.on(ServerSocketIOEvent.ResponseUpdate)
        def handle_client_update(data: Mapping[str, Any]):

            if data['round_number'] == self._current_round:

                with self._thread_lock:
                    data['weights'] = pickle_string_to_obj(data['weights'])
                    data['time_receive_update'] = time.time()
                    self._c_up.append(data)
                    receive_all = len(self._c_up) == self._num_clients_contacted_per_round

                if receive_all:

                    self.logger.info("Received update from all clients")

                    receive_update_time = [e['time_receive_request'] - self._time_send_train for e in self._c_up]
                    finish_update_time = [e['time_finish_update'] - e['time_start_update'] for e in self._c_up]
                    update_receive_time = [e['time_receive_update'] - e['time_finish_update'] for e in self._c_up]
                    latest_time_record = self._time_record[-1]
                    cur_round_info = self._info_each_round[self._current_round]

                    latest_time_record['update_send'] = np.mean(receive_update_time)
                    latest_time_record['update_run'] = np.mean(finish_update_time)
                    latest_time_record['update_receive'] = np.mean(update_receive_time)

                    # From request update, until receives all clients' update
                    self._time_agg_train_start = time.time()

                    # current train
                    client_params = [x['weights'] for x in self._c_up]
                    aggregate_weights = np.array([x['train_size'] for x in self._c_up])

                    self._current_params = self.fed_model.update_host_params(
                        client_params, aggregate_weights / np.sum(aggregate_weights)
                    )

                    self.save_weight(weights_filename_pattern)

                    aggr_train_loss = self.aggregate_train_loss(
                        [x['train_loss'] for x in self._c_up],
                        [x['train_size'] for x in self._c_up],
                        self._current_round
                    )
                    cur_round_info['train_loss'] = aggr_train_loss

                    self.logger.info("=== Train ===")
                    self.logger.info('Receive update result form %s clients' % len(self._c_up))
                    self.logger.info("aggr_train_loss {}".format(aggr_train_loss))

                    # Fed Aggregate : computation time
                    self._time_agg_train_end = time.time()
                    latest_time_record['agg_server'] = self._time_agg_train_end - self._time_agg_train_start

                    cur_round_info['time_train_send'] = latest_time_record['update_send']
                    cur_round_info['time_train_run'] = latest_time_record['update_send']
                    cur_round_info['time_train_receive'] = latest_time_record['update_receive']
                    cur_round_info['time_train_agg'] = latest_time_record['agg_server']

                    # Collect the send and received bytes
                    self._server_receive_bytes, self._server_send_bytes = self._get_comm_in_and_out()

                    if self._current_round % self._rounds_between_val == 0:
                        self.evaluate()
                    else:
                        self.train_next_round()

                    cur_round_info['round_finish_time'] = time.time()

        @self.on(ServerSocketIOEvent.ResponseEvaluate)
        def handle_client_evaluate(data: Mapping[str, Any]):

            if data['round_number'] == self._current_round:

                with self._thread_lock:
                    data['time_receive_evaluate'] = time.time()
                    self._c_eval.append(data)
                    num_clients_required = self._num_clients_contacted_per_round if self._lazy_update and not self._STOP else self._num_clients
                    receive_all = len(self._c_eval) == num_clients_required
                # self.logger.info('Receive evaluate result form %s' % request.sid)

                if receive_all:
                    # sort according to the client id
                    try:
                        self._c_eval = sorted(self._c_eval, key=lambda x: int(x['cid']))
                    except TypeError as error:
                        print('Debug Mode', error)

                    self.logger.info("=== Evaluate ===")
                    self.logger.info('Receive evaluate result form %s clients' % len(self._c_eval))

                    receive_eval_time = [e['time_receive_request'] - self._time_agg_train_end for e in self._c_eval]
                    finish_eval_time = [e['time_finish_evaluate'] - e['time_start_evaluate'] for e in self._c_eval]
                    eval_receive_time = [e['time_receive_evaluate'] - e['time_finish_evaluate'] for e in self._c_eval]

                    self.logger.info(
                        'Update Run min %s max %s mean %s'
                        % (min(finish_eval_time), max(finish_eval_time), np.mean(finish_eval_time))
                    )

                    self._time_agg_eval_start = time.time()

                    avg_val_metrics = {}
                    avg_test_metrics = {}
                    full_test_metric = {}
                    for key in self._c_eval[0]['evaluate']:
                        if key == 'val_size':
                            continue
                        if key == 'test_size':
                            continue
                            # full_test_metric['test_size'] = [
                            #     float(update['evaluate']['test_size']) for update in self.c_eval]
                        if key.startswith('val_'):
                            avg_val_metrics[key] = np.average(
                                [float(update['evaluate'][key]) for update in self._c_eval],
                                weights=[float(update['evaluate']['val_size']) for update in self._c_eval]
                            )
                            self.logger.info('Val %s : %s' % (key, avg_val_metrics[key]))
                        if key.startswith('test_'):
                            full_test_metric[key] = [float(update['evaluate'][key]) for update in self._c_eval]
                            avg_test_metrics[key] = np.average(
                                full_test_metric[key],
                                weights=[float(update['evaluate']['test_size']) for update in self._c_eval]
                            )
                            self.logger.info('Test %s : %s' % (key, avg_test_metrics[key]))

                    latest_time_record = self._time_record[-1]
                    cur_round_info = self._info_each_round[self._current_round]
                    cur_round_info.update(avg_val_metrics)
                    cur_round_info.update(avg_test_metrics)

                    avg_test_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    avg_val_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    self._time_agg_eval_end = time.time()
                    latest_time_record['server_eval'] = self._time_agg_eval_end - self._time_agg_eval_start

                    latest_time_record['eval_send'] = np.mean(receive_eval_time)
                    latest_time_record['eval_run'] = np.mean(finish_eval_time)
                    latest_time_record['eval_receive'] = np.mean(eval_receive_time)

                    self._avg_test_metrics.append(avg_test_metrics)
                    self._avg_val_metrics.append(avg_val_metrics)

                    current_metric = avg_val_metrics.get('val_loss')
                    self.logger.info('val loss %s' % current_metric)

                    cur_round_info['time_eval_send'] = latest_time_record['eval_send']
                    cur_round_info['time_eval_run'] = latest_time_record['eval_run']
                    cur_round_info['time_eval_receive'] = latest_time_record['eval_receive']
                    cur_round_info['time_eval_agg'] = latest_time_record['server_eval']

                    if self._STOP:
                        # Another round of testing after the training is finished
                        self._best_test_metric_full = full_test_metric
                        self._best_test_metric.update(avg_test_metrics)
                    else:
                        if self._best_val_metric is None or self._best_val_metric > current_metric:
                            self._best_val_metric = current_metric
                            self._best_round = self._current_round
                            self._invalid_tolerate = 0
                            self._best_test_metric.update(avg_test_metrics)
                            obj_to_pickle_string(self._current_params,
                                                 os.path.join(self._model_path, server_best_weight_filename))
                            self.logger.info(str(self._best_test_metric))
                            if not self._lazy_update:
                                self._best_test_metric_full = full_test_metric
                        else:
                            self._invalid_tolerate += 1

                        if self._invalid_tolerate > self._num_tolerance:
                            self.logger.info("converges! starting test phase..")
                            self._STOP = True

                        if self._current_round >= self._max_num_rounds:
                            self.logger.info("get to maximum step, stop...")
                            self._STOP = True

                    # Collect the send and received bytes
                    self._server_receive_bytes, self._server_send_bytes = self._get_comm_in_and_out()

                    if self._STOP:
                        # Another round of testing after the training is finished
                        if self._lazy_update and self._best_test_metric_full is None:
                            self.evaluate(self._ready_clients, server_best_weight_filename)
                        else:
                            self.logger.info("== done ==")
                            self.logger.info("Federated training finished ... ")
                            self.logger.info("best full test metric: " +
                                             json.dumps(self._best_test_metric_full))
                            self.logger.info("best model at round {}".format(self._best_round))
                            for key in self._best_test_metric:
                                self.logger.info(
                                    "get best test {} {}".format(key, self._best_test_metric[key])
                                )
                            self._training_stop_time = int(round(time.time()))
                            # Time
                            m, s = divmod(self._training_stop_time - self._training_start_time, 60)
                            h, m = divmod(m, 60)
                            self.logger.info('Total time: {}:{}:{}'.format(h, m, s))

                            avg_time_records = []
                            keys = ['update_send', 'update_run', 'update_receive', 'agg_server',
                                    'eval_send', 'eval_run', 'eval_receive', 'server_eval']
                            for key in keys:
                                avg_time_records.append(np.mean([e.get(key, 0) for e in self._time_record]))
                            self.logger.info('Time Detail: ' + str(avg_time_records))
                            self.logger.info('Total Rounds: %s' % self._current_round)
                            self.logger.info('Server Send(GB): %s' % (self._server_send_bytes / (2 ** 30)))
                            self.logger.info('Server Receive(GB): %s' % (self._server_receive_bytes / (2 ** 30)))
                            # save data to file
                            self.result_json = {
                                'best_metric': self._best_test_metric,
                                'best_metric_full': self._best_test_metric_full,
                                'total_time': '{}:{}:{}'.format(h, m, s),
                                'time_detail': str(avg_time_records),
                                'total_rounds': self._current_round,
                                'server_send': self._server_send_bytes / (2 ** 30),
                                'server_receive': self._server_receive_bytes / (2 ** 30),
                                'info_each_round': self._info_each_round
                            }
                            with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
                                json.dump(self.result_json, f)
                            save_config(self.data_config, self.model_config, self.runtime_config, self.log_dir)
                            # Stop all the clients
                            self.invoke(ClientSocketIOEvent.Stop, broadcast=True)
                            # Call the server exit job
                            self.fed_model.host_exit_job(self)
                            # Server job finish
                            self._server_job_finish = True
                    else:
                        self.save_result_to_json()
                        self.logger.info("start to next round...")
                        self.train_next_round()

    def response(self, mode, cid):
        self._check_list.append(cid)
        # self.logger.info('Response: ' + mode + ' %s' % cid)

    def retrieval_session_information(self, selected_clients):
        send_target = {}
        for container in self._ready_container_id_dict:
            for cid in self._ready_container_id_dict[container]:
                if cid in selected_clients:
                    send_target[container] = send_target.get(container, []) + [cid]
        return send_target

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):

        selected_clients = self.fed_model.host_select_train_clients(self._ready_clients)
        self._current_round += 1
        self._info_each_round[self._current_round] = {'timestamp': time.time()}

        # Record the time
        self._time_send_train = time.time()
        self._time_record.append({'round': self._current_round})
        self.logger.info("##### Round {} #####".format(self._current_round))

        # buffers all client updates
        self._c_up = []

        # get the session information
        actual_send = self.retrieval_session_information(selected_clients)

        # Start the update
        data_send = {'round_number': self._current_round, 'selected_clients': None}

        for container_id, target_clients in actual_send.items():
            self.logger.info('Sending train requests to container %s targeting clients %s' % (
                container_id, str(target_clients)))
            data_send['selected_clients'] = target_clients
            self.invoke(ClientSocketIOEvent.RequestUpdate, data_send, room=self._ready_container_sid_dict[container_id], callback=self.response)

        self.logger.info('Finished sending update requests, waiting resp from clients')

    def evaluate(self, selected_clients=None, specified_model_file=None):
        self.logger.info('Starting eval')
        self._c_eval = []
        data_send = { 'round_number': self._current_round }
        if specified_model_file is not None and os.path.isfile(os.path.join(self._model_path, specified_model_file)):
            data_send['weights_file_name'] = specified_model_file

        # retrieval send information
        if selected_clients is None:
            # selected_clients = self.fed_model.host_select_evaluate_clients(self._ready_clients)
            selected_clients = self._ready_clients
        actual_send = self.retrieval_session_information(selected_clients)

        self.logger.info('Sending eval requests to %s clients' % len(selected_clients))
        for container_id, target_clients in actual_send.items():
            data_send['selected_clients'] = target_clients
            self.invoke(ClientSocketIOEvent.RequestEvaluate, data_send,
                        room=self._ready_container_sid_dict[container_id], callback=self.response)
        self.logger.info('Waiting resp from clients')

    def start(self):
        # start to provide services
        self._run_server()
