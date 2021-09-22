import base64
import datetime
import json
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from flask import render_template, send_file

from ..communicaiton import ServerFlaskCommunicator
from ..communicaiton.events import *
from ..config import ClientId, ConfigurationManager, Role, ServerFlaskInterface
from ..strategy import FedStrategyInterface
from ..strategy.build_in import *
from ..utils import pickle_string_to_obj  # TODO(fgh) remove from this file
from .container import ContainerId
from .logger import HyperLogger
from .node import Node


class Server(Node):
    """a central server implementation based on FlaskNode."""

    def __init__(self):
        ConfigurationManager().role = Role.Server
        super().__init__()
        self._construct_fed_model()
        self._init_logger()
        self._init_states()

        self._communicator = ServerFlaskCommunicator()
        self._register_handles()
        self._register_services()

        # save the init weights
        self._hyper_logger.snapshot_model_weights_into_file(self._current_params, self._current_round)

    def _construct_fed_model(self):
        """Construct a federated model according to `self.model_config` and bind it to `self.fed_model`.
            This method only works after `self._bind_configs()`.
        """
        cfg_mgr = ConfigurationManager()
        fed_strategy_type: type = eval(cfg_mgr.model_config.strategy_name)
        self._strategy: FedStrategyInterface = fed_strategy_type()

    def _init_logger(self):
        self._hyper_logger = HyperLogger('Server', 'Server')
        self.logger = self._hyper_logger.get()

        cfg_mgr = ConfigurationManager()
        _run_config = {
            'num_clients': cfg_mgr.runtime_config.client_num,
            'max_num_rounds': cfg_mgr.model_config.max_round_num,
            'num_tolerance': cfg_mgr.model_config.tolerance_num,
            'num_clients_contacted_per_round': cfg_mgr.num_of_clients_contacted_per_round,
            'rounds_between_val': cfg_mgr.model_config.num_of_rounds_between_val,
        }
        self.logger.info(_run_config)
        self.logger.info(self._get_strategy_description())

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
        """initilize attributes for controlling."""
        self._thread_lock = threading.Lock()
        self._STOP = False
        self._server_job_finish = False
        self._client_sids_selected: Optional[List[Any]] = None
        self._invalid_tolerate: int = 0  # for client-side evaluation

        self._lazy_update = False

    def _init_states(self):
        self._init_statistical_states()
        self._init_control_states()
        self._current_params = self._strategy.host_get_init_params()
        self._init_metric_states()

    def _register_services(self):
        @self._communicator.route(ServerFlaskInterface.Dashboard.value)
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

            cfg_mgr = ConfigurationManager()
            metrics = cfg_mgr.model_config.metrics
            test_accuracy_key = f'test_{metrics[0]}'

            return render_template(
                'dashboard.html',
                status='Finish' if self._STOP else 'Running',
                rounds=f"{self._current_round} / {cfg_mgr.model_config.max_round_num}",
                num_online_clients=f"{cfg_mgr.num_of_clients_contacted_per_round} / {len(self._communicator.ready_client_ids)} / {cfg_mgr.runtime_config.client_num}",
                avg_test_metric=self._avg_test_metrics,
                avg_test_metric_keys=avg_test_metric_keys,
                avg_val_metric=self._avg_val_metrics,
                avg_val_metric_keys=avg_val_metric_keys,
                time_record=time_record,
                current_used_time="%02d:%02d:%02d" % (h, m, s),
                test_accuracy=self._best_test_metric.get(test_accuracy_key, 0),
                test_loss=self._best_test_metric.get('test_loss', 0),
                server_send=self._server_send_bytes / (1<<30),
                server_receive=self._server_receive_bytes / (1<<30),
            )

        # TMP use
        @self._communicator.route(ServerFlaskInterface.Status.value)
        def status_page():
            return json.dumps({
                'finished': self._server_job_finish,
                'rounds': self._current_round,
                'results': [
                    None if len(self._avg_val_metrics) == 0 else self._avg_val_metrics[-1], 
                    None if len(self._avg_test_metrics) == 0 else self._avg_test_metrics[-1]
                    ],
                'log_dir': self._hyper_logger.dir_path,
            })

        @self._communicator.route(ServerFlaskInterface.DownloadPattern.value.format('<encoded_file_path>'), methods=['GET'])
        def download_file(encoded_file_path: str):
            file_path = base64.b64decode(encoded_file_path.encode(
                encoding='utf8')).decode(encoding='utf8')
            if os.path.isfile(file_path):
                return send_file(file_path, as_attachment=True)
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

    def _get_strategy_description(self):
        return_value = """\nmodel parameters:\n"""
        for attr in dir(self._strategy):
            attr_value = getattr(self._strategy, attr)
            if type(attr_value) in [str, int, float] and attr.startswith('_') is False:
                return_value += "{}={}\n".format(attr, attr_value)
        return return_value

    def snapshot_result(self, cur_time: float) -> Mapping[str, Any]:
        m, s = divmod(int(round(cur_time) - self._training_start_time), 60)
        h, m = divmod(m, 60)
        keys = ['update_send', 'update_run', 'update_receive', 'agg_server',
                'eval_send', 'eval_run', 'eval_receive', 'server_eval']
        avg_time_records = [np.mean([e.get(key, 0)
                                    for e in self._time_record]) for key in keys]
        return {
            'best_metric': self._best_test_metric,
            'best_metric_full': self._best_test_metric_full,
            'total_time': f'{h}:{m}:{s}',
            'time_detail': str(avg_time_records),
            'total_rounds': self._current_round,
            'server_send': self._server_send_bytes / (1 << 30),
            'server_receive': self._server_receive_bytes / (1 << 30),
            'info_each_round': self._info_each_round
        }

    def _register_handles(self):
        # single-threaded async, no need to lock

        @self._communicator.on(ServerSocketIOEvent.Connect)
        def handle_connect():
            pass

        @self._communicator.on(ServerSocketIOEvent.Reconnect)
        def handle_reconnect():
            recovered_clients = self._communicator.handle_reconnection()
            self.logger.info(f'{recovered_clients} reconnected')

        @self._communicator.on(ServerSocketIOEvent.Disconnect)
        def handle_disconnect():
            disconnected_clients = self._communicator.handle_disconnection()
            self.logger.info(f'{disconnected_clients} disconnected')

        @self._communicator.on(ServerSocketIOEvent.WakeUp)
        def handle_wake_up():
            self._communicator.invoke(ClientSocketIOEvent.Init)

        @self._communicator.on(ServerSocketIOEvent.Ready)
        def handle_client_ready(container_id: ContainerId, client_ids: List[ClientId]):
            self.logger.info(
                f'Container {container_id}, with clients {client_ids} are ready for training')

            self._communicator.activate(container_id, client_ids)

            client_num = ConfigurationManager().runtime_config.client_num
            if len(self._communicator.ready_client_ids) >= client_num and self._current_round == 0:
                self.logger.info("start to federated learning.....")
                self._training_start_time = int(round(time.time()))
                self.train_next_round()
            elif len(self._communicator.ready_client_ids) < client_num:
                self.logger.error("not enough client worker running.....")
            else:
                self.logger.warn("current_round is not equal to 0")

        @self._communicator.on(ServerSocketIOEvent.ResponseUpdate)
        def handle_client_update(data: Mapping[str, Any]):

            if data['round_number'] != self._current_round:
                #TODO(fgh) raise an Exception
                return

            num_clients_contacted_per_round = ConfigurationManager().num_of_clients_contacted_per_round
            with self._thread_lock:
                data['weights'] = pickle_string_to_obj(data['weights'])
                data['time_receive_update'] = time.time()
                self._c_up.append(data)
                receive_all = len(self._c_up) == num_clients_contacted_per_round

            if not receive_all:
                #TODO(fgh) raise an Exception
                return

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
            aggregate_weights /= np.sum(aggregate_weights)

            self._current_params = self._strategy.update_host_params(
                client_params, aggregate_weights)
            self._hyper_logger.snapshot_model_weights_into_file(self._current_params, self._current_round)

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
            self._server_receive_bytes, self._server_send_bytes = self._communicator.get_comm_in_and_out()

            if self._current_round % ConfigurationManager().model_config.num_of_rounds_between_val == 0:
                self.evaluate()
            else:
                self.train_next_round()

            cur_round_info['round_finish_time'] = time.time()

        @self._communicator.on(ServerSocketIOEvent.ResponseEvaluate)
        def handle_client_evaluate(data: Mapping[str, Any]):

            if data['round_number'] != self._current_round:
                #TODO(fgh) raise an Exception
                return

            rt_cfg = ConfigurationManager().runtime_config
            num_clients_contacted_per_round = ConfigurationManager().num_of_clients_contacted_per_round
            with self._thread_lock:
                data['time_receive_evaluate'] = time.time()
                self._c_eval.append(data)
                num_clients_required = num_clients_contacted_per_round if self._lazy_update and not self._STOP else rt_cfg.client_num
                receive_all = len(self._c_eval) == num_clients_required

            if not receive_all:
                #TODO(fgh) raise an Exception
                return

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
                    self._hyper_logger.snap_server_side_best_model_weights_into_file(
                        self._current_params)
                    self.logger.info(str(self._best_test_metric))
                    if not self._lazy_update:
                        self._best_test_metric_full = full_test_metric
                else:
                    self._invalid_tolerate += 1

                if self._invalid_tolerate > ConfigurationManager().model_config.tolerance_num:
                    self.logger.info("converges! starting test phase..")
                    self._STOP = True

                max_round_num = ConfigurationManager().model_config.max_round_num
                if self._current_round >= max_round_num:
                    self.logger.info("get to maximum step, stop...")
                    self._STOP = True

            # Collect the send and received bytes
            self._server_receive_bytes, self._server_send_bytes = self._communicator.get_comm_in_and_out()

            if self._STOP:
                # Another round of testing after the training is finished
                if self._lazy_update and self._best_test_metric_full is None:
                    self.evaluate(self._communicator.ready_client_ids, eval_best_model=True)
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
                    result_json = self.snapshot_result(self._training_stop_time)
                    self._hyper_logger.snapshot_results_into_file(result_json)
                    self._hyper_logger.snapshot_config_into_files()
                    self.logger.info(f'Total time: {result_json["total_time"]}')
                    self.logger.info(f'Time Detail: {result_json["time_detail"]}')
                    self.logger.info(f'Total Rounds: {self._current_round}')
                    self.logger.info(f'Server Send(GB): {result_json["server_send"]}')
                    self.logger.info(f'Server Receive(GB): {result_json["server_receive"]}')

                    # Stop all the clients
                    self._communicator.invoke_all(ClientSocketIOEvent.Stop)
                    # Call the server exit job
                    self._strategy.host_exit_job(self)
                    # Server job finish
                    self._server_job_finish = True
            else:
                results = self.snapshot_result(time.time())
                self._hyper_logger.snapshot_results_into_file(results)
                self._hyper_logger.snapshot_config_into_files() # just for backward compatibility
                self.logger.info("start to next round...")
                self.train_next_round() #TODO(fgh) into loop form

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):

        selected_clients = self._strategy.host_select_train_clients(self._communicator.ready_client_ids)
        self._current_round += 1
        self._info_each_round[self._current_round] = {'timestamp': time.time()}

        # Record the time
        self._time_send_train = time.time()
        self._time_record.append({'round': self._current_round})
        self.logger.info("##### Round {} #####".format(self._current_round))

        # buffers all client updates
        self._c_up = []

        previous_round = self._current_round - 1
        weight_file_path = self._hyper_logger.model_weight_file_path(previous_round)
        encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(encoding='utf8')
        data_send = {'round_number': self._current_round,
                     'weights_file_name': encoded_weight_file_path}
        self._communicator.invoke_all(ClientSocketIOEvent.RequestUpdate,
                                      data_send,
                                      callees=[selected_clients])
        self.logger.info('Finished sending update requests, waiting resp from clients')

    def evaluate(self, selected_clients=None, eval_best_model=False):
        self.logger.info('Starting eval')
        self._c_eval = []
        data_send = { 'round_number': self._current_round }
        weight_file_path = (
            self._hyper_logger.server_side_best_model_weight_file_path 
            if eval_best_model else 
            self._hyper_logger.model_weight_file_path(self._current_round))
        encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(encoding='utf8')
        data_send['weights_file_name'] = encoded_weight_file_path

        # retrieval send information
        if selected_clients is None:
            # selected_clients = self.fed_model.host_select_evaluate_clients(self._communicator.ready_client_ids)
            selected_clients = self._communicator.ready_client_ids

        self.logger.info(f'Sending eval requests to {len(selected_clients)} clients')
        self._communicator.invoke_all(ClientSocketIOEvent.RequestEvaluate,
                                      data_send,
                                      callees=selected_clients)
        self.logger.info('Waiting resp from clients')

    def start(self):
        """start to provide services."""
        self._communicator.run_server()
