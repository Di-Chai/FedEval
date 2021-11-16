import base64
import datetime
import json
import pickle
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from flask import render_template, send_file

from ..communicaiton import get_server_communicator
from ..communicaiton.events import *
from ..config import ClientId, ConfigurationManager, Role, ServerFlaskInterface
from ..strategy import FedStrategyInterface
from ..strategy.build_in import *
from ..utils import pickle_string_to_obj  # TODO(fgh) remove from this file
from .container import ContainerId
from .logger import HyperLogger
from .node import Node
from ..strategy.FederatedStrategy import HostParamsType


class Server(Node):
    """a central server implementation based on FlaskNode."""

    def __init__(self):
        ConfigurationManager().role = Role.Server
        super().__init__()
        self._construct_fed_model()
        self._init_logger()
        self._init_states()

        # self._communicator = ServerFlaskCommunicator()
        self._communicator = get_server_communicator()
        self._register_handles()
        self._register_services()

        self._current_params = None

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
        self._info_each_round = {}

    def _init_control_states(self):
        """initilize attributes for controlling."""
        self._thread_lock = threading.Lock()
        self._STOP = False
        self._server_job_finish = False
        self._client_sids_selected: Optional[List[Any]] = None
        self._invalid_tolerate: int = 0  # for client-side evaluation

        self._c_up = []                                      # clients' updates of this round
        self._c_eval = []                                    # clients' evaluations of this round

    def _init_states(self):
        self._init_statistical_states()
        self._init_control_states()
        self._init_metric_states()

    def _refresh_update_cache(self) -> None:
        self._c_up = list()

    def _refresh_evaluation_cache(self) -> None:
        self._c_eval = list()

    def _get_recent_time_records(self, recent_num: int = 0) -> List:
        time_record = [e for e in self._time_record if len(e.keys()) >= 6]
        tmp = {'round', 'eval_receive_time'}
        if len(time_record) > 0:
            time_record.append({'round': 'Average'})
            for key in time_record[0]:
                if key not in tmp:
                    time_record[-1][key] = np.mean([e[key] for e in time_record[:-1]])

            time_record = time_record[-recent_num:]
            # time_record = [time_record[i] for i in range(len(time_record)) if (len(time_record) - 6) <= i]
        return time_record

    def __get_avg_test_metric_keys(self) -> List[str]:
        avg_test_metric = self._avg_test_metrics[0] if self._avg_test_metrics else {}
        return [e for e in avg_test_metric.keys() if e != 'time']

    def __get_avg_val_metric_keys(self) -> List[str]:
        avg_val_metric = self._avg_val_metrics[0] if self._avg_val_metrics else {}
        return [e for e in avg_val_metric.keys() if e != 'time']

    def __get_cur_used_time(self) -> str:
        stopped = self._STOP and self._training_stop_time is not None
        train_stop_time = self._training_stop_time if stopped else round(time.time())
        current_used_time = int(train_stop_time - self._training_start_time)
        m, s = divmod(current_used_time, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)

    def _register_services(self):
        @self._communicator.route(ServerFlaskInterface.Dashboard.value)
        def dashboard():
            """for performance illustration and monitoring.

            Returns:
                the rendered dashboard web page.
            """

            cfg_mgr = ConfigurationManager()
            test_accuracy_key = f'test_{cfg_mgr.model_config.metrics[0]}'

            return render_template(
                'dashboard.html',
                status='Finish' if self._STOP else 'Running',
                rounds=f"{self._current_round} / {cfg_mgr.model_config.max_round_num}",
                num_online_clients=f"{cfg_mgr.num_of_clients_contacted_per_round} / {len(self._communicator.ready_client_ids)} / {cfg_mgr.runtime_config.client_num}",
                avg_test_metric=self._avg_test_metrics,
                avg_test_metric_keys=self.__get_avg_test_metric_keys(),
                avg_val_metric=self._avg_val_metrics,
                avg_val_metric_keys=self.__get_avg_val_metric_keys(),
                time_record=self._get_recent_time_records(6),
                current_used_time=self.__get_cur_used_time(),
                test_accuracy=self._best_test_metric.get(test_accuracy_key, 0),
                test_loss=self._best_test_metric.get('test_loss', 0),
                server_send=self._server_send_bytes / (1<<30),  # in GB
                server_receive=self._server_receive_bytes / (1<<30),    # in GB
            )

        # TMP use
        @self._communicator.route(ServerFlaskInterface.Status.value)
        def status_page():
            return json.dumps({
                'finished': self._server_job_finish,
                'rounds': self._current_round,
                'results': [
                    None if not self._avg_val_metrics else self._avg_val_metrics[-1], 
                    None if not self._avg_test_metrics else self._avg_test_metrics[-1]
                    ],
                'log_dir': self._hyper_logger.dir_path,
            })

        @self._communicator.route(ServerFlaskInterface.DownloadPattern.value.format('<encoded_file_path>'), methods=['GET'])
        def download_file(encoded_file_path: str):
            file_path = base64.urlsafe_b64decode(encoded_file_path.encode(
                encoding='utf8')).decode(encoding='utf8')
            if os.path.isfile(file_path):
                return send_file(file_path, as_attachment=True)
            else:
                return pickle.dumps({'status': 404, 'msg': 'file not found'})

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
        cur_time = self._training_stop_time or cur_time
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

    @property
    def log_dir(self):
        return self._hyper_logger.log_dir_path

    def _register_handles(self):
        # single-threaded async, no need to lock

        @self._communicator.on(ServerEvent.Connect)
        def handle_connect():
            pass

        @self._communicator.on(ServerEvent.Reconnect)
        def handle_reconnect():
            recovered_clients = self._communicator.handle_reconnection()
            self.logger.info(f'{recovered_clients} reconnected')

        @self._communicator.on(ServerEvent.Disconnect)
        def handle_disconnect():
            disconnected_clients = self._communicator.handle_disconnection()
            self.logger.info(f'{disconnected_clients} disconnected')

        @self._communicator.on(ServerEvent.WakeUp)
        def handle_wake_up():
            self._communicator.invoke(ClientEvent.Init)

        @self._communicator.on(ServerEvent.Ready)
        def handle_client_ready(container_id: ContainerId, client_ids: List[ClientId]):
            self.logger.info(
                f'Container {container_id}, with clients {client_ids} are ready for training')

            self._communicator.activate(container_id, client_ids)

            client_num = ConfigurationManager().runtime_config.client_num
            if len(self._communicator.ready_client_ids) >= client_num and self._current_round == 0:
                self.logger.info("start to federated learning.....")
                self._training_start_time = int(round(time.time()))
                del client_num
                self.train_next_round()
            elif len(self._communicator.ready_client_ids) < client_num:
                self.logger.warn("currently, not enough client worker running.....")
            else:
                self.logger.warn("current_round is not equal to 0")

        @self._communicator.on(ServerEvent.ResponseUpdate)
        def handle_client_update(data: Mapping[str, Any]):

            if data['round_number'] != self._current_round:
                #TODO(fgh) raise an Exception
                return

            data['weights'] = pickle_string_to_obj(data['weights'])
            data['time_receive_update'] = time.time()
            with self._thread_lock:
                self._c_up.append(data)
                receive_all = len(self._c_up) == len(self._strategy.train_selected_clients)

            if not receive_all:
                self.logger.debug(
                    f"not all the clients' responses are received during the handling of {ServerEvent.ResponseUpdate.value}")
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
            del receive_update_time, finish_update_time, update_receive_time

            # From request update, until receives all clients' update
            self._time_agg_train_start = time.time()

            # current train
            client_params = [x['weights'] for x in self._c_up]
            aggregate_weights = np.array([x['train_size'] for x in self._c_up]).astype(np.float)
            aggregate_weights /= np.sum(aggregate_weights)

            self._strategy.update_host_params(client_params, aggregate_weights)

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
            cur_round_info['round_finish_time'] = time.time()   # TODO Q(fgh) 这个东西放在这里如果上面调用的是train_next_round那么这个计时合理吗？

            # Collect the send and received bytes
            self._server_receive_bytes, self._server_send_bytes = self._communicator.get_comm_in_and_out()

            if self._current_round % ConfigurationManager().model_config.num_of_rounds_between_val == 0:
                self.evaluate()
            else:
                self.train_next_round()

        @self._communicator.on(ServerEvent.ResponseEvaluate)
        def handle_client_evaluate(data: Mapping[str, Any]):

            if data['round_number'] != self._current_round:
                #TODO(fgh) raise an Exception
                return

            with self._thread_lock:
                data['time_receive_evaluate'] = time.time()
                self._c_eval.append(data)
                num_clients_required = len(self._strategy.eval_selected_clients)
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
            del receive_eval_time, finish_eval_time, eval_receive_time

            self._avg_test_metrics.append(avg_test_metrics)
            self._avg_val_metrics.append(avg_val_metrics)

            current_metric = avg_val_metrics.get('val_loss')
            self.logger.info('val loss %s' % current_metric)

            cur_round_info['time_eval_send'] = latest_time_record['eval_send']
            cur_round_info['time_eval_run'] = latest_time_record['eval_run']
            cur_round_info['time_eval_receive'] = latest_time_record['eval_receive']
            cur_round_info['time_eval_agg'] = latest_time_record['server_eval']

            if self._best_val_metric is None or \
                    (current_metric is not None and self._best_val_metric > current_metric):
                self._best_val_metric = current_metric
                self._best_round = self._current_round
                self._invalid_tolerate = 0
                self._best_test_metric.update(avg_test_metrics)
                self._hyper_logger.snap_server_side_best_model_weights_into_file(
                    self._current_params)
                self.logger.info(str(self._best_test_metric))
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

            if self._strategy.stop:
                self._STOP = True

            # Collect the send and received bytes
            self._server_receive_bytes, self._server_send_bytes = self._communicator.get_comm_in_and_out()

            if self._STOP:
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
                del result_json

                # Clean cached models, while the best model will be kept
                if self._strategy.host_params_type == HostParamsType.Uniform:
                    self._hyper_logger.clear_snapshot(round_num=self._current_round, latest_k=0)
                else:
                    self._hyper_logger.clear_snapshot(
                        round_num=self._current_round,
                        latest_k=0, client_id_list=self._communicator.ready_client_ids
                    )
                # Stop all the clients
                self._communicator.invoke_all(ClientEvent.Stop)
                # Call the server exit job
                self._strategy.host_exit_job(self)
                # Server job finish
                self._server_job_finish = True
            else:
                results = self.snapshot_result(time.time())
                self._hyper_logger.snapshot_results_into_file(results)
                del results
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
        self._refresh_update_cache()

        previous_round = self._current_round - 1

        if not self._hyper_logger.is_snapshot_exist(
                round_num=previous_round, host_params_type=self._strategy.host_params_type,
                client_id_list=selected_clients):
            self._current_params = self._strategy.retrieve_host_download_info()
            self._hyper_logger.snapshot_model_weights_into_file(
                self._current_params, previous_round, self._strategy.host_params_type)

        if self._strategy.host_params_type == HostParamsType.Uniform:
            weight_file_path = self._hyper_logger.model_weight_file_path(previous_round)
            encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(encoding='utf8')
            data_send = {'round_number': self._current_round,
                         'weights_file_name': encoded_weight_file_path}
            self._communicator.invoke_all(ClientEvent.RequestUpdate,
                                          data_send,
                                          callees=selected_clients)
        else:
            for client_id in selected_clients:
                weight_file_path = self._hyper_logger.model_weight_file_path(previous_round, client_id=client_id)
                encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(
                    encoding='utf8')
                data_send = {'round_number': self._current_round,
                             'weights_file_name': encoded_weight_file_path}
                self._communicator.invoke_all(
                    ClientEvent.RequestUpdate, data_send, callees=[client_id]
                )
        self.logger.info('Finished sending update requests, waiting resp from clients')

    def evaluate(self, eval_best_model=False):
        self.logger.info('Starting eval')
        self._refresh_evaluation_cache()

        selected_clients = self._strategy.host_select_evaluate_clients(self._communicator.ready_client_ids)

        if not self._hyper_logger.is_snapshot_exist(
                round_num=self._current_round, host_params_type=self._strategy.host_params_type,
                client_id_list=selected_clients or self._communicator.ready_client_ids):
            self._current_params = self._strategy.retrieve_host_download_info()
            self._hyper_logger.snapshot_model_weights_into_file(
                self._current_params, self._current_round, self._strategy.host_params_type)

        data_send = {'round_number': self._current_round}
        if self._strategy.host_params_type == HostParamsType.Uniform:
            weight_file_path = (
                self._hyper_logger.server_side_best_model_weight_file_path
                if eval_best_model else
                self._hyper_logger.model_weight_file_path(self._current_round))
            encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(
                encoding='utf8')
            data_send['weights_file_name'] = encoded_weight_file_path
            # retrieval send information
            if selected_clients is None:
                # selected_clients = self.fed_model.host_select_evaluate_clients(self._communicator.ready_client_ids)
                selected_clients = self._communicator.ready_client_ids

            self.logger.info(f'Sending eval requests to {len(selected_clients)} clients')
            self._communicator.invoke_all(ClientEvent.RequestEvaluate,
                                          data_send,
                                          callees=selected_clients)
        else:
            for client_id in (selected_clients or self._communicator.ready_client_ids):
                weight_file_path = (
                    self._hyper_logger.server_side_best_model_weight_file_path
                    if eval_best_model else
                    self._hyper_logger.model_weight_file_path(self._current_round, client_id=client_id)
                )
                encoded_weight_file_path = base64.b64encode(weight_file_path.encode(encoding='utf8')).decode(
                    encoding='utf8')
                data_send['weights_file_name'] = encoded_weight_file_path
                self.logger.info(f'Sending eval requests to client {client_id}')
                self._communicator.invoke_all(
                    ClientEvent.RequestEvaluate, data_send, callees=[client_id]
                )

        self.logger.info('Waiting resp from clients')

    def start(self):
        """start to provide services."""
        self._communicator.run_server()
