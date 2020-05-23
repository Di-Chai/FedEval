import os
import sys
import json
import time
import uuid
import random
import pickle
import logging
import datetime
import threading
import numpy as np

from flask import request, Flask, render_template, send_file
from flask_socketio import SocketIO, emit
from ..utils import pickle_string_to_obj, obj_to_pickle_string
from pympler import asizeof
from werkzeug.utils import secure_filename


class Aggregator(object):
    """docstring for GlobalModel"""

    def __init__(self, model, logger):

        self.model = model

        self.model.build()

        self.logger = logger
        self.logger.info(self.get_model_description())

        self.current_weights = self.get_init_parameters()
        self.model_path = os.path.join(self.model.model_dir, self.model.code_version + '.pkl')

        # weights should be a ordered list of parameter
        # for stats
        self.train_losses = []
        self.avg_test_losses = []
        self.avg_test_maps = []
        self.avg_test_recalls = []

        self.avg_val_metrics = []
        self.avg_test_metrics = []

        # for convergence check
        self.prev_metric = None
        self.best_val_metric = None
        self.best_test_metric = {}
        self.best_weight = None
        self.best_round = -1

        self.training_start_time = int(round(time.time()))
        self.training_stop_time = None

    def get_init_parameters(self):
        return self.model.get_weights()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        print(client_sizes)
        total_size = np.sum(client_sizes)
        new_weights = {key: np.zeros(param.shape, dtype=np.float32) for key, param in client_weights[0].items()}
        for c in range(len(client_weights)):
            for key in new_weights:
                new_weights[key] += (client_weights[c][key] * client_sizes[c] / total_size)
        self.current_weights = new_weights

    # cur_round could None
    def aggregate_train_loss(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        return aggr_loss

    def get_model_description(self):
        return_value = """\nmodel parameters:\n"""
        for attr in dir(self.model):
            attr_value = getattr(self.model, attr)
            if type(attr_value) in [str, int, float] and attr.startswith('_') is False:
                return_value += "{}={}\n".format(attr, attr_value)
        return return_value


class Server(object):
    def __init__(self, server_config, model, host, port):
        self.task_config = server_config
        self.ready_client_sids = set()

        self.host = host
        self.port = port
        self.client_resource = {}

        self.MIN_NUM_WORKERS = self.task_config["MIN_NUM_WORKERS"]
        self.MAX_NUM_ROUNDS = self.task_config["MAX_NUM_ROUNDS"]
        self.NUM_TOLERATE = self.task_config["NUM_TOLERATE"]
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = self.task_config["NUM_CLIENTS_CONTACTED_PER_ROUND"]
        print(self.NUM_CLIENTS_CONTACTED_PER_ROUND)
        self.ROUNDS_BETWEEN_VALIDATIONS = self.task_config["ROUNDS_BETWEEN_VALIDATIONS"]
        self.save_gradients = self.task_config["save_gradients"]

        date_str = time.strftime('%m%d')
        time_str = time.strftime('%m%d_%H%M%S', time.localtime())

        self.logger = logging.getLogger("Server")
        self.logger.setLevel(logging.INFO)
        self.log_dir = os.path.join(model.model_dir, model.code_version, "Server")
        self.log_file = os.path.join(self.log_dir, '{}.log'.format(time_str))
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

        self._check_res = None
        self._send_train = None
        self._agg_train_start = None
        self._agg_train_end = None
        self._agg_eval_start = None
        self._agg_eval_end = None
        self.time_record = []
        self.data_send = []
        self.data_receive = []

        self.debug = []
        self.thread_lock = threading.Lock()

        self.STOP = False
        self.server_job_finish = False

        self.wait_time = 0
        self.logger.info(self.task_config)
        self.model_id = str(uuid.uuid4())

        self.aggregator = Aggregator(model, self.logger)

        self.model_path = os.path.abspath(self.log_dir)
        self.latest_weight_filename = 'latest_model.pkl'
        self.best_weight_filename = 'best_model.pkl'
        self.weights_from_clients = {}

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.c_up = []
        self.c_eval = []
        #####

        self.upload_folder = self.log_dir
        self.allowed_file_types = {'.pkl'}

        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'secret!'
        self.app.config['UPLOAD_FOLDER'] = self.upload_folder
        self.socketio = SocketIO(self.app, max_http_buffer_size=10 ** 20, async_handlers=True,
                                 ping_timeout=3600, ping_interval=1800)

        # socket io messages
        self.register_handles()
        self.invalid_tolerate = 0

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

            time_record = [e for e in self.time_record if len(e.keys()) == 10]
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
                rounds="%s / %s" % (self.current_round, self.MAX_NUM_ROUNDS),
                num_online_clients="%s / %s / %s" % (self.NUM_CLIENTS_CONTACTED_PER_ROUND,
                                                     len(self.ready_client_sids), self.MIN_NUM_WORKERS),
                avg_test_metric=self.aggregator.avg_test_metrics,
                avg_test_metric_keys=avg_test_metric_keys,
                avg_val_metric=self.aggregator.avg_val_metrics,
                avg_val_metric_keys=avg_val_metric_keys,
                time_record=time_record,
                current_used_time="%02d:%02d:%02d" % (h, m, s),
                test_accuracy=self.aggregator.best_test_metric.get('test_accuracy', 0),
                test_loss=self.aggregator.best_test_metric.get('test_loss', 0),
                server_send=np.sum(self.data_send) / (2 ** 30),
                server_receive=np.sum(self.data_receive) / (2 ** 30)
            )

        # TMP use
        @self.app.route('/status')
        def status_page():
            return json.dumps({
                'status': self.server_job_finish,
                'rounds': self.current_round,
                'log_file': self.log_file,
            })

        @self.app.route("/download/<filename>", methods=['GET'])
        def download_file(filename):
            if os.path.isfile(os.path.join(self.model_path, filename)):
                return send_file(os.path.join(self.model_path, filename), as_attachment=True)
            else:
                return json.dumps({'status': 404, 'msg': 'file not found'})

        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_file_types

        @self.app.route('/upload/weights', methods=['POST'])
        def upload_file():
            # self.thread_lock.acquire()
            file = request.files['file']
            cid = file.filename.split('.')[0]
            self.logger.info('Receiving weights... ' + cid)
            with open(os.path.join(os.path.abspath(self.log_dir), file.filename), 'wb') as f:
                pickle.dump(pickle.load(file.stream), f)
            self.logger.info('Received weights ' + cid)
            # self.thread_lock.release()

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
            if len(self.ready_client_sids) >= self.MIN_NUM_WORKERS and self.current_round == -1:
                print("start to federated learning.....")
                self.aggregator.training_start_time = int(round(time.time()))
                self.check_client_resource()
            elif len(self.ready_client_sids) < self.MIN_NUM_WORKERS:
                print("not enough client worker running.....")
            else:
                print("current_round is not equal to -1")

        @self.socketio.on('check_client_resource_done')
        def handle_check_client_resource_done(data):
            self.logger.info('Check Res done')
            if data['round_number'] == self.current_round:
                self.thread_lock.acquire()
                self.client_resource[request.sid] = data['load_rate']
                res_check = len(self.client_resource) == self.NUM_CLIENTS_CONTACTED_PER_ROUND
                self.thread_lock.release()
                if res_check:
                    satisfy = 0
                    client_sids_selected = []
                    for client_id, val in self.client_resource.items():
                        self.logger.info(str(client_id) + "cpu rate: " + str(val))
                        if float(val) < 0.4:
                            client_sids_selected.append(client_id)
                            self.logger.info(str(client_id), "satisfy")
                            satisfy = satisfy + 1
                        else:
                            self.logger.info(str(client_id), "reject")

                    if satisfy / len(self.client_resource) > 0.5:
                        self.wait_time = min(self.wait_time, 3)
                        time.sleep(self.wait_time)
                        self.train_next_round(client_sids_selected)
                    else:
                        if self.wait_time < 10:
                            self.wait_time = self.wait_time + 1
                        time.sleep(self.wait_time)
                        self.check_client_resource()

        @self.socketio.on('client_update')
        def handle_client_update(data):

            if data['round_number'] == self.current_round:

                self.thread_lock.acquire()
                self.logger.info("Train %s" % data['cid'])
                data['update_receive_time'] = time.time() - data['finish_update_time']
                self.c_up.append(data)
                receive_all = len(self.c_up) == self.NUM_CLIENTS_CONTACTED_PER_ROUND
                self.thread_lock.release()

                if receive_all:

                    self.logger.info('Receive update result form %s clients' % len(self.c_up))

                    if self.save_gradients:
                        for i in range(len(self.c_up)):
                            obj_to_pickle_string(
                                self.weights_from_clients[self.c_up[i]['cid']],
                                os.path.join(self.log_dir, self.c_up[i]['cid'] + '_%s.pkl' % self.current_round)
                            )

                    self.data_receive.append(asizeof.asizeof(self.c_up))

                    receive_update_time = [e['receive_update_time'] - self._send_train for e in self.c_up]
                    finish_update_time = [e['finish_update_time'] - e['receive_update_time'] for e in self.c_up]
                    update_receive = [e['update_receive_time'] for e in self.c_up]

                    self.logger.info('Update Run min %s max %s mean %s'
                                     % (min(finish_update_time), max(finish_update_time), np.mean(finish_update_time)))

                    self.time_record[-1]['update_send'] = np.mean(receive_update_time)
                    self.time_record[-1]['update_run'] = np.mean(finish_update_time)
                    self.time_record[-1]['update_receive'] = np.mean(update_receive)

                    # From request update, until receives all clients' update
                    self._agg_train_start = time.time()

                    # current train
                    self.aggregator.update_weights(
                        [self.weights_from_clients[x['cid']] for x in self.c_up],
                        [x['train_size'] for x in self.c_up]
                    )
                    aggr_train_loss = self.aggregator.aggregate_train_loss(
                        [x['train_loss'] for x in self.c_up],
                        [x['train_size'] for x in self.c_up],
                        self.current_round
                    )
                    self.aggregator.train_losses.append(aggr_train_loss)
                    self.logger.info("=== training ===")
                    self.logger.info("aggr_train_loss {}".format(aggr_train_loss))

                    # Fed Aggregate : computation time
                    self._agg_train_end = time.time()
                    self.time_record[-1]['agg_server'] = self._agg_train_end - self._agg_train_start

                    self.logger.info('Finish Update, going to eval...')

                    self.evaluate()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):

            self.thread_lock.acquire()
            self.logger.info('Receive eval from %s' % request.sid)
            data['eval_receive_time'] = time.time() - data['finish_eval_time']
            self.c_eval.append(data)
            receive_all = len(self.c_eval) == self.MIN_NUM_WORKERS
            self.thread_lock.release()

            if receive_all:

                for data in self.c_eval:
                    self.data_receive.append(asizeof.asizeof(data))

                self.logger.info('Receive eval result form %s clients' % len(self.c_eval))

                clients_receive_time = [e['receive_time'] - self._agg_train_end for e in self.c_eval]
                clients_eval_time = [e['finish_eval_time'] - e['receive_time'] for e in self.c_eval]

                self.time_record[-1]['eval_send'] = max(clients_receive_time)
                self.time_record[-1]['eval_run'] = max(clients_eval_time)
                self.time_record[-1]['eval_receive'] = np.mean([e['eval_receive_time'] for e in self.c_eval])

                # Wait time
                self._agg_eval_start = time.time()

                self.logger.info("=== testing ===")

                avg_val_metrics = {}
                avg_test_metrics = {}
                for key in self.c_eval[0]:
                    if key.startswith('val_'):
                        avg_val_metrics[key] = np.mean([float(update[key]) for update in self.c_eval])
                    if key.startswith('test_'):
                        avg_test_metrics[key] = np.mean([float(update[key]) for update in self.c_eval])

                avg_test_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                avg_val_metrics['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                avg_val_metrics['train_loss'] = self.aggregator.train_losses[-1]

                self.aggregator.avg_test_metrics.append(avg_test_metrics)
                self.aggregator.avg_val_metrics.append(avg_val_metrics)

                current_metric = np.mean([float(update['val_default']) for update in self.c_eval])
                self.logger.info('val default %s' % current_metric)

                if self.aggregator.best_val_metric is None or self.aggregator.best_val_metric > current_metric:
                    self.aggregator.best_val_metric = current_metric
                    self.aggregator.best_round = self.current_round
                    self.invalid_tolerate = 0
                    self.aggregator.best_test_metric.update(avg_test_metrics)
                    obj_to_pickle_string(self.aggregator.current_weights,
                                         os.path.join(self.model_path, self.best_weight_filename))
                else:
                    self.invalid_tolerate += 1

                if self.invalid_tolerate > self.NUM_TOLERATE > 0:
                    self.logger.info("converges! starting test phase..")
                    self.STOP = True

                if self.current_round >= self.MAX_NUM_ROUNDS:
                    self.logger.info("get to maximum step, stop...")
                    self.STOP = True

                self._agg_eval_end = time.time()
                self.time_record[-1]['server_eval'] = self._agg_eval_end - self._agg_eval_start

                if self.STOP:
                    self.logger.info("== done ==")
                    self.logger.info("Federated training finished ... ")
                    self.logger.info("best model at round {}".format(self.aggregator.best_round))
                    for key in self.aggregator.best_test_metric:
                        self.logger.info("get best test {} {}".format(key, self.aggregator.best_test_metric[key]))
                    self.aggregator.training_stop_time = int(round(time.time()))
                    # Time
                    m, s = divmod(self.aggregator.training_stop_time - self.aggregator.training_start_time, 60)
                    h, m = divmod(m, 60)
                    self.logger.info('Total time: {}:{}:{}'.format(h, m, s))

                    avg_time_records = []
                    keys = ['check_res', 'update_send', 'update_run', 'update_receive', 'agg_server',
                            'eval_send', 'eval_run', 'eval_receive', 'server_eval']
                    for key in keys:
                        avg_time_records.append(np.mean([e[key] for e in self.time_record]))
                    self.logger.info('Time Detail: ' + str(avg_time_records))
                    self.logger.info('Total Rounds: %s' % self.current_round)
                    self.logger.info('Server Send(GB): %s' % (np.sum(self.data_send) / (2 ** 30)))
                    self.logger.info('Server Receive(GB): %s' % (np.sum(self.data_receive) / (2 ** 30)))
                    # Server job finish
                    self.server_job_finish = True
                    # Stop all the clients
                    emit('stop', broadcast=True)

                else:
                    self.logger.info("start to next round...")
                    self.check_client_resource()

    def check_client_resource(self):
        self._check_res = time.time()
        self.client_resource = {}
        client_sids_selected = random.sample(list(self.ready_client_sids), self.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print('send weights')
        for rid in client_sids_selected:
            emit('check_client_resource', {
                'round_number': self.current_round,
            }, room=rid)

    def response(self, mode, cid):
        self.logger.info('Response: ' + mode + ' %s' % cid)

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self, client_sids_selected):

        self.current_round += 1

        # init resource
        self._send_train = time.time()
        self.time_record.append({'round': self.current_round})

        self.time_record[-1]['check_res'] = self._send_train - self._check_res

        # buffers all client updates
        self.c_up = []
        self.debug = []
        del self.weights_from_clients
        self.weights_from_clients = {}

        self.logger.info("### Round {} ###".format(self.current_round))

        # by default each client cnn is in its own "room"
        if self.current_round == 0:
            obj_to_pickle_string(self.aggregator.current_weights,
                                 os.path.join(self.model_path, self.latest_weight_filename))
            data_send = {
                'model_id': self.model_id,
                'round_number': self.current_round,
                'current_weights_file': self.latest_weight_filename
            }
            data_size = asizeof.asizeof(data_send) + \
                        os.path.getsize(os.path.join(self.model_path, self.latest_weight_filename))
        else:
            data_send = {
                'model_id': self.model_id,
                'round_number': self.current_round
            }
            data_size = asizeof.asizeof(data_send)
        counter = 0
        self.logger.info('Sending train requests to %s clients' % len(client_sids_selected))
        for rid in client_sids_selected:
            emit('request_update', data_send, room=rid, callback=self.response)
            self.logger.info('Send Train %s' % counter)
            counter += 1
            self.data_send.append(data_size)
        self.logger.info('Waiting resp from clients')

    def evaluate(self):
        self.logger.info('Starting eval')
        obj_to_pickle_string(self.aggregator.current_weights, os.path.join(self.model_path, self.latest_weight_filename))
        model_file_size = os.path.getsize(os.path.join(self.model_path, self.latest_weight_filename))
        self.c_eval = []
        data = {
            'model_id': self.model_id,
            'current_weights_file': self.latest_weight_filename,
            'weights_format': 'pickle'
        }
        self.logger.info('Sending eval requests to %s clients' % len(self.ready_client_sids))
        # TODO: lazy update
        for c_sid in self.ready_client_sids:
            emit('evaluate', data, room=c_sid, callback=self.response)
            self.data_send.append(asizeof.asizeof(data) + model_file_size)
        self.logger.info('Waiting resp from clients')

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)