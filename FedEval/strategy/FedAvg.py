import os
import pickle
import numpy as np
import tensorflow as tf

from .utils import *
from ..model import *
from ..dataset import get_data_shape


class FedAvg:

    def __init__(self, role, data_config, model_config, runtime_config):

        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.role = role

        self.x_size, self.y_size = get_data_shape(dataset=self.data_config.get('dataset'))

        self.ml_model = self.parse_model()

        self.current_round = None
        if self.role.name == 'server':
            self.params = None
            self.gradients = None
        # only clients parse data
        if self.role.name == 'client':
            self.train_data, self.train_data_size, self.val_data, self.val_data_size, \
            self.test_data, self.test_data_size = self.parse_data()
            self.local_params_pre = None
            self.local_params_cur = None

    # Basic parse functions
    def parse_server_addr(self):
        if self.role.name == 'server':
            return self.runtime_config['server']['listen'], self.runtime_config['server']['port']
        else:
            return self.runtime_config['server']['host'], self.runtime_config['server']['port']

    # Basic parse functions
    def parse_model(self):

        ml_model_config = self.model_config['MLModel']
        ml_model_name = ml_model_config.get('name')

        optimizer = tf.keras.optimizers.get(ml_model_config.get('optimizer'))
        optimizer.learning_rate = ml_model_config.get('lr')

        loss = ml_model_config.get('loss')
        metrics = ml_model_config.get('metrics')

        ml_model = eval(ml_model_name)(target_shape=self.y_size, **ml_model_config)
        ml_model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        if ml_model_name == 'MLP':
            self.x_size = (None, np.prod(self.x_size[1:]))
        ml_model.build(input_shape=self.x_size)

        return ml_model

    # Basic parse functions
    def parse_data(self):
        client_id = os.environ.get('CLIENT_ID', '0')
        with open(os.path.join(self.data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
            data = pickle.load(f)
        train_data = {'x': data['x_train'], 'y': data['y_train']}
        val_data = {'x': data['x_val'], 'y': data['y_val']}
        test_data = {'x': data['x_test'], 'y': data['y_test']}
        return train_data, len(data['x_train']), val_data, len(data['x_val']), test_data, len(data['x_test'])

    # Basic parse functions
    def parse_run_config(self):
        return {
            'num_clients': self.runtime_config['server']['num_clients'],
            'max_num_rounds': self.model_config['FedModel']['max_rounds'],
            'num_tolerance': self.model_config['FedModel']['num_tolerance'],
            'num_clients_contacted_per_round': int(self.runtime_config['server']['num_clients']
                                                   * self.model_config['FedModel']['C']),
            'rounds_between_val': self.model_config['FedModel']['rounds_between_val'],
        }

    # (1) Host functions
    def host_get_init_params(self):
        self.params = self._retrieve_local_params()
        return self.params

    # (1) Host functions
    def update_host_params(self, client_params, aggregate_weights):
        self.params = aggregate_weighted_average(client_params, aggregate_weights)
        return self.params

    # (2) Client functions
    def set_host_params_to_local(self, host_params, current_round):
        self.current_round = current_round
        self.ml_model.set_weights(host_params)

    # (2) Client functions
    def fit_on_local_data(self):
        self.local_params_pre = self._retrieve_local_params()
        train_log = self.ml_model.fit(
            x=self.train_data['x'], y=self.train_data['y'],
            epochs=self.model_config['FedModel']['E'],
            batch_size=self.model_config['FedModel']['B']
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self._retrieve_local_params()
        return train_loss, self.train_data_size

    # Will be used by both server and client
    def _retrieve_local_params(self):
        return self.ml_model.get_weights()

    # (2) Client functions
    def retrieve_local_upload_info(self):
        return self._retrieve_local_params()

    # (2) Client functions
    def local_evaluate(self):
        evaluate = {}
        # val and test
        val_result = self.ml_model.evaluate(x=self.val_data['x'], y=self.val_data['y'])
        test_result = self.ml_model.evaluate(x=self.test_data['x'], y=self.test_data['y'])
        metrics_names = self.ml_model.metrics_names
        # Reformat
        evaluate.update({'val_' + metrics_names[i]: float(val_result[i]) for i in range(len(metrics_names))})
        evaluate.update({'test_' + metrics_names[i]: float(test_result[i]) for i in range(len(metrics_names))})
        # TMP
        evaluate.update({'val_size': self.val_data_size})
        evaluate.update({'test_size': self.test_data_size})
        return evaluate

    # def get_leaked_gradients(self):
    #     pass

