import os
import random
import pickle
import numpy as np
import tensorflow as tf

from .utils import *
from ..model import *
from ..dataset import get_data_shape
from ..utils import ParamParser
from ..callbacks import *


class FedAvg:

    def __init__(self, role, data_config, model_config, runtime_config, param_parser=ParamParser, logger=None):

        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.role = role
        self.logger = logger

        self.param_parser = param_parser(
            data_config=data_config, model_config=model_config, runtime_config=runtime_config
        )

        self.ml_model = self.param_parser.parse_model()
        
        self.current_round = None
        if self.role == 'server':
            self.params = None
            self.gradients = None
            # TMP
            run_config = self.param_parser.parse_run_config()
            self.num_clients_contacted_per_round = run_config['num_clients_contacted_per_round']
            self.train_selected_clients = None
        # only clients parse data
        if self.role == 'client':
            self.train_data, self.train_data_size, self.val_data, self.val_data_size, \
            self.test_data, self.test_data_size = self.param_parser.parse_data()
            self.local_params_pre = None
            self.local_params_cur = None

        # TODO: Add the callback modeler for implementing attacks
        if self.model_config['FedModel'].get('callback') is not None:
            self.callback = eval(self.model_config['FedModel']['callback'])()
        else:
            self.callback = None
    
    # Testing Function, which is not used by any strategy
    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.ml_model(x)
            loss_op = tf.keras.losses.get(self.model_config['MLModel']['loss'])
            loss = loss_op(y, y_hat)
            gradients = tape.gradient(loss, self.ml_model.trainable_variables)
        return gradients

    # (1) Host functions
    def host_get_init_params(self):
        self.params = self.ml_model.get_weights()
        return self.params

    # (1) Host functions
    def update_host_params(self, client_params, aggregate_weights):
        if self.callback is not None:
            client_params = self.callback.on_host_aggregate_begin(client_params)
        self.params = aggregate_weighted_average(client_params, aggregate_weights)
        return self.params

    # (1) Host functions
    def host_exit_job(self, host):
        if self.callback is not None:
            self.callback.on_host_exit()

    def host_select_train_clients(self, ready_clients):
        self.train_selected_clients = random.sample(list(ready_clients), self.num_clients_contacted_per_round)
        return self.train_selected_clients

    def host_select_evaluate_clients(self, ready_clients):
        return [e for e in self.train_selected_clients if e in ready_clients]

    # (2) Client functions
    def set_host_params_to_local(self, host_params, current_round):
        if self.callback is not None:
            host_params = self.callback.on_setting_host_to_local(host_params)
        self.current_round = current_round
        self.ml_model.set_weights(host_params)

    # (2) Client functions
    def fit_on_local_data(self):
        if self.callback is not None:
            self.train_data, model = self.callback.on_client_train_begin(
                data=self.train_data, model=self.ml_model.get_weights()
            )
            self.ml_model.set_weights(model)
        self.local_params_pre = self.ml_model.get_weights()
        train_log = self.ml_model.fit(
            x=self.train_data['x'], y=self.train_data['y'],
            epochs=self.model_config['FedModel']['E'],
            batch_size=self.model_config['FedModel']['B']
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self.ml_model.get_weights()
        return train_loss, self.train_data_size

    # (2) Client functions
    def retrieve_local_upload_info(self):
        model = self.ml_model.get_weights()
        if self.callback is not None:
            model = self.callback.on_client_upload_begin(model)
        return model

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

    # (2) Client functions
    def client_exit_job(self, client):
        if self.callback is not None:
            self.callback.on_client_exit()


class FedSGD(FedAvg):
    pass
