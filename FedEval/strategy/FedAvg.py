import os
import pickle
import numpy as np
import tensorflow as tf

from .utils import *
from ..model import *
from ..dataset import get_data_shape
from ..utils import ParamParser


class FedAvg:

    def __init__(self, role, data_config, model_config, runtime_config, param_parser=ParamParser):

        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.role = role

        self.param_parser = param_parser(
            data_config=data_config, model_config=model_config, runtime_config=runtime_config
        )

        self.ml_model = self.param_parser.parse_model()

        self.current_round = None
        if self.role.name == 'server':
            self.params = None
            self.gradients = None
        # only clients parse data
        if self.role.name == 'client':
            self.train_data, self.train_data_size, self.val_data, self.val_data_size, \
            self.test_data, self.test_data_size = self.param_parser.parse_data()
            self.local_params_pre = None
            self.local_params_cur = None

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


class FedSGD(FedAvg):
    pass