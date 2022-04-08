import random
import numpy as np
import tensorflow as tf

from ..callbacks import *
from ..config.configuration import ConfigurationManager
from ..model import *
from ..utils import ParamParser
from .FederatedStrategy import FedStrategy


class FedAvg(FedStrategy):

    def __init__(self, param_parser=ParamParser):
        super().__init__(param_parser)

    # Testing Function, which is not used by any strategy
    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.ml_model(x)
            loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
            loss = loss_op(y, y_hat)
            gradients = tape.gradient(loss, self.ml_model.trainable_variables)
        return gradients

    def host_select_train_clients(self, ready_clients):
        num_selected_clients = min(100, ConfigurationManager().num_of_clients_contacted_per_round)
        if self.eval_selected_clients is not None and len(self.eval_selected_clients) >= num_selected_clients:
            self.train_selected_clients = np.random.choice(
                list(self.eval_selected_clients), num_selected_clients, replace=False
            )
        else:
            self.train_selected_clients = np.random.choice(
                list(ready_clients), num_selected_clients, replace=False
            )
        return self.train_selected_clients

    def host_select_evaluate_clients(self, ready_clients):
        self.eval_selected_clients = random.sample(
            list(ready_clients),
            min(100, ConfigurationManager().num_of_clients_contacted_per_round)
        )
        return self.eval_selected_clients


class FedSGD(FedAvg):
    pass
