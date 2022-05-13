import numpy as np
import tensorflow as tf

from ..callbacks import *
from ..config.configuration import ConfigurationManager
from ..model import *
from ..utils import ParamParser
from .FederatedStrategy import FedStrategy


class FedAvg(FedStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def host_select_train_clients(self, ready_clients):
        cfg = ConfigurationManager()
        if self.eval_selected_clients is not None and \
                len(self.eval_selected_clients) >= cfg.num_of_train_clients_contacted_per_round:
            self.train_selected_clients = np.random.choice(
                list(self.eval_selected_clients), cfg.num_of_train_clients_contacted_per_round, replace=False
            )
            # Clear the selected evaluation clients
            self.eval_selected_clients = None
        else:
            self.train_selected_clients = np.random.choice(
                list(ready_clients), cfg.num_of_train_clients_contacted_per_round, replace=False
            )
        return self.train_selected_clients.tolist()

    def host_select_evaluate_clients(self, ready_clients):
        cfg = ConfigurationManager()
        self.eval_selected_clients = np.random.choice(
            list(ready_clients),
            cfg.num_of_eval_clients_contacted_per_round,
            replace=False
        )
        return self.eval_selected_clients.tolist()


class FedSGD(FedAvg):

    # Testing Function, which is not used by any strategy
    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.ml_model(x)
            loss_op = tf.keras.losses.get(ConfigurationManager().model_config.loss_calc_method)
            loss = loss_op(y, y_hat)
            gradients = tape.gradient(loss, self.ml_model.trainable_variables)
        for i in range(len(gradients)):
            try:
                gradients[i] = gradients[i].numpy()
            except AttributeError:
                gradients[i] = tf.convert_to_tensor(gradients[0]).numpy()
        try:
            loss = loss.numpy()
        except AttributeError:
            loss = tf.convert_to_tensor(loss).numpy()
        return loss, gradients

    def host_select_train_clients(self, ready_clients):
        self.train_selected_clients = ready_clients
        return self.train_selected_clients

    def host_select_evaluate_clients(self, ready_clients):
        self.eval_selected_clients = ready_clients
        return self.eval_selected_clients

    def fit_on_local_data(self):
        batched_gradients = []
        batched_loss = []
        actual_size = []
        x_train = self.train_data['x']
        y_train = self.train_data['y']
        parallel_size = 1024
        for i in range(0, len(x_train), parallel_size):
            actual_size.append(min(parallel_size, len(x_train) - i))
            tmp_loss, tmp_gradients = self.compute_gradients(
                x_train[i:i + parallel_size], y_train[i:i + parallel_size])
            batched_gradients.append([e / float(actual_size[-1]) for e in tmp_gradients])
            batched_loss.append(np.mean(tmp_loss))
        actual_size = np.array(actual_size) / np.sum(actual_size)
        aggregated_gradients = []
        for i in range(len(batched_gradients[0])):
            aggregated_gradients.append(np.average([e[i] for e in batched_gradients], axis=0, weights=actual_size))
        batched_loss = np.average(batched_loss, weights=actual_size)
        self.local_params_pre = self.ml_model.get_weights()
        self.ml_model.optimizer.apply_gradients(zip(aggregated_gradients, self.ml_model.trainable_variables))
        self.local_params_cur = self.ml_model.get_weights()
        return batched_loss, len(x_train)
