import numpy as np
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

from ..aggregater import aggregate_weighted_average
from ..config.configuration import ConfigurationManager
from ..utils import ParamParser
from .FedAvg import FedAvg


class FedSCAOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, name='FedSCAOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._lr = learning_rate

    def create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'variate_diff', initializer='zeros')

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

    def _resource_apply_dense(self, grad, handle, apply_state):
        var_device, var_dtype = handle.device, handle.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        delta = grad + self.get_slot(handle, "variate_diff")

        return gen_training_ops.ResourceApplyGradientDescent(
            var=handle.handle, alpha=coefficients['lr_t'],
            delta=delta, use_locking=self._use_locking
        )


class FedSCAParser(ParamParser):
    @staticmethod
    def parse_model(client_id=None):
        ml_model = super(FedSCAParser).parse_model(client_id=client_id)
        # Customize the model optimizer
        optimizer = FedSCAOptimizer(lr=ConfigurationManager().model_config.learning_rate)
        optimizer.create_slots(ml_model.variables)
        ml_model.optimizer = optimizer
        return ml_model


class FedSCA(FedAvg):

    def __init__(self, param_parser=ParamParser, logger=None):
        super().__init__(param_parser=param_parser)
        self.set_logger(logger)

        param_shapes = [e.shape for e in self.ml_model.get_weights()]

        self.server_c = [np.zeros(e) for e in param_shapes]
        self.client_c = [np.zeros(e) for e in param_shapes]

    def retrieve_host_download_info(self):
        self.host_params = self.ml_model.get_weights()
        return self.host_params, self.server_c

    def set_host_params_to_local(self, host_params, current_round):
        server_params, self.server_c = host_params
        super(FedSCA, self).set_host_params_to_local(server_params, current_round)

    def update_host_params(self, client_params, aggregate_weights):
        delta_x = aggregate_weighted_average([e[0] for e in client_params], aggregate_weights)
        delta_c = aggregate_weighted_average([e[1] for e in client_params], aggregate_weights)
        mdl_cfg = ConfigurationManager().model_config
        for i in range(len(self.host_params)):
            self.host_params[i] += mdl_cfg.learning_rate * delta_x[i]
            self.server_c[i] += mdl_cfg.C * delta_c[i]
        self.ml_model.set_weights(self.host_params)

    def fit_on_local_data(self):
        variate_diff = [self.server_c[i] - self.client_c[i] for i in range(len(self.client_c))]
        if len(self.ml_model.optimizer.variables()) == len(variate_diff):
            self.ml_model.optimizer.set_weights(variate_diff)
        else:
            self.ml_model.optimizer.set_weights(variate_diff + [np.array(self.current_round)])
        return super(FedSCA, self).fit_on_local_data()

    def retrieve_local_upload_info(self):
        mdl_cfg = ConfigurationManager().model_config
        B, E, lr = mdl_cfg.B, mdl_cfg.E, mdl_cfg.learning_rate
        K = np.ceil(len(self.train_data['x']) / B * E)
        new_c = [
            self.client_c[i] - self.server_c[i] + 
            1 / (K * lr) * (self.local_params_pre[i] - self.local_params_cur[i])
            for i in range(len(self.client_c))
        ]
        delta_params = [self.local_params_cur[i]-self.local_params_pre[i] for i in range(len(self.local_params_cur))]
        delta_c = [new_c[i]-self.client_c[i] for i in range(len(self.client_c))]
        self.client_c = new_c
        return delta_params, delta_c
