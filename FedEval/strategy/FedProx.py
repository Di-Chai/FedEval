import tensorflow as tf
import numpy as np

from .FedAvg import FedAvg
from ..utils import ParamParser
from ..model import *
from tensorflow.python.training import gen_training_ops


class FedProxOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, mu=0.01, name='FedProxOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

        self._lr = learning_rate
        self._mu = mu

    def create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'vstar', initializer='zeros')

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError

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

        delta = grad + self._mu * (handle - self.get_slot(handle, "vstar"))

        return gen_training_ops.ResourceApplyGradientDescent(
            var=handle.handle, alpha=coefficients['lr_t'],
            delta=delta, use_locking=self._use_locking
        )


class FedProxParamsParser(ParamParser):
    def parse_model(self):
        ml_model = super(FedProxParamsParser, self).parse_model()
        optimizer = FedProxOptimizer(
            lr=self.model_config['MLModel']['optimizer']['lr'], mu=self.model_config['FedModel']['mu']
        )
        optimizer.create_slots(ml_model.variables)
        ml_model.optimizer = optimizer
        return ml_model


class FedProx(FedAvg):

    def __init__(self, role, data_config, model_config, runtime_config, param_parser=FedProxParamsParser):
        super().__init__(role, data_config, model_config, runtime_config, param_parser=param_parser)

    def fit_on_local_data(self):
        cur_params = self._retrieve_local_params()
        if len(self.ml_model.optimizer.variables()) == len(cur_params):
            self.ml_model.optimizer.set_weights(self._retrieve_local_params())
        else:
            self.ml_model.optimizer.set_weights(self._retrieve_local_params() + [np.array(self.current_round)])
        return super(FedProx, self).fit_on_local_data()
