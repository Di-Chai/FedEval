import numpy as np
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

from ..config import ConfigurationManager
from ..model import *
from ..utils import ParamParser
from .FedAvg import FedAvg


class FedProxOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, mu=0.01, name='FedProxOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

        self._lr = learning_rate
        self._mu = mu

    def create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'vstar', initializer='zeros')

    # def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    #     var_device, var_dtype = handle.device, handle.dtype.base_dtype
    #     coefficients = ((apply_state or {}).get((var_device, var_dtype))
    #                     or self._fallback_apply_state(var_device, var_dtype))
    #     delta = grad + self._mu * (
    #             tf.gather(handle, indices) - tf.gather(self.get_slot(handle, "vstar"), indices))
    #     delta_dense = tf.zeros(handle.shape, handle.dtype)
    #
    #     return gen_training_ops.ResourceApplyGradientDescent(
    #         var=handle.handle, alpha=coefficients['lr_t'],
    #         delta=delta, use_locking=self._use_locking
    #     )

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                                 **kwargs):

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        delta = grad + self._mu * (
                tf.gather(var, indices) - tf.gather(self.get_slot(var, "vstar"), indices))

        return tf.raw_ops.ResourceScatterAdd(
            resource=var.handle,
            indices=indices,
            updates=-delta * coefficients["lr_t"])

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
    @staticmethod
    def parse_model():
        ml_model = ParamParser.parse_model()
        mdl_cfg = ConfigurationManager().model_config
        optimizer = FedProxOptimizer(
            lr=mdl_cfg.learning_rate, mu=mdl_cfg.prox_mu)
        optimizer.create_slots(ml_model.variables)
        ml_model.optimizer = optimizer
        return ml_model


class FedProx(FedAvg):

    def __init__(self, *args, **kwargs):
        if 'param_parser' in kwargs:
            kwargs.pop('param_parser')
        super().__init__(param_parser=FedProxParamsParser, *args, **kwargs)
        # super().__init__(param_parser=ParamParser, *args, **kwargs)

    def fit_on_local_data(self):
        cur_params = self._retrieve_local_params()
        if len(self.ml_model.optimizer.variables()) == len(cur_params):
            self.ml_model.optimizer.set_weights(self._retrieve_local_params())
        else:
            self.ml_model.optimizer.set_weights(self._retrieve_local_params() + [np.array(self.current_round)])
        return super(FedProx, self).fit_on_local_data()
