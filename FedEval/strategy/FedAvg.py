import tensorflow as tf

from .utils import *
from ..model import *
from ..utils import ParamParser
from ..callbacks import *
from . import FedStrategy
from ..role import Role

class FedAvg(FedStrategy):

    def __init__(self, role: Role, data_config, model_config, runtime_config, param_parser=ParamParser):
        super().__init__(role, data_config, model_config, runtime_config, param_parser)

    # Testing Function, which is not used by any strategy
    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.ml_model(x)
            loss_op = tf.keras.losses.get(self.model_config['MLModel']['loss'])
            loss = loss_op(y, y_hat)
            gradients = tape.gradient(loss, self.ml_model.trainable_variables)
        return gradients


class FedSGD(FedAvg):
    pass
