import tensorflow as tf

from ..callbacks import *
from ..config.configuration import ConfigurationManager
from ..model import *
from ..utils import ParamParser
from . import FedStrategy
from .utils import *


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


class FedSGD(FedAvg):
    pass
