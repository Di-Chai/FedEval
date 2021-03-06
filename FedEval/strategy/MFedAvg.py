import os
import pickle
import numpy as np
import tensorflow as tf

from .utils import *
from ..model import *
from ..dataset import get_data_shape
from ..utils import ParamParser
from .FedAvg import FedAvg


class MFedAvg(FedAvg):

    def __init__(self, role, data_config, model_config, runtime_config):
        super().__init__(role, data_config, model_config, runtime_config)

        if role.name == 'server':
            self.v = None

    def update_host_params(self, client_params, aggregate_weights):
        if self.v is None:
            self.v = [np.zeros(e.shape) for e in self.params]
        agg_params = aggregate_weighted_average(client_params, aggregate_weights)
        agg_delta = [self.params[i] - agg_params[i] for i in range(len(self.params))]
        self.v = [
            self.v[i] * self.model_config['FedModel']['momentum'] + agg_delta[i]
            for i in range(len(self.params))
        ]
        self.params = [
            self.params[i] - self.v[i]
            for i in range(len(self.params))
        ]
        return self.params


class MFedSGD(MFedAvg):
    pass
