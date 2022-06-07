import numpy as np

from ..aggregater import aggregate_weighted_average
from ..config import ConfigurationManager, Role
from ..model import *
from .FedAvg import FedAvg


class MFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if ConfigurationManager().role == Role.Server:
            self.v = None

    def update_host_params(self, client_params, aggregate_weights):
        if self.v is None:
            self.v = [np.zeros(e.shape) for e in self.host_params]
        agg_params = aggregate_weighted_average(client_params, aggregate_weights)
        agg_delta = [self.host_params[i] - agg_params[i] for i in range(len(self.host_params))]
        momentum = ConfigurationManager().model_config.momentum
        self.v = [
            self.v[i] * momentum + agg_delta[i] * (1 - momentum)
            for i in range(len(self.host_params))
        ]
        self.host_params = [
            self.host_params[i] - self.v[i]
            for i in range(len(self.host_params))
        ]
        self.ml_model.set_weights(self.host_params)


class MFedSGD(MFedAvg):
    pass
