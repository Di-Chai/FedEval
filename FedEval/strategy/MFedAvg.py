import numpy as np

from ..config import ConfigurationManager, Role
from ..model import *
from .FedAvg import FedAvg
from .utils import *


class MFedAvg(FedAvg):

    def __init__(self, **kwags):
        super().__init__(**kwags)

        if ConfigurationManager().role == Role.Server:
            self.v = None

    def update_host_params(self, client_params, aggregate_weights):
        if self.v is None:
            self.v = [np.zeros(e.shape) for e in self.params]
        agg_params = aggregate_weighted_average(client_params, aggregate_weights)
        agg_delta = [self.params[i] - agg_params[i] for i in range(len(self.params))]
        momentum = ConfigurationManager().model_config.momentum
        self.v = [
            self.v[i] * momentum + agg_delta[i] * (1 - momentum)
            for i in range(len(self.params))
        ]
        self.params = [
            self.params[i] - self.v[i]
            for i in range(len(self.params))
        ]
        return self.params


class MFedSGD(MFedAvg):
    pass
