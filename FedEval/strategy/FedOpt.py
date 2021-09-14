import numpy as np

from ..config import ConfigurationManager, Role
from .FedAvg import FedAvg
from .utils import aggregate_weighted_average


class FedOpt(FedAvg):

    def __init__(self, **kwags):
        super().__init__(**kwags)
        cfg_mgr = ConfigurationManager()
        role, mdl_cfg = cfg_mgr.role, cfg_mgr.model_config
        if role == Role.Server:
            self.tau = mdl_cfg.opt_tau
            self.beta1 = mdl_cfg.opt_beta_1
            self.beta2 = mdl_cfg.opt_beta_2
            self.eta = mdl_cfg.server_learning_rate
            self.params_shape = [e.shape for e in self.ml_model.get_weights()]
            self.v = [np.zeros(e) + self.tau**2 for e in self.params_shape]
            self.pre_delta_x = [np.zeros(e) for e in self.params_shape]
            self.cur_delta_x = None
        elif role != Role.Client:
            raise NotImplementedError

    # Clients' upload info
    def retrieve_local_upload_info(self):
        delta_x = [self.local_params_cur[i] - self.local_params_pre[i] for i in range(len(self.local_params_cur))]
        return delta_x

    def update_host_params(self, client_params, aggregate_weights):
        delta_x_agg = aggregate_weighted_average(client_params, aggregate_weights)
        self.cur_delta_x = delta_x_agg

        self.cur_delta_x = [
            self.beta1*self.pre_delta_x[i]+(1-self.beta1)*delta_x_agg[i]
            for i in range(len(delta_x_agg))
        ]
        self.pre_delta_x = self.cur_delta_x

        opt_name = ConfigurationManager().model_config.optimizer_name.lower()
        # TODO(fgh) add opt_name restrictions in config module
        if opt_name == 'fedadagrad':
            self.v = [self.v[i] + self.cur_delta_x[i]**2 for i in range(len(self.cur_delta_x))]
        elif opt_name == 'fedyogi':
            self.v = [
                self.v[i] - (1 - self.beta2) * self.cur_delta_x[i]**2 * np.sign(self.v[i] - self.cur_delta_x[i]**2)
                for i in range(len(self.cur_delta_x))
            ]
        elif opt_name == 'fedadam':
            self.v = [self.beta2 * self.v[i] + (1 - self.beta2) * self.cur_delta_x[i]**2
                      for i in range(len(self.cur_delta_x))]

        self.params = [
            self.params[i] + self.eta * self.cur_delta_x[i] / (np.sqrt(self.v[i]) + self.tau)
            for i in range(len(self.params))
        ]

        return self.params

