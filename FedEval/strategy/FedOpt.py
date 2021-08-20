import copy
import numpy as np
from .utils import aggregate_weighted_average
from .FedAvg import FedAvg


class FedOpt(FedAvg):

    def __init__(self, role, data_config, model_config, runtime_config, **kwags):
        super().__init__(role, data_config, model_config, runtime_config, **kwags)

        if self.role == 'server':
            self.tau = self.model_config['FedModel']['tau']
            self.beta1 = self.model_config['FedModel']['beta1']
            self.beta2 = self.model_config['FedModel']['beta2']
            self.eta = self.model_config['FedModel']['eta']
            self.params_shape = [e.shape for e in self.ml_model.get_weights()]
            self.v = [np.zeros(e.shape) + self.tau**2 for e in self.params_shape]
            self.pre_delta_x = None
            self.cur_delta_x = None

            # Clients' upload info
    def retrieve_local_upload_info(self):
        delta_x = [self.local_params_cur[i] - self.local_params_pre[i] for i in range(len(self.local_params_cur))]
        return delta_x

    def update_host_params(self, client_params, aggregate_weights):
        delta_x_agg = aggregate_weighted_average(client_params, aggregate_weights)
        if self.pre_delta_x is None:
            self.pre_delta_x = copy.deepcopy(delta_x_agg)
            self.cur_delta_x = delta_x_agg
        else:
            self.cur_delta_x = [
                self.beta1*self.pre_delta_x[i]+(1-self.beta1)*delta_x_agg
                for i in range(len(delta_x_agg))
            ]
            self.pre_delta_x = copy.deepcopy(delta_x_agg)

        if self.model_config['FedModel']['opt_name'].lower == 'fedadagrad':
            self.v = [self.v[i] + self.cur_delta_x[i]**2 for i in range(len(self.cur_delta_x))]
        elif self.model_config['FedModel']['opt_name'].lower == 'fedyogi':
            self.v = [
                self.v[i] - (1 - self.beta2) * self.cur_delta_x[i]**2 * np.sign(self.v[i] - self.cur_delta_x**2)
                for i in range(len(self.cur_delta_x))
            ]
        elif self.model_config['FedModel']['opt_name'].lower == 'fedadam':
            self.v = [self.beta2 * self.v[i] + (1 - self.beta2) * delta_x_agg[i]**2
                      for i in range(len(self.cur_delta_x))]

        self.params = [
            self.params[i] + self.eta * self.cur_delta_x[i] / (np.sqrt(self.v[i]) + self.tau)
            for i in range(len(self.params))
        ]

        return self.params

