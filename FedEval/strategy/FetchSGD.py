import torch
import numpy as np
from .FedAvg import FedAvg
from csvec import CSVec


"""
  # FetchSGD
  num_row: 5
  num_col: 10000
  num_block: 10
  top_k: 0.1
  momentum: 0.9
"""


class FetchSGD(FedAvg):

    def __init__(self, role, data_config, model_config, runtime_config):
        super().__init__(role, data_config, model_config, runtime_config)

        self.param_shapes = [e.shape for e in self.ml_model.get_weights()]
        self.sketch_dim = np.sum([np.prod(e) for e in self.param_shapes])
        self.unSketch_k = int(self.sketch_dim * self.model_config['FedModel']['top_k'])

        self.momentum = self.init_sketch()
        self.error = self.init_sketch()

        print('###############################')
        print(self.sketch_dim)

    def init_sketch(self):
        return CSVec(
            d=self.sketch_dim, c=self.model_config['FedModel']['num_col'],
            r=self.model_config['FedModel']['num_row'], numBlocks=self.model_config['FedModel']['num_block'],
        )

    def retrieve_local_upload_info(self):
        # Client use SGD optimizer
        gradients = [(self.local_params_pre[i] - self.local_params_cur[i]) / self.model_config['MLModel']['lr']
                     for i in range(len(self.local_params_pre))]
        client_sketch = self.init_sketch()
        client_sketch.accumulateVec(torch.from_numpy(np.concatenate([e.reshape([-1, ]) for e in gradients])))
        return client_sketch.table.numpy()

    def _recover_g(self, vector):
        params_lens = [np.prod(e) for e in self.param_shapes]
        gradients = [vector[sum(params_lens[:i]):sum(params_lens[:i+1])]
                     for i in range(len(self.param_shapes))]
        gradients = [gradients[i].reshape(self.param_shapes[i]) for i in range(len(gradients))]
        return gradients

    def update_host_params(self, client_params, aggregate_weights):
        # Aggregate sketches
        agg_tables = np.sum([client_params[i] * aggregate_weights[i] for i in range(len(client_params))], axis=0)
        # Momentum
        self.momentum *= self.model_config['FedModel']['momentum']
        self.momentum.accumulateTable(torch.from_numpy(agg_tables))
        # Error feedback
        self.error.accumulateTable(self.momentum.table * self.model_config['MLModel']['lr'])
        # UnSketch
        delta = self.error.unSketch(k=self.unSketch_k)
        # Error accumulation
        delta_sketch = self.init_sketch()
        delta_sketch.accumulateVec(delta)
        self.error.accumulateTable(delta_sketch.table * -1)
        # update the delta to parameters
        delta_recover = self._recover_g(delta.numpy())
        self.params = [self.params[i] - delta_recover[i] for i in range(len(self.params))]
        return self.params

