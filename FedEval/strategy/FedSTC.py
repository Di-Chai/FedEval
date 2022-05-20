import gc
import time

gc.set_threshold(700, 10, 5)

import numpy as np
from scipy.sparse import csc_matrix

from ..config import ConfigurationManager, Role
from .FedAvg import FedAvg
from .FederatedStrategy import HostParamsType


class FedSTC(FedAvg):

    def __init__(self, **kwargs):
        super(FedSTC, self).__init__(**kwargs)
        if ConfigurationManager().role == Role.Client:
            self.client_residual = self._init_residual()
        else:
            self.server_residual = self._init_residual()
            self._delta_W_plus_r = None
            self._host_params_type = HostParamsType.Personalized
            self._client_rounds = {}
            self._sparse_update_list = []

    @staticmethod
    def stc(input_tensor, sparsity=0.01):
        results = np.zeros(input_tensor.shape)
        sparse_size = int(len(input_tensor) * sparsity)
        index = np.argpartition(np.abs(input_tensor), -sparse_size)[-sparse_size:]
        sliced_input = input_tensor[index]
        mu = np.mean(np.abs(sliced_input))
        results[index] = mu * np.sign(sliced_input)
        del sliced_input
        return results

    def _init_residual(self):
        self._param_shape = [e.shape for e in self.ml_model.get_weights()]
        self._param_size = int(sum([np.prod(e) for e in self._param_shape]))
        return np.zeros([self._param_size])

    def _tensor_to_vector(self, input_tensor):
        results = np.concatenate([e.flatten() for e in input_tensor])
        assert len(results) == self._param_size
        return results

    def _vector_to_tensor(self, input_vector):
        results = []
        pointer = 0
        for i in range(len(self._param_shape)):
            tmp_size = np.prod(self._param_shape[i])
            results.append(np.reshape(
                input_vector[pointer:pointer+tmp_size], self._param_shape[i]))
            pointer += tmp_size
        return results

    @staticmethod
    def compress(input_tensor):
        return csc_matrix(input_tensor)

    def retrieve_local_upload_info(self):
        # 1 Get the delta_params
        if self.local_params_pre is None or self.local_params_cur is None:
            raise ValueError('Please call the local fit function first')
        delta_params = [
            self.local_params_cur[i] - self.local_params_pre[i]
            for i in range(len(self.local_params_cur))
        ]
        delta_params = self._tensor_to_vector(delta_params)

        self.client_residual += delta_params
        delta_w_plus_r = self.stc(self.client_residual)
        # update the local residual
        self.client_residual -= delta_w_plus_r
        # Compress the stc(delta_w + R) and return
        results = self.compress(delta_w_plus_r)
        del delta_params
        del delta_w_plus_r
        gc.collect()
        return results

    def set_host_params_to_local(self, host_params, current_round):
        self.current_round = current_round
        if isinstance(host_params[0], np.ndarray):
            # Receive the init params from the server
            self.ml_model.set_weights(host_params)
        else:
            new_local_params = self._vector_to_tensor(host_params.toarray()[0])
            cur_weights = self.ml_model.get_weights()
            self.ml_model.set_weights([cur_weights[e] + new_local_params[e] for e in range(len(cur_weights))])

    def update_host_params(self, client_params, aggregate_weights):
        client_params_dense = [e.toarray()[0] for e in client_params]
        delta_W = np.average(client_params_dense, weights=aggregate_weights, axis=0)
        self.server_residual += delta_W
        del client_params, client_params_dense, delta_W
        self._delta_W_plus_r = self.stc(self.server_residual)
        # update the residual
        self.server_residual -= self._delta_W_plus_r
        model_updates = self._vector_to_tensor(self._delta_W_plus_r)
        if self.host_params is None:
            self.host_params = self.ml_model.get_weights()
        self.host_params = [self.host_params[e] + model_updates[e] for e in range(len(self.host_params))]
        self.ml_model.set_weights(self.host_params)

    def retrieve_host_download_info(self):
        if self._delta_W_plus_r is not None:
            self._sparse_update_list.append(self.compress(self._delta_W_plus_r))
        sparse_update = {}
        for cid in self.train_selected_clients:
            if self._delta_W_plus_r is None or cid not in self._client_rounds:
                sparse_update[cid] = self.ml_model.get_weights()
            else:
                sparse_update[cid] = sum(self._sparse_update_list[self._client_rounds[cid]:])
            self._client_rounds[cid] = len(self._sparse_update_list)
        return sparse_update
