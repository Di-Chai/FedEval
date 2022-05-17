import gc
import time

gc.set_threshold(700, 10, 5)

import numpy as np
from scipy.sparse import lil_matrix

from ..aggregater import aggregate_weighted_average
from ..config import ConfigurationManager, Role
from .FederatedStrategy import FedStrategy
from .FedAvg import FedSGD


def sparse_mask(value_list, p=0.01):

    mask = np.zeros(value_list.shape)
    target = np.abs(value_list.reshape([-1, ]))
    target_len = max(int(len(target) * p), 1)

    max_v = np.max(target)
    min_v = np.min(target)
    cut_v = np.median(target)

    pre_cut_v = cut_v
    while True:
        cut_result = np.sum(target > cut_v)
        if cut_result == target_len:
            mask[np.where(np.abs(value_list) > cut_v)] = 1.0
            break
        else:
            if cut_result > target_len:
                target = np.compress(target > cut_v, target)
                min_v = cut_v
            else:
                max_v = cut_v
            cut_v = (max_v + min_v) / 2
            if pre_cut_v == cut_v:
                mask[np.where(np.abs(value_list) > cut_v)] = 1.0
                if cut_result < target_len:
                    c1 = min_v <= np.abs(value_list)
                    c2 = np.abs(value_list) <= max_v
                    mask[[e[:target_len - cut_result] for e in np.where(c1 & c2)]] = 1
                if cut_result > target_len:
                    c1 = min_v <= np.abs(value_list)
                    c2 = np.abs(value_list) <= max_v
                    mask[[e[:cut_result - target_len] for e in np.where(c1 & c2)]] = 0
                break
            else:
                pre_cut_v = cut_v
    return mask


class FedSTC(FedSGD):

    def __init__(self, **kwargs):
        super(FedSTC, self).__init__(**kwargs)
        if ConfigurationManager().role == Role.Client:
            self.client_residual = self._init_residual()
        else:
            self.server_residual = self._init_residual()
            self._delta_W_plus_r = None

    @staticmethod
    def stc(input_tensor, sparsity=0.01):
        results = np.zeros(input_tensor.shape)
        sparse_size = int(len(input_tensor) * sparsity)
        index = np.argpartition(np.abs(input_tensor), -sparse_size)[-sparse_size:]
        results[index] = input_tensor[index]
        mu = np.mean(results[index])
        results[index] = mu * np.sign(results[index])
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
        return lil_matrix(input_tensor)

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
            self.ml_model.set_weights(new_local_params)

    def update_host_params(self, client_params, aggregate_weights):
        st = time.time()
        client_params_dense = [e.toarray()[0] for e in client_params]
        delta_W = np.average(client_params_dense, weights=aggregate_weights, axis=0)
        self.server_residual += delta_W
        del client_params, client_params_dense, delta_W
        self._delta_W_plus_r = self.stc(self.server_residual)
        # update the residual
        self.server_residual -= self._delta_W_plus_r
        model_updates = self._vector_to_tensor(self._delta_W_plus_r)
        self.host_params = [self.host_params[e] + model_updates[e] for e in range(len(self.host_params))]
        self.ml_model.set_weights(self.host_params)
        print(f'Server update params cost {time.time() - st}')

    def retrieve_host_download_info(self):
        if self._delta_W_plus_r is None:
            return super(FedSTC, self).retrieve_host_download_info()
        else:
            return self.compress(self._delta_W_plus_r)
