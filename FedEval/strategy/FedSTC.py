import gc
gc.set_threshold(700, 10, 5)
import numpy as np
from scipy.sparse import lil_matrix

from ..config.configuration import ConfigurationManager
from ..role import Role
from .FedAvg import FedAvg
from .utils import aggregate_weighted_average


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


class FedSTC(FedAvg):

    def __init__(self, **kwags):
        super().__init__(**kwags)

        if self.role == Role.Client:
            self.client_residual = self.init_residual()
        elif self.role == Role.Server:
            self.server_residual = self.init_residual()
        else:
            raise NotImplementedError

    @staticmethod
    def stc(input_tensor, sparsity=0.01):
        mask = sparse_mask(input_tensor, p=sparsity)
        masked_input = mask * input_tensor
        mu = np.sum(np.abs(masked_input)) / np.sum(mask)
        output_tensor = mu * np.sign(masked_input)
        return output_tensor

    def init_residual(self):
        return [np.zeros(e.shape) for e in self.ml_model.get_weights()]

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
        # 2 Compress the upload tensor and update the local residual
        # stc(delta_w + R), which will be sent to server
        delta_w_plus_r = [
            self.stc(delta_params[i] + self.client_residual[i],
                     sparsity=ConfigurationManager().model_config.stc_sparsity)
            for i in range(len(delta_params))
        ]
        # update the local residual
        self.client_residual = [
            self.client_residual[i] + delta_params[i] - delta_w_plus_r[i]
            for i in range(len(self.client_residual))
        ]
        # Compress the stc(delta_w + R) and return
        result = [self.compress(e.reshape([-1, ])) for e in delta_w_plus_r]
        del delta_params
        del delta_w_plus_r
        gc.collect()
        return result

    def set_host_params_to_local(self, host_params, current_round):
        self.current_round = current_round
        if isinstance(host_params[0], np.ndarray):
            # Receive the init params from the server
            self.ml_model.set_weights(host_params)
        else:
            new_local_params = [
                host_params[i].toarray().reshape(self.local_params_pre[i].shape) + self.local_params_pre[i]
                for i in range(len(host_params))
            ]
            self.ml_model.set_weights(new_local_params)

    def update_host_params(self, client_params, aggregate_weights):
        client_params = [[e[i].toarray().reshape(self.server_residual[i].shape)
                          for i in range(len(e))] for e in client_params]
        delta_W = aggregate_weighted_average(client_params, aggregate_weights)
        delta_W_plus_r = [
            self.stc(delta_W[i] + self.server_residual[i],
                     sparsity=ConfigurationManager().model_config.stc_sparsity)
            for i in range(len(delta_W))
        ]
        # update the residual
        self.server_residual = [
            self.server_residual[i] + delta_W[i] - delta_W_plus_r[i]
            for i in range(len(self.server_residual))
        ]
        # Compress the stc(delta_w + R) and return
        result = [self.compress(e.reshape([-1, ])) for e in delta_W_plus_r]
        del delta_W
        del delta_W_plus_r
        del client_params
        gc.collect()
        return result
