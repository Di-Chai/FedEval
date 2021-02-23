import gc
gc.set_threshold(700, 10, 5)

import time
import heapq
import numpy as np

from pympler import asizeof
from scipy.sparse import lil_matrix


def parse_strategy_name(name_str):
    if name_str not in ['FedSGD', 'FedAvg', 'FedSTC', 'FedDistillate', 'FedMAML']:
        assert ValueError('Cannot understand the specified strategy name, given', name_str)
    return eval(name_str)


def aggregate_weighted_average(client_params, aggregate_weights):
    """
    Args:
        client_params: [{key: value, key: value, ...}, {...}, {...}] are the weights form
            different clients
        aggregate_weights: aggregate weights of different clients, usually set according to the
            clients' training samples. E.g., A, B, and C have 10, 20, and 30 images, then the
            aggregate_weights = [1/6, 1/3, 1/2]

    Returns: the aggregated parameters, which have the same format with any instance from the
        client_params
    """
    new_param = {}
    for c in range(len(client_params)):
        for key in client_params[0]:
            if c == 0:
                new_param[key] = (client_params[c][key] * aggregate_weights[c])
            else:
                new_param[key] += (client_params[c][key] * aggregate_weights[c])
    return new_param


class FedSGD:

    def __init__(self, role, model,
                 train_data=None, val_data=None, test_data=None,
                 upload_strategy=None, train_strategy=None):
        # Init the model
        if upload_strategy is None:
            upload_strategy = {'upload_optimizer': False, 'upload_dismiss': ()}
        if train_strategy is None:
            pass
        self.model = model
        self.model.build()
        self.upload_strategy = upload_strategy
        self.train_strategy = train_strategy
        self.current_round = None
        if role == 'server':
            self.params = None
            self.gradients = None
        # Receive the data
        if role == 'client':
            assert train_data and val_data and test_data
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data
            self.local_params_pre = None
            self.local_params_cur = None

    # (1) Host functions
    def host_get_init_params(self):
        self.global_params = self._retrieve_local_params()
        return self.global_params

    # (1) Host functions
    def update_host_params(self, client_params, aggregate_weights):
        return aggregate_weighted_average(client_params, aggregate_weights)

    # (2) Client functions
    def set_host_params_to_local(self, host_params, current_round):
        self.current_round = current_round
        self.model.set_weights(host_params)

    # (2) Client functions
    def fit_on_local_data(self, train_data, batch_size, epochs):
        self.local_params_pre = self._retrieve_local_params()
        train_loss, train_size = self.model.train_one_round(train_data, epoch=epochs, batch_size=batch_size)
        self.local_params_cur = self._retrieve_local_params()
        return train_loss, train_size

    # Will be used by both server and client
    def _retrieve_local_params(self):
        local_params = self.model.get_trainable_weights(self.upload_strategy.get('upload_dismiss', ()))
        if self.upload_strategy.get('upload_optimizer', False):
            local_params.update(self.model.get_optimizer_weights())
        return local_params
    
    # (2) Client functions
    def retrieve_local_upload_info(self):
        return self._retrieve_local_params()

    # (2) Client functions
    def local_evaluate(self):
        evaluate = {}
        # val and test
        val_result = self.model.evaluate(self.val_data)
        test_result = self.model.evaluate(self.test_data)
        # Reformat
        evaluate.update({'val_' + key: value for key, value in val_result.items()})
        evaluate.update({'test_' + key: value for key, value in test_result.items()})
        # TMP
        evaluate.update({'val_size': self.val_data['x'].shape[0]})
        evaluate.update({'test_size': self.test_data['x'].shape[0]})
        return evaluate

    def get_leaked_gradients(self):
        pass


class FedAvg(FedSGD):
    pass


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

    def __init__(self, role, model,
                 train_data=None, val_data=None, test_data=None,
                 upload_strategy=None, train_strategy=None):
        super(FedSTC, self).__init__(role, model, train_data, val_data, test_data,
                                     upload_strategy=upload_strategy, train_strategy=train_strategy)
        if role == 'client':
            self.client_residual = {}
        else:
            self.server_residual = {}

    @staticmethod
    def stc(input_tensor, sparsity=0.01):
        mask = sparse_mask(input_tensor, p=sparsity)
        masked_input = mask * input_tensor
        mu = np.sum(np.abs(masked_input)) / np.sum(mask)
        output_tensor = mu * np.sign(masked_input)
        return output_tensor

    @staticmethod
    def compress(input_tensor):
        return lil_matrix(input_tensor)

    def retrieve_local_upload_info(self):
        # 1 Get the delta_params
        if self.local_params_pre is None or self.local_params_cur is None:
            raise ValueError('Please call the local fit function first')
        delta_params = {
            key: self.local_params_cur[key] - self.local_params_pre[key]
            for key in self.local_params_cur
        }
        # 2 Compress the upload tensor and update the local residual
        if len(self.client_residual) == 0:
            for key in delta_params:
                self.client_residual[key] = np.zeros(delta_params[key].shape)
        # stc(delta_w + R), which will be sent to server
        delta_w_plus_r = {
            key: self.stc(delta_params[key] + self.client_residual[key],
                          sparsity=self.upload_strategy['upload_sparsity'])
            for key in delta_params
        }
        # update the local residual
        self.client_residual = {
            key: self.client_residual[key] + delta_params[key] - delta_w_plus_r[key]
            for key in self.client_residual
        }
        # Compress the stc(delta_w + R) and return
        result = {key: self.compress(delta_w_plus_r[key].reshape([-1, ])) for key in delta_w_plus_r}
        del delta_params
        del delta_w_plus_r
        gc.collect()
        return result

    def set_host_params_to_local(self, host_params, current_round):
        self.current_round = current_round
        if isinstance(list(host_params.values())[0], np.ndarray):
            # Receive the init params from the server
            self.model.set_weights(host_params)
        else:
            new_local_params = {
                key: host_params[key].toarray().reshape(self.local_params_pre[key].shape) + self.local_params_pre[key]
                for key in host_params
            }
            self.model.set_weights(new_local_params)

    def update_host_params(self, client_params, aggregate_weights):
        client_params = [{key: e[key].toarray() for key in e} for e in client_params]
        delta_W = aggregate_weighted_average(client_params, aggregate_weights)
        if len(self.server_residual) == 0:
            for key in delta_W:
                self.server_residual[key] = np.zeros(delta_W[key].shape)
        delta_W_plus_r = {
            key: self.stc(delta_W[key] + self.server_residual[key], sparsity=self.upload_strategy['upload_sparsity'])
            for key in delta_W
        }
        # update the residual
        self.server_residual = {
            key: self.server_residual[key] + delta_W[key] - delta_W_plus_r[key]
            for key in self.server_residual
        }
        # Compress the stc(delta_w + R) and return
        result = {key: self.compress(delta_W_plus_r[key]) for key in delta_W_plus_r}
        del delta_W
        del delta_W_plus_r
        del client_params
        gc.collect()
        return result


class FedProx:
    pass

