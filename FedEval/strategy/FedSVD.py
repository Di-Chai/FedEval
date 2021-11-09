import os
import pickle
import numpy as np

from .FederatedStrategy import FedStrategy, HostParamsType
from ..config import ClientId, ConfigurationManager, Role
from ..dataset import get_data_shape
from typing import List
from ..utils import ParamParser, ParamParserInterface

"""
Step 1: Server generate P, Q
Step 2: Clients download P, Q
Step 3: Clients compute 
"""


def generate_orthogonal_matrix(n, reuse=False, block_reduce=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_reduce is not None:
        file_name += '_blc%s' % block_reduce

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_reduce is not None:
            qs = [block_reduce] * int(n / block_reduce)
            if n % block_reduce != 0:
                qs[-1] += (n - np.sum(qs))
            q = []
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_reduce=None)
                q.append(tmp)
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def retrieve_array_from_list(q, start, end):
    results = []
    size_of_q = [len(e) for e in q]
    size_of_q = [sum(size_of_q[:e]) for e in range(len(size_of_q)+1)]
    for i in range(1, len(size_of_q)):
        if start > size_of_q[i]:
            continue
        print(start-size_of_q[i-1], min(end-size_of_q[i-1], size_of_q[i]-size_of_q[i-1]))
        results.append(
            [start, size_of_q[i-1], q[i-1][start-size_of_q[i-1]:min(end-size_of_q[i-1], size_of_q[i]-size_of_q[i-1])]]
        )
        start += len(results[-1][-1])
        # print(start, end)
        if start >= end or start >= size_of_q[-1]:
            return results


class FedSVD(FedStrategy):

    def __init__(self, *args, **kwargs):
        super(FedSVD, self).__init__(*args, **kwargs)

        self.fed_svd_step_count = 0
        self._m = None
        self._ns = None
        self._client_ids_on_receiving = None
        # Masking server
        self._random_seed_of_p: int = None
        self._q: list = None
        self._process_of_sending_q: dict = None
        self._times_of_sending_q: dict = None

    def _init_host_params_type(self):
        # Each client holds different Q
        self._host_params_type = HostParamsType.Personalized

    def _init_model(self):
        # No machine learning model in FedSVD
        pass

    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        self.train_selected_clients = ready_clients
        return self.train_selected_clients

    def retrieve_host_download_info(self):
        # Masking Server
        if self.fed_svd_step_count == 0:
            # Wait for the clients to send n and m
            return None
        elif self.fed_svd_step_count == 1:
            if self._random_seed_of_p is None:
                self._random_seed_of_p = np.random.randint(0, 10000000)
            if self._q is None:
                self._q = generate_orthogonal_matrix(
                    sum(self._ns), block_reduce=ConfigurationManager().model_config.block_size)
            if self._process_of_sending_q is None:
                for i in range(len(self._client_ids_on_receiving)):
                    client_id = self._client_ids_on_receiving[i]
                    self._process_of_sending_q[client_id] = 0
            print('debug')
            results_data = {}

            return {
                client_id: [self._random_seed_of_p, ]
                for client_id in self._client_ids_on_receiving
            }
        print('debug')

    def update_host_params(self, client_params, *args):
        if self.fed_svd_step_count == 0:
            # By default, the shape of the matrix is n * m
            ms = [e[1][1] for e in client_params]
            assert len(set(ms)) == 1, 'm is not the same among the clients'
            self._client_ids_on_receiving = [e[0] for e in client_params]
            self._ns = [e[1][0] for e in client_params]
            self._m = ms[0]
            self.fed_svd_step_count += 1
        print('debug')

    def fit_on_local_data(self):
        if self.fed_svd_step_count == 0:
            return -1, len(self.train_data['x'])
        print('debug')

    def retrieve_local_upload_info(self):
        if self.fed_svd_step_count == 0:
            return self._client_id, self.train_data['x'].shape
        else:
            pass
        print('debug')

    def set_host_params_to_local(self, host_params, **kwargs):
        if self.fed_svd_step_count == 0:
            pass
        print('debug')