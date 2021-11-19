import os
import enum
import pickle
import numpy as np

from .FederatedStrategy import FedStrategy, HostParamsType
from ..config import ClientId, ConfigurationManager, Role
from ..dataset import get_data_shape
from typing import List
from sklearn.decomposition import TruncatedSVD
from ..utils import ParamParser, ParamParserInterface


def generate_orthogonal_matrix(
        n, reuse=False, block_reduce=None, random_seed=None, only_inverse=False):

    if random_seed:
        np.random.seed(random_seed)

    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_reduce is not None:
        block_reduce = min(n, block_reduce)
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
                tmp = generate_orthogonal_matrix(
                    sub_n, reuse=False, block_reduce=None, only_inverse=only_inverse,
                    random_seed=random_seed if random_seed is None else (random_seed + 1)
                )
                q.append(tmp)
        else:
            if not only_inverse:
                q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
            else:
                q = np.random.randn(n, n)
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def retrieve_array_from_list(q, start, end):
    results = []
    size_of_q = [len(e) for e in q]
    size_of_q = [sum(size_of_q[:e]) for e in range(len(size_of_q) + 1)]
    for i in range(1, len(size_of_q)):
        if start > size_of_q[i]:
            continue
        print(start - size_of_q[i - 1], min(end - size_of_q[i - 1], size_of_q[i] - size_of_q[i - 1]))
        results.append(
            [start, size_of_q[i-1], q[i - 1][start - size_of_q[i - 1]:min(end - size_of_q[i - 1], size_of_q[i] - size_of_q[i - 1])]]
        )
        start += len(results[-1][-1])
        # print(start, end)
        if start >= end or start >= size_of_q[-1]:
            return results


class FedSVDStatus(enum.Enum):
    Init = 'init'
    SendMask = 'send_mask'
    ApplyMask = 'apply_mask'
    Factorization = 'factorization'
    RemoveMask = 'remove_mask'
    Evaluate = 'evaluate'


class FedSVD(FedStrategy):

    def __init__(self, *args, **kwargs):
        super(FedSVD, self).__init__(*args, **kwargs)

        self.host_params_type = HostParamsType.Personalized

        self._masked_u = None
        self._sigma = None
        self._masked_vt = None

        if ConfigurationManager().role is Role.Server:
            # Masking server
            self._m = None
            self._ns = None
            self._client_ids_on_receiving = None
            self._fed_svd_status = None
            self._random_seed_of_p: int = None
            self._q: list = None
            self._process_of_sending_q: dict = {}
            self._times_of_sending_q: dict = {}
            self._apply_mask_progress: int = 0
            self._pxq = []
            self._masked_u = None

        if ConfigurationManager().role is Role.Client:
            # Client
            self._received_q_masks: list = None
            self._received_p_masks: list = None
            self._current_status = None
            self._p_times_x = None
            self._sliced_pqx_with_secure_agg = None
            np.random.seed(0)
            # We assume that the
            self._secure_agg_random_seed = np.random.random_integers(
                0, 100, [ConfigurationManager().runtime_config.client_num] * 2)
            self._secure_agg_random_seed += self._secure_agg_random_seed.T
            self._received_apply_mask_params = None
            self._u = None
            self._local_vt = None
            self._local_m = None
            self._local_n = None
            self._q_random_mask = None

    def _init_model(self):
        # No machine learning model in FedSVD
        pass

    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        self.train_selected_clients = ready_clients
        return self.train_selected_clients

    def _retrieve_host_download_info(self):
        # Masking Server
        if self._fed_svd_status is FedSVDStatus.Init:
            # Wait for the clients to send n and m
            return {client_id: {'fed_svd_status': self._fed_svd_status}
                    for client_id in self.train_selected_clients}
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            if self._random_seed_of_p is None:
                # self._random_seed_of_p = np.random.randint(0, 10000000)
                # TODO: Remove debug
                self._random_seed_of_p = 1000
            if self._q is None:
                self._q = generate_orthogonal_matrix(
                    sum(self._ns), block_reduce=ConfigurationManager().model_config.block_size,
                    random_seed=0
                )
            results_data = {}
            for i in range(len(self._client_ids_on_receiving)):
                client_id = self._client_ids_on_receiving[i]
                results_data[client_id] = retrieve_array_from_list(
                    self._q, start=sum(self._ns[:i]), end=sum(self._ns[:i + 1])
                )
            return {
                client_id: {
                    'fed_svd_status': self._fed_svd_status,
                    'random_seed_of_p': self._random_seed_of_p,
                    'sliced_q': results_data[client_id]
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            slice_start = self._apply_mask_progress
            slice_end = slice_start + min(
                ConfigurationManager().model_config.block_size, sum(self._ns) - slice_start
            )
            self._apply_mask_progress += (slice_end - slice_start)
            return {
                client_id: {
                    'fed_svd_status': self._fed_svd_status,
                    'slice_start': slice_start, 'slice_end': slice_end
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            return {
                client_id: {
                    'fed_svd_status': self._fed_svd_status,
                    'masked_u': self._masked_u, 'sigma': self._sigma, 'masked_vt': self._masked_vt
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            return {client_id: {'fed_svd_status': self._fed_svd_status}
                    for client_id in self.train_selected_clients}

    def host_get_init_params(self):
        self._fed_svd_status = FedSVDStatus.Init
        return self._retrieve_host_download_info()

    def update_host_params(self, client_params, *args):
        if self._fed_svd_status is FedSVDStatus.Init:
            # By default, the shape of the matrix is n * m
            ms = [e['mn'][0] for e in client_params]
            assert len(set(ms)) == 1, 'm is not the same among the clients'
            self._client_ids_on_receiving = [e['client_id'] for e in client_params]
            self._ns = [e['mn'][1] for e in client_params]
            self._m = ms[0]
            self._fed_svd_status = FedSVDStatus.SendMask
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            self._fed_svd_status = FedSVDStatus.ApplyMask
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            self._pxq.append(np.sum(client_params, axis=0, dtype=np.float64))
            del client_params
            if self._apply_mask_progress == sum(self._ns):
                self._fed_svd_status = FedSVDStatus.Factorization
                self._server_svd()
                self._fed_svd_status = FedSVDStatus.RemoveMask
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            self._fed_svd_status = FedSVDStatus.Evaluate
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            # FedSVD Finished
            self._stop = True
            return None
        return self._retrieve_host_download_info()

    def _server_svd(self):
        self._pxq = np.concatenate(self._pxq, axis=-1)
        if ConfigurationManager().model_config.svd_top_k == -1:
            # We do not need to compute the full matrices when the matrix is not full rank
            self._masked_u, self._sigma, self._masked_vt = np.linalg.svd(self._pxq, full_matrices=False)
        elif ConfigurationManager().model_config.svd_top_k > 0:
            assert ConfigurationManager().model_config.svd_top_k <= min(self._m, sum(self._ns))
            truncated_svd = TruncatedSVD(
                n_components=ConfigurationManager().model_config.svd_top_k, algorithm='arpack',
            )
            # By default, we firstly compute the left truncated singular vectors
            truncated_svd.fit(self._pxq.T)
            self._masked_u = truncated_svd.components_.T
            self._sigma = truncated_svd.singular_values_
            if ConfigurationManager().model_config.svd_mode == 'svd':
                self._masked_vt = np.diag(self._sigma ** -1) @ self._masked_u.T @ self._pxq
        else:
            raise ValueError(f'Unknown svd top k {ConfigurationManager().model_config.svd_top_k}')

    def set_host_params_to_local(self, host_params, **kwargs):
        self._current_status = host_params.get('fed_svd_status')
        if self._current_status is FedSVDStatus.SendMask:
            if self._received_p_masks is None:
                self._received_p_masks = host_params['random_seed_of_p']
                self._received_p_masks = generate_orthogonal_matrix(
                    n=self._local_m, block_reduce=ConfigurationManager().model_config.block_size,
                    random_seed=self._received_p_masks
                )
            self._received_q_masks = host_params['sliced_q']
            self._received_q_masks = sorted(self._received_q_masks, key=lambda x: x[0])
        elif self._current_status is FedSVDStatus.ApplyMask:
            self._received_apply_mask_params = host_params
        elif self._current_status is FedSVDStatus.RemoveMask:
            self._masked_u = host_params['masked_u']
            self._sigma = host_params['sigma']
            self._masked_vt = host_params['masked_vt']
    
    def _generate_masked_data_in_secure_agg(self, shape, base=None):
        if base is not None:
            assert shape[0] == base.shape[0] and shape[1] == base.shape[1]
            result = base
        else:
            result = None
        # Apply the secure aggregation
        for j in range(self._secure_agg_random_seed.shape[1]):
            if self._client_id == j:
                continue
            else:
                np.random.seed(self._secure_agg_random_seed[self._client_id, j])
                r1 = np.random.random(shape) * 2 - 1
                r2 = np.random.random(shape) * 2 - 1
                p = (r1 - r2) if self._client_id < j else (r2 - r1)
                if result is None:
                    result = p.copy()
                else:
                    result += p.copy()
                del r1, r2, p
        return result

    def fit_on_local_data(self):
        if self._current_status is FedSVDStatus.ApplyMask:
            slice_start = self._received_apply_mask_params.get('slice_start')
            slice_end = self._received_apply_mask_params.get('slice_end')
            if self._p_times_x is None:
                # P @ X_i
                p_times_x = []
                p_sizes = [len(e) for e in self._received_p_masks]
                for i in range(len(self._received_p_masks)):
                    p_times_x.append(
                        self._received_p_masks[i] @ self.train_data['x'][sum(p_sizes[:i]): sum(p_sizes[:i + 1])])
                self._p_times_x = np.concatenate(p_times_x, axis=0)

            min_index = self._received_q_masks[0][1]
            max_index = self._received_q_masks[-1][1] + self._received_q_masks[-1][-1].shape[-1]

            if slice_start >= max_index or slice_end <= min_index:
                self._sliced_pqx_with_secure_agg = self._generate_masked_data_in_secure_agg(
                    shape=(self._local_m, slice_end - slice_start))
            else:
                matrix_base = []
                counter = slice_start
                while counter < slice_end:
                    if counter < min_index:
                        matrix_base.append(np.zeros([self._local_m, min_index - counter]))
                    elif counter < max_index:
                        x_slice_index = 0
                        j = 0
                        while counter >= (self._received_q_masks[j][1] + self._received_q_masks[j][-1].shape[-1]):
                            x_slice_index += self._received_q_masks[j][-1].shape[0]
                            j += 1
                        matrix_base.append(self._p_times_x
                            [:, x_slice_index:x_slice_index + self._received_q_masks[j][-1].shape[0]] @
                            self._received_q_masks[j][-1]
                            [:, counter-self._received_q_masks[j][1]:
                                min(slice_end-self._received_q_masks[j][1], self._received_q_masks[j][-1].shape[-1])]
                        )
                    else:
                        matrix_base.append(np.zeros([self._local_m, slice_end - max_index]))
                    counter += matrix_base[-1].shape[-1]
                matrix_base = np.concatenate(matrix_base, axis=-1)
                self._sliced_pqx_with_secure_agg = self._generate_masked_data_in_secure_agg(
                    shape=[self._local_m, slice_end - slice_start], base=matrix_base
                )

        elif self._current_status is FedSVDStatus.RemoveMask:
            # Remove the mask of U
            u = []
            p_sizes = [len(e) for e in self._received_p_masks]
            for i in range(len(self._received_p_masks)):
                u.append(
                    self._received_p_masks[i].T @ self._masked_u[sum(p_sizes[:i]): sum(p_sizes[:i + 1])])
            self._u = np.concatenate(u, axis=0)
            # Remove the mask of VT
            vt = []
            for i, j, data in self._received_q_masks:
                vt.append(self._masked_vt[:, j:j+data.shape[1]] @ data.T)
            self._local_vt = np.concatenate(vt, axis=-1)

    def local_evaluate(self):
        if self._current_status is FedSVDStatus.Evaluate:
            if ConfigurationManager().model_config.svd_mode == 'svd' and \
                    ConfigurationManager().model_config.svd_top_k == -1:
                recovered_data = self._u @ np.diag(self._sigma) @ self._local_vt
                reconstruct_mae_error = np.mean(np.abs(recovered_data - self.train_data['x']))
                reconstruct_rmse_error = np.sqrt(np.mean(np.square(recovered_data - self.train_data['x'])))
                return {
                    'test_mae': reconstruct_mae_error,
                    'test_rmse': reconstruct_rmse_error,
                    'test_size': self._local_n,
                }

    def retrieve_local_upload_info(self):
        if self._current_status is FedSVDStatus.Init:
            self.train_data['x'] = self.train_data['x'].T
            self._local_m, self._local_n = self.train_data['x'].shape
            return {'client_id': self._client_id, 'mn': self.train_data['x'].shape}
        elif self._current_status is FedSVDStatus.ApplyMask:
            return self._sliced_pqx_with_secure_agg

    def client_exit_job(self, client):
        with open(os.path.join(client._hyper_logger.dir_path, f'fedsvd_client_{self.client_id}.pkl'), 'wb') as f:
            pickle.dump({'client_id': self._client_id, 'u': self._u, 'sigma': self._sigma, 'vt': self._local_vt}, f)