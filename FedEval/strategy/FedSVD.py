import os
import enum
import pickle
import time
import json
import shutil
import hickle
import psutil
from matplotlib.pyplot import axis
import numpy as np
from paramiko import client
import tensorflow as tf

from .FederatedStrategy import FedStrategy, HostParamsType
from ..config import ClientId, ConfigurationManager, Role
from ..dataset import get_data_shape
from typing import List
from sklearn.decomposition import TruncatedSVD
from ..utils import ParamParser, ParamParserInterface
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.lib.format import open_memmap


def generate_orthogonal_matrix(
        n=1000, reuse=False, block_reduce=None, random_seed=None,
        only_inverse=False, file_name=None, memory_efficient=False, clear_cache=False
):
    orthogonal_matrix_cache_dir = '.tmp_orthogonal_matrices'

    if clear_cache:
        shutil.rmtree(orthogonal_matrix_cache_dir, ignore_errors=True)
        return True

    if random_seed:
        np.random.seed(random_seed)

    file_name = file_name or 'cached_matrix'

    if memory_efficient:
        assert reuse

    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    if block_reduce is not None:
        block_reduce = min(block_reduce, n)
        qs = [block_reduce] * int(n / block_reduce)
        if n % block_reduce != 0:
            qs[-1] += (n - np.sum(qs))
        q = []
        for i in range(len(qs)):
            sub_n = qs[i]
            piece_file_name = file_name + f'_piece{i}'
            if reuse and piece_file_name in existing:
                if memory_efficient:
                    tmp = os.path.join(orthogonal_matrix_cache_dir, piece_file_name)
                else:
                    with open(os.path.join(orthogonal_matrix_cache_dir, piece_file_name), 'rb') as f:
                        tmp = pickle.load(f)
            else:
                tmp = generate_orthogonal_matrix(
                    sub_n, reuse=False, block_reduce=None, only_inverse=only_inverse,
                    random_seed=random_seed if random_seed is None else (random_seed + i),
                )
                if reuse:
                    with open(os.path.join(orthogonal_matrix_cache_dir, piece_file_name), 'wb') as f:
                        pickle.dump(tmp, f, protocol=4)
                    if memory_efficient:
                        del tmp
                        tmp = os.path.join(orthogonal_matrix_cache_dir, piece_file_name)
            q.append(tmp)
    else:
        cache_file_name = os.path.join(orthogonal_matrix_cache_dir, file_name)
        if reuse and file_name in existing:
            if memory_efficient:
                q = cache_file_name
            else:
                with open(cache_file_name, 'rb') as f:
                    q = pickle.load(f)
        else:
            if not only_inverse:
                q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
            else:
                q = np.random.randn(n, n)
            if reuse:
                with open(cache_file_name, 'wb') as f:
                    pickle.dump(q, f, protocol=4)
                if memory_efficient:
                    del q
                    q = cache_file_name
    return q


def retrieve_array_from_list(q, start, end):
    results = []
    size_of_q = [len(e) for e in q]
    size_of_q = [sum(size_of_q[:e]) for e in range(len(size_of_q) + 1)]
    for i in range(1, len(size_of_q)):
        if start > size_of_q[i]:
            continue
        # [x, y, array]
        results.append(
            [start, size_of_q[i - 1],
             q[i - 1][start - size_of_q[i - 1]:min(end - size_of_q[i - 1], size_of_q[i] - size_of_q[i - 1])]]
        )
        start += len(results[-1][-1])
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
        self._svd_mode = ConfigurationManager().model_config.svd_mode

        self._masked_u = None
        self._sigma = None
        self._masked_vt = None

        self._memory_map = False

        self._tmp_dir = '.tmp_fedsvd'
        os.makedirs(self._tmp_dir, exist_ok=True)

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
            self._pxq = None
            self._masked_u = None
            self._mask_step_size: int = 0
            self._slice_start = None
            self._slice_end = None
            self._memory_map = False
            # Only for evaluation
            self._evaluation_u = None
            self._evaluation_sigma = None
            self._evaluation_vt = None

        if ConfigurationManager().role is Role.Client:
            # Client
            self._server_machine_id = None
            self._local_machine_id = os.getenv('machine_id', 'local')
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

        if self._svd_mode == 'lr':
            # Parameters for linear regression
            # Client
            self._masked_y = []
            self._is_active = False
            self._local_parameters = None
            # Server
            self._masked_parameters = None
            self._evaluate_parameters = None

        # Clear the cache before starting
        generate_orthogonal_matrix(clear_cache=True)

    def _init_model(self):
        # No machine learning model in FedSVD
        pass

    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        self.train_selected_clients = ready_clients
        return self.train_selected_clients

    def _retrieve_host_download_info(self):
        self.logger.info(f'FedSVD Server Reporting Status: {self._fed_svd_status}')
        # Masking Server
        if self._fed_svd_status is FedSVDStatus.Init:
            # Wait for the clients to send n and m
            return {client_id: {
                'fed_svd_status': self._fed_svd_status, 'server_machine_id': os.getenv('machine_id', 'local')}
                for client_id in self.train_selected_clients
            }
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            if self._random_seed_of_p is None:
                self._random_seed_of_p = np.random.randint(0, 10000000)
            if self._q is None:
                self._q = generate_orthogonal_matrix(
                    sum(self._ns),
                    block_reduce=min(ConfigurationManager().model_config.block_size, sum(self._ns)),
                    random_seed=0, reuse=False, memory_efficient=False
                )
            results_data = {}
            for i in range(len(self._client_ids_on_receiving)):
                client_id = self._client_ids_on_receiving[i]
                results_data[client_id] = retrieve_array_from_list(
                    self._q, start=sum(self._ns[:i]), end=sum(self._ns[:i + 1])
                )
            # Release memory
            del self._q
            return {
                client_id: {
                    'fed_svd_status': self._fed_svd_status,
                    'random_seed_of_p': self._random_seed_of_p,
                    'sliced_q': results_data[client_id]
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            self.logger.info(f'FedSVD Server Reporting Status: apply mask progress '
                             f'{self._apply_mask_progress}/{sum(self._ns)}')
            self._slice_start = self._apply_mask_progress
            self._slice_end = self._slice_start + min(self._mask_step_size, sum(self._ns) - self._slice_start)
            self._apply_mask_progress += (self._slice_end - self._slice_start)
            return {
                client_id: {
                    'fed_svd_status': self._fed_svd_status,
                    'slice_start': self._slice_start, 'slice_end': self._slice_end,
                    'apply_mask_finish': self._apply_mask_progress == sum(self._ns)
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            if self._svd_mode == 'lr':
                # Vt.T @ np.diag(S / (S ** 2 + l2_regularization)) @ U.T @ y_train
                masked_parameters = self._masked_vt.T @ np.diag(
                    self._sigma / (self._sigma ** 2 + ConfigurationManager().model_config.svd_lr_l2)
                ) @ self._masked_u.T @ self._masked_y
                return {
                    client_id: {
                        'fed_svd_status': self._fed_svd_status,
                        'masked_parameters': masked_parameters
                    }
                    for client_id in self._client_ids_on_receiving
                }
            else:
                return {
                    client_id: {
                        'fed_svd_status': self._fed_svd_status,
                        'masked_u': self._masked_u, 'sigma': self._sigma, 'masked_vt': self._masked_vt
                    }
                    for client_id in self._client_ids_on_receiving
                }
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            return {client_id: {'fed_svd_status': self._fed_svd_status}
                    for client_id in self._client_ids_on_receiving}

    def host_get_init_params(self):
        self._fed_svd_status = FedSVDStatus.Init
        return self._retrieve_host_download_info()

    def update_host_params(self, client_params, *args):
        if self._fed_svd_status is FedSVDStatus.Init:
            # By default, the shape of the matrix is n * m
            client_params = sorted(client_params, key=lambda x: x['client_id'])
            ms = [e['mn'][0] for e in client_params]
            assert len(set(ms)) == 1, 'm is not the same among the clients'
            self._client_ids_on_receiving = [e['client_id'] for e in client_params]
            self._ns = [e['mn'][1] for e in client_params]
            self._m = ms[0]
            if client_params[0]['memory_map']:
                self._memory_map = True
                self._pxq = np.memmap(
                    filename=os.path.join(self._tmp_dir, 'server_pxq.npy'),
                    dtype=np.float64, mode='write', shape=(self._m, sum(self._ns))
                )
            else:
                self._pxq = np.zeros([int(self._m), int(sum(self._ns))])
            # Determine the step size when applying the masks
            if self._m <= 1e5:
                self._mask_step_size = ConfigurationManager().model_config.block_size
            else:
                self._mask_step_size = max(int(ConfigurationManager().model_config.block_size / (self._m / 1e5)), 1)
            self._fed_svd_status = FedSVDStatus.SendMask
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            self._fed_svd_status = FedSVDStatus.ApplyMask
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            client_params = sorted(client_params, key=lambda x: x['client_id'])
            self._pxq[:, self._slice_start:self._slice_end] = \
                np.sum([e['secure_agg'] for e in client_params], axis=0, dtype=np.float64)
            if self._memory_map:
                self._pxq.flush()
            if self._svd_mode == 'lr':
                self._masked_y = client_params[-1].get('masked_y')
            del client_params
            if self._apply_mask_progress == sum(self._ns):
                self.logger.info('Server received all masked data. Proceed to SVD.')
                self._fed_svd_status = FedSVDStatus.Factorization
                self._masked_u, self._sigma, self._masked_vt = self._server_svd(self._pxq)
                del self._pxq
                self._fed_svd_status = FedSVDStatus.RemoveMask
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            # Release memory
            del self._masked_u
            del self._masked_vt
            del self._sigma
            self._fed_svd_status = FedSVDStatus.Evaluate
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            # FedSVD Finished
            self._stop = True
            # Only for evaluation, and the clients should not upload local results in real applications
            client_params = sorted(client_params, key=lambda x: x['client_id'])
            if self._svd_mode == 'lr':
                self._evaluate_parameters = np.concatenate([e['parameters'] for e in client_params], axis=0)
            else:
                self._evaluation_u = client_params[0]['u']
                self._evaluation_sigma = client_params[0]['sigma']
                self._evaluation_vt = np.concatenate([e['vt'] for e in client_params], axis=-1)
            # Release memory
            del client_params
            # Nothing to download for clients, since the server is stoping
            return None
        return self._retrieve_host_download_info()

    def _server_svd(self, data_matrix):

        if not self._memory_map:
            if ConfigurationManager().model_config.svd_top_k == -1:
                # We do not need to compute the full matrices when the matrix is not full rank
                self.logger.info(f'Server running SVD on matrix {data_matrix.shape[0]}x{data_matrix.shape[1]}')
                return np.linalg.svd(data_matrix, full_matrices=False)
            elif ConfigurationManager().model_config.svd_top_k > 0:
                assert ConfigurationManager().model_config.svd_top_k <= min(self._m, sum(self._ns))
                # log information
                if self._svd_mode == 'svd':
                    self.logger.info(f'Server running Truncated SVD '
                                     f'using k={ConfigurationManager().model_config.svd_top_k} on matrix '
                                     f'{data_matrix.shape[0]}x{data_matrix.shape[1]}')
                else:
                    self.logger.info(f'Server running PCA '
                                     f'using k={ConfigurationManager().model_config.svd_top_k} on matrix '
                                     f'{data_matrix.shape[0]}x{data_matrix.shape[1]}')
                truncated_svd = TruncatedSVD(
                    n_components=ConfigurationManager().model_config.svd_top_k, algorithm='arpack',
                )
                # By default, we firstly compute the left truncated singular vectors
                truncated_svd.fit(data_matrix.T)
                masked_u = truncated_svd.components_.T
                sigma = truncated_svd.singular_values_
                if self._svd_mode == 'svd':
                    masked_vt = np.diag(sigma ** -1) @ masked_u.T @ data_matrix
                    return masked_u, sigma, masked_vt
                return masked_u, sigma, None
            else:
                raise ValueError(f'Unknown svd top k {ConfigurationManager().model_config.svd_top_k}')
        else:
            m, n = data_matrix.shape
            if m < n:
                x2 = data_matrix @ data_matrix.T
            else:
                x2 = data_matrix.T @ data_matrix
            if ConfigurationManager().model_config.svd_top_k == -1:
                # We do not need to compute the full matrices when the matrix is not full rank
                self.logger.info(f'Server running SVD on matrix {data_matrix.shape[0]}x{data_matrix.shape[1]}')
                left, sigma, _ = np.linalg.svd(x2, full_matrices=False)
                sigma = sigma ** 0.5
                if m < n:
                    right = np.diag(sigma ** -1) @ left.T @ data_matrix
                    return left, sigma, right
                else:
                    right = data_matrix @ left.T @ np.diag(sigma ** -1)
                    return right, sigma, left
            elif ConfigurationManager().model_config.svd_top_k > 0:
                truncated_svd = TruncatedSVD(
                    n_components=ConfigurationManager().model_config.svd_top_k, algorithm='arpack',
                )
                # By default, we firstly compute the left truncated singular vectors
                truncated_svd.fit(x2)
                left = truncated_svd.components_.T
                sigma = truncated_svd.singular_values_ ** 0.5
                if m < n:
                    if self._svd_mode == 'pca':
                        return left, sigma, None
                    else:
                        right = np.diag(sigma ** -1) @ left.T @ data_matrix
                        return left, sigma, right
                else:
                    right = data_matrix @ left.T @ np.diag(sigma ** -1)
                    if self._svd_mode == 'pca':
                        return right, sigma, None
                    else:
                        return right, sigma, left

    def set_host_params_to_local(self, host_params, **kwargs):
        self._current_status = host_params.get('fed_svd_status')
        self.logger.info(f'Client {self.client_id} received server commend: {self._current_status}')
        if self._current_status is FedSVDStatus.Init:
            self._server_machine_id = host_params['server_machine_id']
        elif self._current_status is FedSVDStatus.SendMask:
            if self._received_p_masks is None:
                self.logger.info(f'Client {self.client_id} is generating random mask P')
                self._received_p_masks = host_params['random_seed_of_p']
                self._received_p_masks = generate_orthogonal_matrix(
                    n=self._local_m,
                    block_reduce=min(ConfigurationManager().model_config.block_size, self._local_m),
                    random_seed=self._received_p_masks, reuse=True, file_name=f'client_{self.client_id}',
                    memory_efficient=True
                )
            self._received_q_masks = host_params['sliced_q']
            self._received_q_masks = sorted(self._received_q_masks, key=lambda x: x[0])
            self.logger.info(f'Client {self.client_id} has received random mask Q')
        elif self._current_status is FedSVDStatus.ApplyMask:
            self._received_apply_mask_params = host_params
        elif self._current_status is FedSVDStatus.RemoveMask:
            if self._svd_mode == 'lr':
                self._masked_parameters = host_params['masked_parameters']
            else:
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
                self.logger.info(f'Client {self.client_id} computing P@X')
                # P @ X_i
                if not self._memory_map:
                    self._p_times_x = np.zeros(self.train_data['x'].shape)
                else:
                    self._p_times_x = np.memmap(
                        filename=os.path.join(self._tmp_dir, f'client_{self.client_id}_p_times_x.npy'),
                        dtype=self.train_data['x'].dtype, mode='write', shape=self.train_data['x'].shape
                    )
                p_size_counter = 0
                self._masked_y = []
                for i in range(len(self._received_p_masks)):
                    with open(self._received_p_masks[i], 'rb') as f:
                        tmp_block_mask = pickle.load(f)
                    self._p_times_x[p_size_counter:p_size_counter+len(tmp_block_mask)] =\
                        tmp_block_mask @ self.train_data['x'][p_size_counter: p_size_counter + len(tmp_block_mask)]
                    if self._memory_map:
                        self._p_times_x.flush()
                    # p_times_x.append(
                    #     tmp_block_mask @ self.train_data['x'][p_size_counter: p_size_counter + len(tmp_block_mask)])
                    if self._svd_mode == 'lr' and self._is_active:
                        self._masked_y.append(
                            tmp_block_mask @ self.train_data['y'][p_size_counter: p_size_counter + len(tmp_block_mask)]
                        )
                    p_size_counter += len(tmp_block_mask)
                    del tmp_block_mask
                if self._svd_mode == 'lr' and self._is_active:
                    self._masked_y = np.concatenate(self._masked_y, axis=0)
                if not self._memory_map:
                    del self.train_data

            min_index = self._received_q_masks[0][1]
            max_index = self._received_q_masks[-1][1] + self._received_q_masks[-1][-1].shape[-1]

            self.logger.info(f'Client {self.client_id} SecureAgg from {slice_start} to {slice_end}')

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
                                           [:, counter - self._received_q_masks[j][1]:
                                               min(slice_end - self._received_q_masks[j][1],
                                                   self._received_q_masks[j][-1].shape[-1])]
                                           )
                    else:
                        matrix_base.append(np.zeros([self._local_m, slice_end - max_index]))
                    counter += matrix_base[-1].shape[-1]
                matrix_base = np.concatenate(matrix_base, axis=-1)
                self._sliced_pqx_with_secure_agg = self._generate_masked_data_in_secure_agg(
                    shape=[self._local_m, slice_end - slice_start], base=matrix_base
                )
            if self._received_apply_mask_params['apply_mask_finish']:
                # Release memory
                del self._p_times_x

        elif self._current_status is FedSVDStatus.RemoveMask:

            if self._svd_mode == 'lr':
                self._local_parameters = []
                for x, y, data in self._received_q_masks:
                    self._local_parameters.append(data @ self._masked_parameters[y:y+data.shape[1]])
                del self._masked_parameters
                self._local_parameters = np.concatenate(self._local_parameters, axis=0)
            else:
                self.logger.info(f'Client {self.client_id} is removing masks')
                # Remove the mask of U
                self._u = np.zeros(self._masked_u.shape)
                p_size_counter = 0
                u_counter = 0
                for i in range(len(self._received_p_masks)):
                    with open(self._received_p_masks[i], 'rb') as f:
                        tmp_block_mask = pickle.load(f)
                    self._u[u_counter: u_counter + tmp_block_mask.shape[0]] = \
                        tmp_block_mask.T @ self._masked_u[p_size_counter: p_size_counter + len(tmp_block_mask)]
                    p_size_counter += tmp_block_mask.shape[1]
                    u_counter += tmp_block_mask.shape[0]
                    del tmp_block_mask
                # Remove the mask of VT
                self._local_vt = np.zeros([self._masked_vt.shape[0], self._local_n])
                v_counter = 0
                for i, j, data in self._received_q_masks:
                    self._local_vt[:, v_counter:v_counter + data.shape[0]] = \
                        self._masked_vt[:, j:j + data.shape[1]] @ data.T
                    v_counter += data.shape[0]
                del self._masked_u
                del self._masked_vt
            # Release the memory
            del self._received_p_masks
            del self._received_q_masks
            self.logger.info(f'Client {self.client_id} has finished removing masks')

    def local_evaluate(self):
        # if self._current_status is FedSVDStatus.Evaluate:
        #     if self._svd_mode == 'svd' and \
        #             ConfigurationManager().model_config.svd_top_k == -1:
        #         recovered_data = self._u @ np.diag(self._sigma) @ self._local_vt
        #         reconstruct_mae_error = np.mean(np.abs(recovered_data - self.train_data['x']))
        #         reconstruct_rmse_error = np.sqrt(np.mean(np.square(recovered_data - self.train_data['x'])))
        #         self.logger.info(f'Client {self.client_id} Test RMSE {reconstruct_rmse_error}')
        #         self.logger.info(f'Local Singular Values {self._sigma[:10]}')
        #         self.logger.info(f'Local Singular Vectors {self._local_vt}')
        #         return {
        #             'test_mae': reconstruct_mae_error,
        #             'test_rmse': reconstruct_rmse_error,
        #             'test_size': self._local_n,
        #             'val_loss': reconstruct_rmse_error,
        #             'val_size': self._local_n,
        #         }
        return None

    def client_exit_job(self, client):
        # Clear the cached orthogonal matrices
        generate_orthogonal_matrix(clear_cache=True)

    def retrieve_local_upload_info(self):
        if self._current_status is FedSVDStatus.Init:
            if type(self.train_data['x']) is str:
                self.train_data['x'] = open_memmap(self.train_data['x'])
                self._memory_map = True
            else:
                self._memory_map = False
            self.train_data['x'] = self.train_data['x'].T
            if self._svd_mode == 'lr':
                self._is_active = (self.client_id + 1) == ConfigurationManager().runtime_config.client_num
                if self._is_active:
                    self.train_data['x'] = np.concatenate(
                        [self.train_data['x'], np.ones([self.train_data['x'].shape[0], 1])], axis=-1
                    )
            self._local_m, self._local_n = self.train_data['x'].shape
            return {'client_id': self.client_id, 'mn': self.train_data['x'].shape, 'memory_map': self._memory_map}
        elif self._current_status is FedSVDStatus.ApplyMask:
            if self._received_apply_mask_params['apply_mask_finish'] and self._svd_mode == 'lr' and self._is_active:
                return {
                    'client_id': self._client_id,
                    'secure_agg': self._sliced_pqx_with_secure_agg, 'masked_y': self._masked_y
                }
            else:
                return {'client_id': self._client_id, 'secure_agg': self._sliced_pqx_with_secure_agg}
        elif self._current_status is FedSVDStatus.Evaluate:
            # Only for evaluation, and the clients should not upload local results in real applications
            if self._svd_mode == 'lr':
                return {'client_id': self._client_id, 'parameters': self._local_parameters}
            else:
                if self.client_id == 0:
                    return {'client_id': self._client_id, 'u': self._u, 'sigma': self._sigma, 'vt': self._local_vt}
                else:
                    return {'client_id': self._client_id, 'vt': self._local_vt}

    def host_exit_job(self, host):

        self.logger.info('FedSVD Server Status: Computing the metrics.')

        def mse(x1, x2):
            return np.mean((x1 - x2) ** 2)

        def signed_rmse(x1, x2):
            res = []
            for i in range(x1.shape[0]):
                signed_error = min(mse(x1[i], x2[i]), mse(x1[i], -x2[i]))
                res.append(signed_error)
            return np.sqrt(np.sum(res) / x1.shape[0])

        def project_distance(u1, u2, block=10000):
            distance_sum = 0
            for i in range(0, len(u1), block):
                for j in range(0, len(u1), block):
                    distance_sum += np.sum(
                        (u1[i:i + block] @ u1[j:j + block].T - u2[i:i + block] @ u2[j:j + block].T) ** 2
                    )
            return np.sqrt(distance_sum)

        cfg_mgr = ConfigurationManager()
        client_num = cfg_mgr.runtime_config.client_num
        data_dir = cfg_mgr.data_config.dir_name
        # Get current results
        result_json = host.snapshot_result(None)

        self.logger.info(f'Debug 1 Memory Usage {psutil.virtual_memory().used / 2**30}GB')
        # Load clients' data for evaluation
        if self._memory_map:
            x_train = np.memmap(
                filename=os.path.join(self._tmp_dir, 'server_data.npy'),
                mode='write', dtype=np.float64, shape=(self._m, sum(self._ns))
            )
        else:
            x_train = np.zeros((self._m, sum(self._ns)))
        for client_id in range(client_num):
            with open(os.path.join(data_dir, 'client_%s.pkl' % client_id), 'r') as f:
                data = hickle.load(f)
            if self._memory_map:
                x_train[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])] = open_memmap(data['x_train']).T
                x_train.flush()
            else:
                x_train[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])] = data['x_train'].T
            self.logger.info(f'Memory Usage after loading Client '
                             f'{client_id} {psutil.virtual_memory().used / 2 ** 30}GB')
            if self._svd_mode == 'lr' and client_id == (client_num - 1):
                y_train = data['y_train']

        if self._svd_mode == 'lr':
            x_train = np.hstack([x_train, np.ones([x_train.shape[0], 1])])
            # SVD prediction and errors
            y_hat_svd = x_train @ self._evaluate_parameters
            svd_mse = mean_squared_error(y_true=y_train, y_pred=y_hat_svd)
            svd_mape = mean_absolute_percentage_error(y_true=y_train, y_pred=y_hat_svd)
            svd_r2 = r2_score(y_true=y_train, y_pred=y_hat_svd)
            self.logger.info(f'SVD-LR evaluation done. MSE {svd_mse} MAPE {svd_mape} R2 {svd_r2}')
            # Centralized SGD and errors
            x_train = x_train[:, :-1]
            learning_rate = 0.01
            tolerance = 10
            batch_size = min(16, x_train.shape[0])
            max_epoch = 10000
            l2 = ConfigurationManager().model_config.svd_lr_l2
            linear_regression_model = tf.keras.Sequential()
            linear_regression_model.add(tf.keras.layers.Dense(
                1 if len(y_train.shape) == 1 else y_train.shape[-1],
                kernel_regularizer=tf.keras.regularizers.l2(l2),
            ))
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            linear_regression_model.compile(optimizer=optimizer, loss=tf.keras.losses.mse)
            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=0, patience=tolerance, verbose=0,
                mode='auto', baseline=None, restore_best_weights=True
            )
            linear_regression_model.fit(
                x_train, y_train, batch_size=batch_size, epochs=max_epoch, verbose=1, callbacks=[es]
            )
            y_hat_sgd = linear_regression_model.predict(x_train)
            sgd_mse = float(mean_squared_error(y_true=y_train, y_pred=y_hat_sgd))
            sgd_mape = float(mean_absolute_percentage_error(y_true=y_train, y_pred=y_hat_sgd))
            sgd_r2 = float(r2_score(y_true=y_train, y_pred=y_hat_sgd))
            self.logger.info(
                f"SGD-LR evaluation done. MSE {sgd_mse} MAPE {sgd_mape} R2 {sgd_r2}"
            )
            result_json.update({
                'svd_mse': svd_mse, 'svd_mape': svd_mape, 'svd_r2': svd_r2,
                'sgd_mse': sgd_mse, 'sgd_mape': sgd_mape,
                'sgd_r2': sgd_r2
            })
        else:
            # Centralized SVD
            self.logger.info('FedSVD Server Status: Centralized SVD.')
            c_svd_start = time.time()
            c_u, c_sigma, c_vt = self._server_svd(x_train)
            c_svd_end = time.time()

            result_json.update({'local_svd_time': c_svd_end - c_svd_start})

            # Filter out the very small singular values before calculating the metrics
            if (c_sigma < 10e-10).any():
                significant_index = np.where(c_sigma < 10e-10)[0][0]
                c_sigma = c_sigma[:significant_index]
                c_u = c_u[:, :significant_index]
                c_vt = c_vt[:significant_index, :]
                self._evaluation_sigma = self._evaluation_sigma[:significant_index]
                self._evaluation_u = self._evaluation_u[:, :significant_index]
                self._evaluation_vt = self._evaluation_vt[:significant_index, :]

            # RMSE metric
            self.logger.info('FedSVD Server Status: Computing the RMSE.')
            singular_value_rmse = signed_rmse(c_sigma, self._evaluation_sigma)
            singular_vector_rmse = signed_rmse(c_u.T, self._evaluation_u.T)
            if self._svd_mode == 'svd':
                singular_vector_rmse += signed_rmse(c_vt, self._evaluation_vt)
                singular_vector_rmse /= 2
            result_json.update({'singular_value_rmse': singular_value_rmse})
            result_json.update({'singular_vector_rmse': singular_vector_rmse})

            self.logger.info(f'FedSVD Server Status: Singular Value RMSE {singular_value_rmse}.')
            self.logger.info(f'FedSVD Server Status: Singular Vectors RMSE {singular_vector_rmse}.')

            if (ConfigurationManager().model_config.svd_top_k != -1 and self._svd_mode == 'svd') or \
                    self._svd_mode == 'pca':
                self.logger.info('FedSVD Server Status: Computing the project_distance.')
                singular_vector_project_distance = project_distance(c_u, self._evaluation_u)
                if self._svd_mode == 'svd':
                    singular_vector_project_distance += project_distance(c_vt.T, self._evaluation_vt.T)
                result_json.update({'singular_vector_project_distance': singular_vector_project_distance})

        with open(os.path.join(host.log_dir, 'results.json'), 'w') as f:
            json.dump(result_json, f)

        result_path = os.path.join(host.log_dir, 'results.json')
        self.logger.info(f'FedSVD Server Status: Results saved to {result_path}.')

        # Clear cached data
        shutil.rmtree(ConfigurationManager().data_config.dir_name, ignore_errors=True)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
