import datetime
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
import datetime

from .FederatedStrategy import FedStrategy, HostParamsType
from ..config import ClientId, ConfigurationManager, Role
from ..dataset import get_data_shape
from typing import List
from sklearn.decomposition import TruncatedSVD
from ..utils import ParamParser, ParamParserInterface
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.lib.format import open_memmap
from functools import reduce
from dateutil.parser import parse
from apscheduler.schedulers.background import BackgroundScheduler


def generate_orthogonal_matrix(
        n=1000, reuse=False, block_reduce=None, random_seed=None,
        only_inverse=False, file_name=None, offload_to_disk=False, clear_cache=False
):
    orthogonal_matrix_cache_dir = 'tmp_orthogonal_matrices'

    if clear_cache:
        shutil.rmtree(orthogonal_matrix_cache_dir, ignore_errors=True)
        return True

    if random_seed:
        np.random.seed(random_seed)

    file_name = file_name or 'cached_matrix'

    if offload_to_disk:
        assert reuse

    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    existing = set([e.split('.')[0] for e in os.listdir(orthogonal_matrix_cache_dir)])

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
                if offload_to_disk:
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
                    if offload_to_disk:
                        del tmp
                        tmp = os.path.join(orthogonal_matrix_cache_dir, piece_file_name)
            q.append(tmp)
    else:
        cache_file_name = os.path.join(orthogonal_matrix_cache_dir, file_name)
        if reuse and file_name in existing:
            if offload_to_disk:
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
                if offload_to_disk:
                    del q
                    q = cache_file_name
    return q


def retrieve_array_from_list(q, start, end):
    results = []
    size_of_q = [len(e) for e in q]
    size_of_q = [sum(size_of_q[:e]) for e in range(len(size_of_q) + 1)]
    for i in range(1, len(size_of_q)):
        if start >= size_of_q[i]:
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

        cfg = ConfigurationManager()

        self.host_params_type = HostParamsType.Personalized
        self._svd_mode = ConfigurationManager().model_config.svd_mode

        self._masked_u = None
        self._sigma = None
        self._masked_vt = None

        # The block-based optimization are decided by the following flag and the block-size
        self._block_based_optimization = cfg.model_config.svd_opt_1
        self._mini_batch_secure_agg = cfg.model_config.svd_opt_2
        # memory_map are automatically determined according to the input data
        self._memory_map = None

        # Set to false when benchmarking
        self._evaluate_for_debug = cfg.model_config.svd_evaluate

        self._tmp_dir = 'tmp_fedsvd'
        os.makedirs(self._tmp_dir, exist_ok=True)

        if ConfigurationManager().role is Role.Server:
            # Masking server
            self._m = None
            self._ns = None
            self._client_ids_on_receiving = None
            self._fed_svd_status = None
            self._msg_of_p: int = None
            self._q: list = None
            self._process_of_sending_q: dict = {}
            self._times_of_sending_q: dict = {}
            self._apply_mask_progress: int = 0
            self._vector_transfer_progress: int = 0
            self._vector_transfer_finish: bool = False
            self._pxq = None
            self._masked_u = None
            self._mask_step_size: int = 0
            self._slice_start = None
            self._slice_end = None
            self._vertical_slice = None
            self._vector_transfer_start: int = 0
            self._vt_transfer_emd: int = 0
            self._client_masked_q: dict = {}
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
            self._save_p_to_disk = None
            self._current_status = None
            self._p_times_x = None
            self._sliced_pxq_with_secure_agg = None
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
            self._q_random_mask_inverse = None
            self._masked_q = None
            self._global_ns = None
            # used in memory map mode
            self._received_server_params = None

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

        self._fed_svd_status = FedSVDStatus.Init

        self._process_memory_usage = {}
        self._total_memory_usage = [0]
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        # Start tracking the memory usage, report memory every 5 seconds
        self._scheduler.add_job(
            self._log_hardware_usage, trigger='cron', second=f'*/5',
            id='log_memory'
        )
    
    def _log_hardware_usage(self):
        process_memory_usage = psutil.Process().memory_full_info().data + psutil.Process().memory_full_info().swap
        process_memory_usage /= 2**30  # GB
        date_string = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self._process_memory_usage[date_string] = process_memory_usage
        self.logger.info(f'{date_string} Memory Usage {process_memory_usage}')

    def _init_model(self):
        # No machine learning model in FedSVD
        pass

    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        self.train_selected_clients = ready_clients
        return self.train_selected_clients

    def retrieve_host_download_info(self):
        self.logger.info(f'FedSVD Server Reporting Status: {self._fed_svd_status}')
        # Masking Server
        if self._fed_svd_status is FedSVDStatus.Init:
            # Wait for the clients to send n and m
            return {client_id: {
                'fed_svd_status': self._fed_svd_status, 'server_machine_id': os.getenv('machine_id', 'local')}
                for client_id in self.train_selected_clients
            }
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            if self._msg_of_p is None:
                if self._block_based_optimization:
                    self._msg_of_p = np.random.randint(0, 10000000)
                else:
                    self._msg_of_p = [np.zeros([self._m, self._m])]
                    block_list = generate_orthogonal_matrix(
                        self._m,
                        block_reduce=min(ConfigurationManager().model_config.block_size, self._m),
                        random_seed=0, reuse=False, offload_to_disk=False
                    )
                    index_counter = 0
                    for data in block_list:
                        self._msg_of_p[0][index_counter:index_counter+data.shape[0],
                        index_counter:index_counter+data.shape[1]] = data
                        index_counter += data.shape[0]
            if self._q is None:
                if self._block_based_optimization:
                    self._q = generate_orthogonal_matrix(
                        sum(self._ns),
                        block_reduce=min(ConfigurationManager().model_config.block_size, sum(self._ns)),
                        random_seed=0, reuse=False, offload_to_disk=False
                    )
                else:
                    self._q = [np.zeros([sum(self._ns), sum(self._ns)])]
                    block_list = generate_orthogonal_matrix(
                        sum(self._ns),
                        block_reduce=min(ConfigurationManager().model_config.block_size, sum(self._ns)),
                        random_seed=0, reuse=False, offload_to_disk=False
                    )
                    index_counter = 0
                    for data in block_list:
                        self._q[0][index_counter:index_counter + data.shape[0],
                        index_counter:index_counter + data.shape[1]] = data
                        index_counter += data.shape[0]
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
                    'msg_of_p': self._msg_of_p,
                    'sliced_q': results_data[client_id],
                    'ns': sum(self._ns)
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            if hasattr(self, '_msg_of_p'):
                del self._msg_of_p
            self._slice_start = self._apply_mask_progress
            if self._vertical_slice:
                self.logger.info(f'FedSVD Server Reporting Status: apply mask progress '
                                 f'{self._apply_mask_progress}/{sum(self._ns)}')
                self._slice_end = self._slice_start + min(self._mask_step_size, sum(self._ns) - self._slice_start)
            else:
                self.logger.info(f'FedSVD Server Reporting Status: apply mask progress '
                                 f'{self._apply_mask_progress}/{self._m}')
                self._slice_end = self._slice_start + min(self._mask_step_size, self._m - self._slice_start)

            self._apply_mask_progress += (self._slice_end - self._slice_start)
            return {
                client_id: {
                    'vertical_slice': self._vertical_slice,
                    'fed_svd_status': self._fed_svd_status,
                    'slice_start': self._slice_start, 'slice_end': self._slice_end,
                    'apply_mask_finish': (self._vertical_slice and self._apply_mask_progress == sum(self._ns)) or \
                                         (not self._vertical_slice and self._apply_mask_progress == self._m),
                    'ns': sum(self._ns)
                }
                for client_id in self._client_ids_on_receiving
            }
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            if self._svd_mode == 'lr':
                # Vt.T @ np.diag(S / (S ** 2 + l2_regularization)) @ U.T @ y_train
                masked_parameters = self._masked_u.T @ self._masked_y
                masked_parameters = self._masked_vt.T @ np.diag(
                    self._sigma / (self._sigma ** 2 + ConfigurationManager().model_config.svd_lr_l2)
                ) @ masked_parameters
                # masked_parameters = self._masked_vt.T @ np.diag(
                #     self._sigma / (self._sigma ** 2 + ConfigurationManager().model_config.svd_lr_l2)
                # ) @ self._masked_u.T @ self._masked_y
                return {
                    client_id: {
                        'fed_svd_status': self._fed_svd_status,
                        'masked_parameters': masked_parameters
                    }
                    for client_id in self._client_ids_on_receiving
                }
            elif self._svd_mode == 'pca':
                return {
                    client_id: {
                        'fed_svd_status': self._fed_svd_status,
                        'masked_u': self._masked_u, 'sigma': self._sigma,
                        'masked_vt': self._masked_vt  # self._masked_vt should be None
                    }
                    for client_id in self._client_ids_on_receiving
                }
            else:
                if self._memory_map and ConfigurationManager().model_config.svd_top_k == -1:
                    self._vector_transfer_start = self._vector_transfer_progress
                    client_wise_download_info = {client_id: {} for client_id in self._client_ids_on_receiving}

                    for client_id in client_wise_download_info:
                        client_wise_download_info[client_id].update({
                            'fed_svd_status': self._fed_svd_status,
                            'm<n': self._m < sum(self._ns),
                            'vector_transfer_start': self._vector_transfer_start,
                        })

                        if self._m < sum(self._ns):
                            if self._vector_transfer_progress >= self._ns[client_id]:
                                client_wise_download_info[client_id].update({
                                    'masked_vt': None,
                                    'vector_transfer_end': None
                                })
                            else:
                                vector_end = min(self._ns[client_id], self._vector_transfer_start+self._mask_step_size)
                                left_slice = sum(self._ns[:client_id]) + self._vector_transfer_start
                                right_slice = sum(self._ns[:client_id]) + vector_end
                                client_wise_download_info[client_id].update({
                                    'masked_vt': self._masked_vt[:, left_slice:right_slice],
                                    'vector_transfer_end': vector_end
                                })
                            if self._vector_transfer_progress == 0:
                                client_wise_download_info[client_id].update({
                                    'masked_u': self._masked_u, 'sigma': self._sigma,
                                    'm': self._m, 'n': sum(self._ns)
                                })
                        else:
                            vector_end = self._vector_transfer_progress + min(self._mask_step_size, self._m)
                            client_wise_download_info[client_id].update({
                                'masked_u': self._masked_u[self._vector_transfer_start:vector_end],
                                'vector_transfer_end': vector_end,
                            })
                            if self._vector_transfer_progress == 0:
                                client_wise_download_info[client_id].update({
                                    'masked_vt': self._masked_vt[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])],
                                    'sigma': self._sigma,
                                    'm': self._m, 'n': sum(self._ns)
                                })
                    if self._m < sum(self._ns):
                        self._vector_transfer_progress += self._mask_step_size
                        self._vector_transfer_finish = (self._vector_transfer_progress >= np.array(self._ns)).all()
                    else:
                        self._vector_transfer_progress += (
                                client_wise_download_info[0]['vector_transfer_end'] - self._vector_transfer_start
                        )
                        self._vector_transfer_finish = self._vector_transfer_progress == self._m
                    for client_id in client_wise_download_info:
                        client_wise_download_info[client_id]['finish'] = self._vector_transfer_finish
                    return client_wise_download_info
                else:
                    return {
                        client_id: {
                            'fed_svd_status': self._fed_svd_status,
                            'masked_u': self._masked_u, 'sigma': self._sigma,
                            'masked_vt': self._masked_vt[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])]
                        }
                        for client_id in self._client_ids_on_receiving
                    }
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            return {client_id: {'fed_svd_status': self._fed_svd_status}
                    for client_id in self._client_ids_on_receiving}

    def update_host_params(self, client_params, *args):
        if self._fed_svd_status is FedSVDStatus.Init:

            # By default, the shape of the matrix is n * m
            client_params = sorted(client_params, key=lambda x: x['client_id'])
            ms = [e['mn'][0] for e in client_params]
            assert len(set(ms)) == 1, 'm is not the same among the clients'
            self._client_ids_on_receiving = [e['client_id'] for e in client_params]
            self.logger.info(f'Debug Client Ids on Receiving {self._client_ids_on_receiving}')
            self._ns = [e['mn'][1] for e in client_params]
            self._m = ms[0]
            self.logger.info(f'Data Shape m {self._m}, n {self._ns}')
            self._vertical_slice = sum(self._ns) > self._m
            self.logger.info(f'Vertical Slice {self._vertical_slice}')
            if client_params[0]['memory_map']:
                self._memory_map = True
                if self._vertical_slice:
                    self._pxq = np.memmap(
                        filename=os.path.join(self._tmp_dir, 'server_pxq.npy'),
                        dtype=np.float64, mode='write', shape=(sum(self._ns), self._m)
                    )
                else:
                    self._pxq = np.memmap(
                        filename=os.path.join(self._tmp_dir, 'server_pxq.npy'),
                        dtype=np.float64, mode='write', shape=(self._m, sum(self._ns))
                    )
            else:
                self._pxq = np.zeros([int(self._m), int(sum(self._ns))])
            self.logger.info(f'Debug server pxq shape {self._pxq.shape}')
            # Determine the step size when applying the masks
            if self._mini_batch_secure_agg:
                if self._vertical_slice:
                    if self._m <= 1e5:
                        self._mask_step_size = 1000
                    else:
                        self._mask_step_size = max(int(1000 / (self._m / 1e5)), 1)
                else:
                    base = 100000
                    step_size = 2000
                    if sum(self._ns) <= step_size:
                        self._mask_step_size = base
                    else:
                        self._mask_step_size = int(max(base / (sum(self._ns) / step_size), 1))
            else:
                self._mask_step_size = max(self._m, sum(self._ns))
            self._fed_svd_status = FedSVDStatus.SendMask
        elif self._fed_svd_status is FedSVDStatus.SendMask:
            self._fed_svd_status = FedSVDStatus.ApplyMask
        elif self._fed_svd_status is FedSVDStatus.ApplyMask:
            client_params = sorted(client_params, key=lambda x: x['client_id'])
            self.logger.info('Server Agg Started')
            start_time = time.time()
            if self._memory_map:
                if self._vertical_slice:
                    self._pxq[self._slice_start:self._slice_end] = \
                        np.sum([e['secure_agg'] for e in client_params], axis=0, dtype=np.float64).T
                else:
                    self._pxq[self._slice_start:self._slice_end] = \
                        np.sum([e['secure_agg'] for e in client_params], axis=0, dtype=np.float64)
            else:
                if self._vertical_slice:
                    self._pxq[:, self._slice_start:self._slice_end] = \
                        np.sum([e['secure_agg'] for e in client_params], axis=0, dtype=np.float64)
                else:
                    self._pxq[self._slice_start:self._slice_end] = \
                        np.sum([e['secure_agg'] for e in client_params], axis=0, dtype=np.float64)
            self.logger.info(f'Server Agg Finished. Using {time.time() - start_time}')
            apply_mask_finish = (self._vertical_slice and self._apply_mask_progress == sum(self._ns)) or \
                                (not self._vertical_slice and self._apply_mask_progress == self._m)
            if apply_mask_finish and self._svd_mode == 'lr':
                self._masked_y = client_params[-1].get('masked_y')
            if apply_mask_finish and self._svd_mode == 'svd':
                self._client_masked_q = {e['client_id']: e['masked_q'] for e in client_params}
            del client_params
            if apply_mask_finish:
                self.logger.info('Server received all masked data. Proceed to SVD.')
                self._fed_svd_status = FedSVDStatus.Factorization
                if self._memory_map:
                    if self._vertical_slice:
                        self._masked_u, self._sigma, self._masked_vt = self._server_svd(self._pxq.T)
                    else:
                        self._masked_u, self._sigma, self._masked_vt = self._server_svd(self._pxq)
                    pxq_filename = self._pxq.filename
                    del self._pxq
                    os.remove(pxq_filename)
                else:
                    self._masked_u, self._sigma, self._masked_vt = self._server_svd(self._pxq)
                    del self._pxq
                if self._svd_mode == 'svd':
                    # Compute V @ [Q*R]
                    client_masked_q = None
                    for client_id in self._client_ids_on_receiving:
                        masked_q = self._client_masked_q[client_id]
                        if client_masked_q is None:
                            client_masked_q = [e[-1] for e in masked_q]
                        else:
                            if masked_q[0][-1].shape[0] != masked_q[0][-1].shape[1]:
                                client_masked_q[-1] = np.concatenate([
                                    client_masked_q[-1], masked_q[0][-1]
                                ], axis=-1)
                                client_masked_q += [e[-1] for e in masked_q[1:]]
                            else:
                                client_masked_q += [e[-1] for e in masked_q]
                    v_counter = 0
                    for data in client_masked_q:
                        self._masked_vt[:, v_counter:v_counter + data.shape[0]] = \
                            self._masked_vt[:, v_counter:v_counter + data.shape[0]] @ data
                        v_counter += data.shape[0]
                self._fed_svd_status = FedSVDStatus.RemoveMask
        elif self._fed_svd_status is FedSVDStatus.RemoveMask:
            if (self._memory_map and self._vector_transfer_finish) or (not self._memory_map) or\
                    (ConfigurationManager().model_config.svd_top_k != -1) or self._svd_mode == 'lr':
                del self._masked_u
                del self._masked_vt
                del self._sigma
                self._fed_svd_status = FedSVDStatus.Evaluate
        elif self._fed_svd_status is FedSVDStatus.Evaluate:
            # FedSVD Finished
            self._stop = True
            if self._evaluate_for_debug:
                # Only for evaluation, and the clients should not upload local results in real applications
                client_params = sorted(client_params, key=lambda x: x['client_id'])
                if self._svd_mode == 'lr':
                    self._evaluate_parameters = np.concatenate([e['parameters'] for e in client_params], axis=0)
                else:
                    self._evaluation_u = client_params[0]['u']
                    self._evaluation_sigma = client_params[0]['sigma']
                    if self._svd_mode == 'svd':
                        self._evaluation_vt = np.concatenate([e['vt'] for e in client_params], axis=-1)
            client_memory_usage = [e['memory_usage'] for e in client_params]
            for date, usage in sorted(self._process_memory_usage.items(), key=lambda x: parse(x[0])):
                tmp = usage
                for cmu in client_memory_usage:
                    tmp += cmu.get(date, 0)
                self._total_memory_usage.append(tmp)
            # Release memory
            del client_params

    def safe_memmap_matmul(self, a, b, step_size=1000):
        self.logger.info(f'safe_memmap_matmul {a.shape} {b.shape}')
        result = np.zeros([a.shape[0], b.shape[1]])
        for i in range(0, a.shape[0], step_size):
            st = time.time()
            np.matmul(a[i:i+step_size], b, out=result[i:i+step_size])
            self.logger.info(f'Iter {i} Cost {time.time() - st}')
        return result

    def _server_svd(self, data_matrix):

        if not self._memory_map:
            if ConfigurationManager().model_config.svd_top_k == -1:
                # We do not need to compute the full matrices when the matrix is not full rank
                self.logger.info(f'Server running SVD on matrix {data_matrix.shape[0]}x{data_matrix.shape[1]}')
                m, n = data_matrix.shape
                if m <= n:
                    x2 = data_matrix @ data_matrix.T
                    left, sigma_square, _ = np.linalg.svd(x2, full_matrices=False)
                    return left, sigma_square**0.5, np.diag(sigma_square ** -0.5) @ left.T @ data_matrix
                else:
                    x2 = data_matrix.T @ data_matrix
                    right_t, sigma_square, _ = np.linalg.svd(x2, full_matrices=False)
                    return data_matrix @ right_t @ np.diag(sigma_square ** -0.5), sigma_square**0.5, right_t.T
                # return np.linalg.svd(data_matrix, full_matrices=False)
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
            self.logger.info(f'Server running SVD Attempt {m}x{n}')
            if m < n:
                x2 = self.safe_memmap_matmul(data_matrix, data_matrix.T)
            else:
                left, sigma, right = self._server_svd(data_matrix.T)
                if right is not None:
                    return right.T, sigma, left.T
                else:
                    return None, sigma, left.T

            if ConfigurationManager().model_config.svd_top_k == -1:
                # We do not need to compute the full matrices when the matrix is not full rank
                self.logger.info(f'Server running SVD on matrix {m}x{n}')
                left, sigma, _ = np.linalg.svd(x2, full_matrices=False)
                sigma = sigma ** 0.5
                step_size = 10000
                right = np.memmap(
                    filename=os.path.join(self._tmp_dir, 'server_large_vector.npy'), mode='write',
                    dtype=np.float64, shape=(n, len(sigma))
                )
                left_mul = np.diag(sigma ** -1) @ left.T
                for i in range(0, data_matrix.shape[1], step_size):
                    index_end = min(i+step_size, data_matrix.shape[1])
                    right[i:index_end] = (left_mul @ data_matrix[:, i:index_end]).T
                # right = np.diag(sigma ** -1) @ left.T @ data_matrix
                return left, sigma, right.T
            elif ConfigurationManager().model_config.svd_top_k > 0:
                truncated_svd = TruncatedSVD(
                    n_components=ConfigurationManager().model_config.svd_top_k, algorithm='arpack',
                )
                # By default, we firstly compute the left truncated singular vectors
                truncated_svd.fit(x2)
                left = truncated_svd.components_.T
                sigma = truncated_svd.singular_values_ ** 0.5
                if self._svd_mode == 'pca':
                    return left, sigma, None
                else:
                    right = np.diag(sigma ** -1) @ left.T @ data_matrix
                    return left, sigma, right

    def set_host_params_to_local(self, host_params, **kwargs):
        self._current_status = host_params.get('fed_svd_status')
        self.logger.info(f'Client {self.client_id} received server commend: {self._current_status}')
        if self._current_status is FedSVDStatus.Init:
            self._server_machine_id = host_params['server_machine_id']
        elif self._current_status is FedSVDStatus.SendMask:
            if self._received_p_masks is None:
                if self._block_based_optimization:
                    self.logger.info(f'Client {self.client_id} is generating random mask P')
                    self._received_p_masks = host_params['msg_of_p']
                    self._received_p_masks = generate_orthogonal_matrix(
                        n=self._local_m,
                        block_reduce=min(ConfigurationManager().model_config.block_size, self._local_m),
                        random_seed=self._received_p_masks, file_name=f'client_{self.client_id}',
                        reuse=self._save_p_to_disk,
                        offload_to_disk=self._save_p_to_disk
                    )
                else:
                    self._received_p_masks = host_params['msg_of_p']
            self._received_q_masks = host_params['sliced_q']
            self._received_q_masks = sorted(self._received_q_masks, key=lambda x: x[0])
            self._global_ns = host_params['ns']
            self.logger.info(f'Client {self.client_id} has received random mask Q')
        elif self._current_status is FedSVDStatus.ApplyMask:
            self._received_apply_mask_params = host_params
        elif self._current_status is FedSVDStatus.RemoveMask:
            if self._svd_mode == 'lr':
                self._masked_parameters = host_params['masked_parameters']
            else:
                if self._memory_map and ConfigurationManager().model_config.svd_top_k == -1:
                    self._received_server_params = host_params
                else:
                    self._masked_u = host_params['masked_u']
                    self._sigma = host_params['sigma']
                    self._masked_vt = host_params['masked_vt']

    def _generate_masked_data_in_secure_agg(self, shape, base=None):
        if base is not None:
            assert shape[0] == base.shape[0] and shape[1] == base.shape[1]
            result = base
        else:
            result = np.zeros(shape)
        # Apply the secure aggregation
        for j in range(self._secure_agg_random_seed.shape[1]):
            if self._client_id == j:
                continue
            else:
                np.random.seed(self._secure_agg_random_seed[self._client_id, j])
                r1 = np.random.random(shape) * 2 - 1
                r2 = np.random.random(shape) * 2 - 1
                if self._client_id < j:
                    p = r1 - r2
                else:
                    p = r2 - r1
                result += p
                del r1, r2, p
        return result

    def fit_on_local_data(self):
        if self._current_status is FedSVDStatus.ApplyMask:
            slice_start = self._received_apply_mask_params.get('slice_start')
            slice_end = self._received_apply_mask_params.get('slice_end')
            vertical_slice = self._received_apply_mask_params.get('vertical_slice')
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
                    if self._save_p_to_disk:
                        with open(self._received_p_masks[i], 'rb') as f:
                            tmp_block_mask = pickle.load(f)
                    else:
                        tmp_block_mask = self._received_p_masks[i]
                    self._p_times_x[p_size_counter:p_size_counter+len(tmp_block_mask)] =\
                        tmp_block_mask @ self.train_data['x'][p_size_counter: p_size_counter + len(tmp_block_mask)]
                    if self._svd_mode == 'lr' and self._is_active:
                        self._masked_y.append(
                            tmp_block_mask @ self.train_data['y'][p_size_counter: p_size_counter + len(tmp_block_mask)]
                        )
                    p_size_counter += len(tmp_block_mask)
                    if self._save_p_to_disk:
                        del tmp_block_mask
                if self._svd_mode == 'lr' and self._is_active:
                    self._masked_y = np.concatenate(self._masked_y, axis=0)
                    self.logger.info(f'masked y shape {self._masked_y.shape}')
                if not self._memory_map and not self._evaluate_for_debug:
                    del self.train_data
                if self._memory_map:
                    if not self._evaluate_for_debug:
                        tmp_file_name = self.train_data['x'].filename
                        del self.train_data
                        os.remove(tmp_file_name)
            
            self.logger.info(f'Client {self.client_id} SecureAgg from {slice_start} to {slice_end}')

            if vertical_slice:

                min_index = self._received_q_masks[0][1]
                max_index = self._received_q_masks[-1][1] + self._received_q_masks[-1][-1].shape[-1]

                if slice_start >= max_index or slice_end <= min_index:
                    self.logger.info(f'Client {self.client_id} SecAgg using zero arrays')
                    start_time = time.time()
                    self._sliced_pxq_with_secure_agg = self._generate_masked_data_in_secure_agg(
                        shape=(self._local_m, slice_end - slice_start))
                    self.logger.info(f'Client {self.client_id} SecAgg using {time.time() - start_time}')
                else:
                    self.logger.info(f'Client {self.client_id} SecAgg using real data')
                    start_time = time.time()
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
                            matrix_base.append(
                                self._p_times_x[:, x_slice_index:x_slice_index+self._received_q_masks[j][-1].shape[0]]@
                                self._received_q_masks[j][-1]
                                    [:, counter - self._received_q_masks[j][1]:
                                                min(slice_end - self._received_q_masks[j][1],
                                                    self._received_q_masks[j][-1].shape[-1])]
                            )
                        else:
                            matrix_base.append(np.zeros([self._local_m, slice_end - max_index]))
                        counter += matrix_base[-1].shape[-1]
                    matrix_base = np.concatenate(matrix_base, axis=-1)
                    self._sliced_pxq_with_secure_agg = self._generate_masked_data_in_secure_agg(
                        shape=[self._local_m, slice_end - slice_start], base=matrix_base
                    )
                    self.logger.info(f'Client {self.client_id} SecAgg using {time.time() - start_time}')
            
            else:
                start_time = time.time()
                matrix_base = []
                if self._received_q_masks[0][1] > 0:
                    matrix_base.append(np.zeros([slice_end-slice_start, self._received_q_masks[0][1]]))
                init_n_slice = self._received_q_masks[0][0]
                for i in range(len(self._received_q_masks)):
                    xi, yi, qi = self._received_q_masks[i]
                    matrix_base.append(
                        self._p_times_x[slice_start:slice_end, xi-init_n_slice:xi-init_n_slice+qi.shape[0]] @ qi
                    )
                if self._received_q_masks[-1][1] < self._global_ns:
                    matrix_base.append(np.zeros([
                        slice_end-slice_start,
                        self._global_ns - self._received_q_masks[-1][1] - self._received_q_masks[-1][-1].shape[-1]
                    ]))
                matrix_base = np.concatenate(matrix_base, axis=-1)
                self._sliced_pxq_with_secure_agg = self._generate_masked_data_in_secure_agg(
                    shape=matrix_base.shape, base=matrix_base
                )
                self.logger.info(f'Client {self.client_id} SecAgg using {time.time() - start_time}')
                
            if self._received_apply_mask_params['apply_mask_finish']:
                # Release memory and disk
                if self._memory_map:
                    tmp_file = self._p_times_x.filename
                    del self._p_times_x
                    os.remove(tmp_file)
                else:
                    del self._p_times_x

                # Apply mask to Q_i (Preparing for removing the mask)
                # self._q_random_mask = []
                self._q_random_mask_inverse = []
                self._masked_q = []
                for x, y, q_slice in self._received_q_masks:
                    random_mask_size, _ = q_slice.shape
                    random_mask_block = np.random.random([random_mask_size, random_mask_size])
                    # self._q_random_mask.append(random_mask_block)
                    self._q_random_mask_inverse.append(np.linalg.inv(random_mask_block))
                    self._masked_q.append([y, x, q_slice.T @ random_mask_block])

        elif self._current_status is FedSVDStatus.RemoveMask:

            if self._svd_mode == 'lr':
                self._local_parameters = []
                for x, y, data in self._received_q_masks:
                    self._local_parameters.append(data @ self._masked_parameters[y:y+data.shape[1]])
                del self._masked_parameters
                self._local_parameters = np.concatenate(self._local_parameters, axis=0)
            else:
                if self._memory_map and ConfigurationManager().model_config.svd_top_k == -1:
                    vector_transfer_start = self._received_server_params['vector_transfer_start']
                    vector_transfer_end = self._received_server_params['vector_transfer_end']
                    transfer_finish = self._received_server_params['finish']
                    if self._received_server_params['m<n']:
                        if vector_transfer_start == 0:
                            self._sigma = self._received_server_params['sigma']
                            self._masked_u = self._received_server_params['masked_u']
                            self._masked_vt = np.memmap(
                                filename=os.path.join(self._tmp_dir, f'client_{self.client_id}_masked_vt.npy'),
                                mode='write', dtype=np.float64,
                                shape=(self._received_server_params['m'], self._local_n)
                            )
                        self._masked_vt[:, vector_transfer_start:vector_transfer_end] =\
                            self._received_server_params['masked_vt']
                    else:
                        if vector_transfer_start == 0:
                            self._sigma = self._received_server_params['sigma']
                            self._masked_vt = self._received_server_params['masked_vt']
                            self._masked_u = np.memmap(
                                filename=os.path.join(self._tmp_dir, f'client_{self.client_id}_masked_u.npy'),
                                mode='write', dtype=np.float64,
                                shape=(self._received_server_params['m'], self._received_server_params['n'])
                            )
                        self._masked_u[vector_transfer_start:vector_transfer_end] = \
                            self._received_server_params['masked_u']
                    if not transfer_finish:
                        return None

                self.logger.info(f'Client {self.client_id} is removing masks')

                # Remove the mask of U and
                if self._memory_map and ConfigurationManager().model_config.svd_top_k == -1 \
                        and not self._received_server_params.get('m<n'):
                    # Overwrite the disk
                    self._u = self._masked_u
                    # np.memmap(
                    #     filename=os.path.join(self._tmp_dir, f'client_{self.client_id}_u.npy'),
                    #     mode='write', dtype=np.float64,
                    #     shape=self._masked_u.shape
                    # )
                else:
                    self._u = np.zeros(self._masked_u.shape)
                p_size_counter = 0
                for i in range(len(self._received_p_masks)):
                    if self._save_p_to_disk:
                        with open(self._received_p_masks[i], 'rb') as f:
                            tmp_block_mask = pickle.load(f)
                    else:
                        tmp_block_mask = self._received_p_masks[i]
                    self._u[p_size_counter: p_size_counter + tmp_block_mask.shape[0]] = \
                        tmp_block_mask.T @ self._masked_u[p_size_counter: p_size_counter + tmp_block_mask.shape[0]]
                    p_size_counter += tmp_block_mask.shape[1]
                    if self._save_p_to_disk:
                        del tmp_block_mask

                if self._svd_mode == 'svd':
                    # Remove the mask of VT, Overwrite the disk
                    if self._memory_map and ConfigurationManager().model_config.svd_top_k == -1 \
                            and self._received_server_params.get('m<n'):
                        self._local_vt = self._masked_vt
                        # np.memmap(
                        #     filename=os.path.join(self._tmp_dir, f'client_{self.client_id}_vt.npy'),
                        #     mode='write', dtype=np.float64,
                        #     shape=(self._masked_vt.shape[0], self._local_n)
                        # )
                    else:
                        self._local_vt = np.zeros([self._masked_vt.shape[0], self._local_n])

                    v_counter = 0
                    for data in self._q_random_mask_inverse:
                        self._local_vt[:, v_counter:v_counter + data.shape[0]] = \
                            self._masked_vt[:, v_counter:v_counter + data.shape[0]] @ data
                        v_counter += data.shape[0]
                del self._masked_u
                del self._masked_vt
            # Release the memory
            del self._received_p_masks
            del self._received_q_masks
            self.logger.info(f'Client {self.client_id} has finished removing masks')

    def local_evaluate(self):
        # Warning: The following evaluation need self.train_data, which is deleted when self.evaluate
        if self._current_status is FedSVDStatus.Evaluate and self._evaluate_for_debug:
            if self._svd_mode == 'svd' and \
                    ConfigurationManager().model_config.svd_top_k == -1:
                recovered_data = self._u @ np.diag(self._sigma) @ self._local_vt
                reconstruct_mae_error = np.mean(np.abs(recovered_data - self.train_data['x']))
                reconstruct_rmse_error = np.sqrt(np.mean(np.square(recovered_data - self.train_data['x'])))
                self.logger.info(f'Client {self.client_id} Test RMSE {reconstruct_rmse_error}')
                # self.logger.info(f'Local Singular Values {self._sigma[:10]}')
                # self.logger.info(f'Local Singular Vectors {self._local_vt}')
                return {
                    'test_mae': reconstruct_mae_error,
                    'test_rmse': reconstruct_rmse_error,
                    'test_size': self._local_n,
                    'val_loss': reconstruct_rmse_error,
                    'val_size': self._local_n,
                }
        return None

    def client_exit_job(self, client):
        self._scheduler.remove_job('log_memory')
        # Clear the cached orthogonal matrices
        generate_orthogonal_matrix(clear_cache=True)
        # Clear disk files
        local_disk_files = [e for e in os.listdir(self._tmp_dir) if e.startswith(f'client_{self.client_id}')]
        for file in local_disk_files:
            try:
                os.remove(os.path.join(self._tmp_dir, file))
            except FileNotFoundError:
                self.logger.info(f'{file} not found, cache clean failed')

    def retrieve_local_upload_info(self):
        if self._current_status is FedSVDStatus.Init:
            if type(self.train_data['x']) is str:
                self.train_data['x'] = open_memmap(self.train_data['x'])
                self._memory_map = True
                self._save_p_to_disk = True
            else:
                self._memory_map = False
                self._save_p_to_disk = False
            self.train_data['x'] = self.train_data['x'].T
            if self._svd_mode == 'lr':
                self._is_active = (self.client_id + 1) == ConfigurationManager().runtime_config.client_num
            self._local_m, self._local_n = self.train_data['x'].shape
            return {'client_id': self.client_id, 'mn': self.train_data['x'].shape, 'memory_map': self._memory_map}
        elif self._current_status is FedSVDStatus.ApplyMask:
            if self._received_apply_mask_params['apply_mask_finish']:
                if self._svd_mode == 'lr' and self._is_active:
                    return {
                        'client_id': self._client_id,
                        'secure_agg': self._sliced_pxq_with_secure_agg, 'masked_y': self._masked_y
                    }
                if self._svd_mode == 'svd':
                    return {
                        'client_id': self._client_id,
                        'secure_agg': self._sliced_pxq_with_secure_agg, 'masked_q': self._masked_q
                    }
            return {'client_id': self._client_id, 'secure_agg': self._sliced_pxq_with_secure_agg}
        elif self._current_status is FedSVDStatus.Evaluate:
            evaluate_results = {}
            # Only for debugging, and the clients should not upload local results in real applications
            if self._evaluate_for_debug:
                if self._svd_mode == 'lr':
                    evaluate_results = {'client_id': self._client_id, 'parameters': self._local_parameters}
                else:
                    if self.client_id == 0:
                        evaluate_results = {
                            'client_id': self._client_id, 'u': self._u, 'sigma': self._sigma, 'vt': self._local_vt
                        }
                    else:
                        evaluate_results = {'client_id': self._client_id, 'vt': self._local_vt}
            evaluate_results['memory_usage'] = self._process_memory_usage
            return evaluate_results

    def host_exit_job(self, host):

        self._scheduler.remove_job('log_memory')

        self.logger.info(f'Total Memory Usage: {np.max(self._total_memory_usage)} GB ')

        result_json = host.snapshot_result(None)
        result_json.update({'memory_usage_sequence': self._total_memory_usage})

        result_json.update({
            'block_based_optimization': self._block_based_optimization,
            'mini_batch_secure_agg': self._mini_batch_secure_agg,
            'memory_map': self._memory_map
        })

        self.logger.info(f'FedSVD Version block_based_optimization: {self._block_based_optimization}')
        self.logger.info(f'FedSVD Version mini_batch_secure_agg: {self._mini_batch_secure_agg}')
        self.logger.info(f'FedSVD Version memory_map: {self._memory_map}')

        with open(os.path.join(host.log_dir, 'results.json'), 'w') as f:
            json.dump(result_json, f)

        if self._evaluate_for_debug:
            
            self.logger.info('FedSVD Server Status: Computing the metrics.')

            # Clear disk files
            local_disk_files = [e for e in os.listdir(self._tmp_dir) if e.startswith(f'server')]
            for file in local_disk_files:
                try:
                    os.remove(os.path.join(self._tmp_dir, file))
                except FileNotFoundError:
                    self.logger.info(f'{file} not found, cache clean failed')

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

            # Load clients' data for evaluation
            if self._memory_map:
                x_train = np.memmap(
                    filename=os.path.join(self._tmp_dir, 'server_x_train.npy'),
                    mode='write', dtype=np.float64, shape=(self._m, sum(self._ns))
                )
            else:
                x_train = np.zeros((self._m, sum(self._ns)))
            for client_id in range(client_num):
                with open(os.path.join(data_dir, 'client_%s.pkl' % client_id), 'r') as f:
                    data = hickle.load(f)
                if self._memory_map:
                    x_train[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])] = open_memmap(data['x_train']).T
                    # x_train.flush()
                else:
                    x_train[:, sum(self._ns[:client_id]):sum(self._ns[:client_id+1])] = data['x_train'].T
                self.logger.info(f'Memory Usage after loading Client '
                                f'{client_id} {psutil.virtual_memory().used / 2 ** 30}GB')
                if self._svd_mode == 'lr' and client_id == (client_num - 1):
                    y_train = data['y_train']

            if self._svd_mode == 'lr':
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
                batch_size = 1000000
                max_epoch = 10000
                l2 = ConfigurationManager().model_config.svd_lr_l2
                linear_regression_model = tf.keras.Sequential()
                linear_regression_model.add(tf.keras.layers.Dense(
                    1 if len(y_train.shape) == 1 else y_train.shape[-1],
                    kernel_regularizer=tf.keras.regularizers.l2(l2), use_bias=True
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

                # Filter out the useless singular values according to the rank of the matrix
                m, n = x_train.shape
                if m < n:
                    xx = x_train @ x_train.T
                else:
                    xx = x_train.T @ x_train
                rank_xx = np.linalg.matrix_rank(xx)
                if rank_xx < len(c_sigma):
                    c_sigma = c_sigma[:rank_xx]
                    c_u = c_u[:, :rank_xx]
                    self._evaluation_sigma = self._evaluation_sigma[:rank_xx]
                    self._evaluation_u = self._evaluation_u[:, :rank_xx]
                    if self._svd_mode == 'svd':
                        c_vt = c_vt[:rank_xx, :]
                        self._evaluation_vt = self._evaluation_vt[:rank_xx, :]

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

            # Save again
            with open(os.path.join(host.log_dir, 'results.json'), 'w') as f:
                json.dump(result_json, f)

            result_path = os.path.join(host.log_dir, 'results.json')
            self.logger.info(f'FedSVD Server Status: Results saved to {result_path}.')

        # Clear cached data
        shutil.rmtree(ConfigurationManager().data_config.dir_name, ignore_errors=True)
        # Clear disk files
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
