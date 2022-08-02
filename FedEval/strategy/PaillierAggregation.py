import pdb
import enum
import random
import time

import gmpy2
from phe.paillier import PaillierPublicKey, PaillierPrivateKey
import numpy as np
import tensorflow as tf

from functools import reduce, partial

from ..callbacks import *
from ..model import *
from ..utils import ParamParser
from .FederatedStrategy import FedStrategy, HostParamsType
from ..config.configuration import ConfigurationManager, Role
from ..aggregater import aggregate_weighted_average
from ..secure_protocols import ShamirSecretSharing, GaloisFieldNumber, aes_gcm_decrypt, \
    aes_gcm_encrypt, GaloisFieldParams

from cryptography.hazmat.primitives.asymmetric import dh


def get_prime_over(N, seed=None):
    if seed:
        random.seed(seed)
        rand_func = random
    else:
        rand_func = random.SystemRandom()
    r = gmpy2.mpz(rand_func.getrandbits(N))
    r = gmpy2.bit_set(r, N - 1)
    return int(gmpy2.next_prime(r))


def generate_paillier_keypair(seed=None, private_keyring=None, n_length=2048):
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = get_prime_over(n_length // 2, seed=seed)
        q = p
        while q == p:
            seed += 1
            q = get_prime_over(n_length // 2, seed=seed)
        n = p * q
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    if private_keyring is not None:
        private_keyring.add(private_key)

    return public_key, private_key


class PAStatus(enum.Enum):
    Init = 'Init'
    DHKeyAgree = 'KeyAgree'
    UpdateWeights = 'UpdateWeights'


class PaillierAggregation(FedStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cfg = ConfigurationManager()
        if cfg.role is Role.Server:
            self._server_status = PAStatus.Init
            self.host_params_type = HostParamsType.Personalized
            self._received_pk = None
        else:
            self._paillier_key_length = 1024
            self._client_status = None
            self._num_participants = None
            self._received_p = None
            self._received_g = None
            self._dh_sk = None
            self._dh_pk = None
            self._peer_pk = None
            self._final_pk = None
            self._paillier_pk, self._paillier_sk = None, None
            self._key_exchange_counter = None
            self._params_shape = None

    def host_select_train_clients(self, ready_clients):
        if self._server_status is PAStatus.Init:
            self.train_selected_clients = ready_clients
            return self.train_selected_clients
        elif self._server_status is PAStatus.DHKeyAgree:
            return self.train_selected_clients
        elif self._server_status is PAStatus.UpdateWeights:
            return super(PaillierAggregation, self).host_select_train_clients(ready_clients)

    def retrieve_host_download_info(self):
        if self._server_status is PAStatus.Init:
            ka_parameters = dh.generate_parameters(generator=2, key_size=2048)
            self._server_status = PAStatus.DHKeyAgree
            return {
                client_id: {
                    'status': PAStatus.Init, 'p': ka_parameters.parameter_numbers().p,
                    'g': ka_parameters.parameter_numbers().g, 'num_participants': len(self.train_selected_clients)
                } for client_id in self.train_selected_clients
            }
        elif self._server_status is PAStatus.DHKeyAgree:
            return {
                self.train_selected_clients[i]: {
                    'status': self._server_status,
                    'pk': self._received_pk[self.train_selected_clients[(i + 1) % len(self.train_selected_clients)]]
                }
                for i in range(len(self.train_selected_clients))
            }
        elif self._server_status is PAStatus.UpdateWeights:
            return {cid: {'status': PAStatus.UpdateWeights, 'aggregated_weights': self.host_params}
                    for cid in self.train_selected_clients}
        else:
            raise NotImplementedError

    def update_host_params(self, client_params, aggregate_weights) -> None:
        if self._server_status is PAStatus.DHKeyAgree:
            if type(client_params[0]) is dict:
                self._received_pk = {}
                for record in client_params:
                    self._received_pk.update(record)
            else:
                self.host_params = aggregate_weighted_average(client_params, aggregate_weights)
                self._server_status = PAStatus.UpdateWeights

    def set_host_params_to_local(self, host_params, current_round: int):
        self._client_status = host_params['status']
        if self._client_status is PAStatus.Init:
            self._num_participants = host_params['num_participants']
            self._received_p = host_params['p']
            self._received_g = host_params['g']
            self._dh_sk = random.SystemRandom().randint(0, self._received_p)
            self._dh_pk = gmpy2.powmod(self._received_g, self._dh_sk, self._received_p)
            self._key_exchange_counter = 0
        elif self._client_status is PAStatus.DHKeyAgree:
            self._peer_pk = gmpy2.powmod(host_params['pk'], self._dh_sk, self._received_p)
            self._key_exchange_counter += 1
            if self._key_exchange_counter == (self._num_participants - 1):
                self._client_status = PAStatus.UpdateWeights
        elif self._client_status is PAStatus.UpdateWeights:
            enc_aggregated_weights = host_params['aggregated_weights']
            dec_func = np.vectorize(self._paillier_sk.decrypt)
            aggregated_weights = [dec_func(e) for e in enc_aggregated_weights]
            self.ml_model.set_weights(aggregated_weights)

    def fit_on_local_data(self):
        if self._client_status is PAStatus.UpdateWeights:
            return super(PaillierAggregation, self).fit_on_local_data()

    def retrieve_local_upload_info(self):
        if self._client_status is PAStatus.Init:
            return {self.client_id: self._dh_pk}
        elif self._client_status is PAStatus.DHKeyAgree:
            return {self.client_id: self._peer_pk}
        elif self._client_status is PAStatus.UpdateWeights:
            if self._paillier_pk is None or self._paillier_sk is None:
                self._paillier_pk, self._paillier_sk = generate_paillier_keypair(
                    self._peer_pk, n_length=self._paillier_key_length)
            encrypt_func = np.vectorize(self._paillier_pk.encrypt)
            upload_info = super(PaillierAggregation, self).retrieve_local_upload_info()
            upload_info = [encrypt_func(e.astype(float)) for e in upload_info]
            return upload_info
