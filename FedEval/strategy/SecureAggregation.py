import enum
import random
import numpy as np
import tensorflow as tf

from functools import reduce, partial

from ..callbacks import *
from ..model import *
from ..utils import ParamParser
from .FederatedStrategy import FedStrategy, HostParamsType
from ..config.configuration import ConfigurationManager, Role
from ..secure_protocols import ShamirSecretSharing, GaloisFieldNumber, aes_gcm_decrypt, aes_gcm_encrypt

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
# serialization
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat, ParameterFormat
from cryptography.hazmat.primitives.serialization import load_der_parameters
from cryptography.hazmat.primitives.serialization import load_der_public_key


class SAStatus(enum.Enum):
    Init = 'Init'
    DHKeyAgree = 'KeyAgree'
    ApplyMask = 'ApplyMask'
    UpdateWeights = 'UpdateWeights'


class SecureAggregation(FedStrategy):

    def __init__(self, param_parser=ParamParser):
        super().__init__(param_parser)

        config_manager = ConfigurationManager()
        if config_manager.role == Role.Server:
            # default values of p, m, n
            self._p = "2**521-1"
            self._n = config_manager.num_of_train_clients_contacted_per_round
            self._m = self._n
            self._server_status = SAStatus.Init
            self._host_params_type = HostParamsType.Personalized
            self._received_peer_dh_pk = None
            self._encrypted_shares = None
        if config_manager.role == Role.Client:
            self._client_status = None
            self._m = None
            self._n = None
            self._p = None
            self._received_pp = None
            self._client_dh_maks_sk = None
            self._client_dh_mask_pk = None
            self._client_dh_aes_sk = None
            self._client_dh_aes_pk = None
            self._peer_dh_pk = None
            self._local_mask_seed = None
            self._peer_shares = None

        self._sss: ShamirSecretSharing = None
        self._total_training_samples = None

    def host_get_init_params(self):
        self._server_status = SAStatus.Init
        return self._retrieve_host_download_info()

    def host_select_train_clients(self, ready_clients):
        if self._server_status is SAStatus.Init:
            return super(SecureAggregation, self).host_select_train_clients(ready_clients)
        else:
            return self.train_selected_clients

    def _retrieve_host_download_info(self):
        self.logger.info("#" * 20)
        self.logger.info(f'Server report round {self.current_round} status {self._server_status}')
        self.logger.info("#" * 20)
        if self._server_status is SAStatus.Init:
            ka_parameters = dh.generate_parameters(generator=2, key_size=2048)
            ka_param_bytes = ka_parameters.parameter_bytes(Encoding.DER, ParameterFormat.PKCS3)
            return {
                client_id: {
                    'n': self._n, 'm': self._m, 'p': self._p, 'pp': ka_param_bytes, 'status': self._server_status}
                for client_id in self.train_selected_clients
            }
        elif self._server_status is SAStatus.DHKeyAgree:
            return {
                client_id: {
                    'status': self._server_status,
                    'peer_dh_pk': [e for e in self._received_peer_dh_pk if e['client_id'] != client_id]}
                for client_id in self.train_selected_clients
            }
        elif self._server_status is SAStatus.ApplyMask:
            download_info = {
                cid: {'status': self._server_status, 'encrypted_shares': [],
                      'num_training_samples': self._total_training_samples}
                for cid in self.train_selected_clients
            }
            for record in self._encrypted_shares:
                download_info[int(record[0].split('||')[-1])]['encrypted_shares'].append(record)
            return download_info
        elif self._server_status is SAStatus.UpdateWeights:
            return {
                cid: {'weights': self.ml_model.get_weights(), 'status': SAStatus.UpdateWeights}
                for cid in self.train_selected_clients
            }
        else:
            raise NotImplementedError

    def update_host_params(self, client_params, aggregate_weights):
        if self._server_status is SAStatus.Init:
            self._received_peer_dh_pk = sorted(client_params, key=lambda x: x['client_id'])
            self._total_training_samples = sum([e['num_training_samples'] for e in client_params])
            self._server_status = SAStatus.DHKeyAgree
        elif self._server_status is SAStatus.DHKeyAgree:
            self._encrypted_shares = reduce(lambda a, b: a + b, client_params)
            self._server_status = SAStatus.ApplyMask
        elif self._server_status is SAStatus.ApplyMask:
            aggregated_weights = []
            for i in range(len(client_params[0])):
                aggregated_weights.append(np.sum([e[i] for e in client_params], axis=0))
            decode = np.vectorize(lambda x: x.decode())
            aggregated_weights = [decode(e) for e in aggregated_weights]
            self.ml_model.set_weights(aggregated_weights)
            self._server_status = SAStatus.UpdateWeights
        elif self._server_status is SAStatus.UpdateWeights:
            self._server_status = SAStatus.Init
        else:
            raise NotImplementedError
        return self._retrieve_host_download_info()

    def set_host_params_to_local(self, host_params, current_round: int):
        self._client_status = host_params['status']
        self.logger.info(f'Client {self.client_id} report status {self._client_status}')
        if self._client_status is SAStatus.Init:
            self._m = host_params['m']
            self._n = host_params['n']
            self._p = eval(host_params['p'])
            self._received_pp = host_params['pp']
            self._local_mask_seed = random.SystemRandom().randint(0, self._p)
            self._sss = ShamirSecretSharing(m=self._m, n=self._n, p=self._p)
        elif self._client_status is SAStatus.DHKeyAgree:
            self._peer_dh_pk = host_params['peer_dh_pk']
        elif self._client_status is SAStatus.ApplyMask:
            self._total_training_samples = host_params['num_training_samples']
            tmp_cid_key = {e['client_id']: e['derived_aes_key'] for e in self._peer_dh_pk}
            for record in host_params['encrypted_shares']:
                self._peer_shares.append(aes_gcm_decrypt(tmp_cid_key[int(record[0].split('||')[0])], record)[-1])
        elif self._client_status is SAStatus.UpdateWeights:
            super(SecureAggregation, self).set_host_params_to_local(host_params['weights'], current_round=current_round)
        else:
            raise NotImplementedError
    
    def fit_on_local_data(self):
        if self._client_status is SAStatus.Init:
            # Generate local DH sk and pk
            ka_parameter = load_der_parameters(self._received_pp)
            self._client_dh_maks_sk = ka_parameter.generate_private_key()
            self._client_dh_mask_pk = self._client_dh_maks_sk.public_key()
            self._client_dh_aes_sk = ka_parameter.generate_private_key()
            self._client_dh_aes_pk = self._client_dh_aes_sk.public_key()
        elif self._client_status is SAStatus.DHKeyAgree:
            for i in range(len(self._peer_dh_pk)):
                self._peer_dh_pk[i]['derived_mask_key'] = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=None,
                ).derive(self._client_dh_maks_sk.exchange(load_der_public_key(self._peer_dh_pk[i]['dh_mask_pk'])))
                self._peer_dh_pk[i]['derived_aes_key'] = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=None,
                ).derive(self._client_dh_aes_sk.exchange(load_der_public_key(self._peer_dh_pk[i]['dh_aes_pk'])))
        elif self._client_status is SAStatus.ApplyMask:
            # fit on local data and return the train loss and number of training samples
            return super(SecureAggregation, self).fit_on_local_data()
        
    def retrieve_local_upload_info(self):
        if self._client_status is SAStatus.Init:
            # upload local pk
            return {
                'client_id': self.client_id,
                'dh_mask_pk': self._client_dh_mask_pk.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo),
                'dh_aes_pk': self._client_dh_aes_pk.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo),
                'num_training_samples': len(self.train_data['x'])
            }
        elif self._client_status is SAStatus.DHKeyAgree:
            # Encode local parameters
            # local_weights = self._retrieve_local_params()
            # Create shares
            local_mask_shares = self._sss.share(self._local_mask_seed)
            sa_mask_shares = self._sss.share(self._client_dh_maks_sk.private_numbers().x)
            encrypted_msg = []
            u = self.client_id
            for i in range(self._n - 1):
                v = self._peer_dh_pk[i]['client_id']
                encrypted_msg.append(aes_gcm_encrypt(
                    self._peer_dh_pk[i]['derived_aes_key'],
                    message=f'{u}||{v}||{sa_mask_shares[i]}||{local_mask_shares[i]}', user_add=f'{u}||{v}'
                ))
            self._peer_shares = [f'{u}||{u}||{sa_mask_shares[-1]}||{local_mask_shares[-1]}']
            return encrypted_msg
        elif self._client_status is SAStatus.ApplyMask:
            real_local_weights = self._retrieve_local_params()
            # scale the local parameters before the training
            real_local_weights = [
                e * len(self.train_data['x']) / self._total_training_samples for e in real_local_weights
            ]
            # encode
            self.logger.info('Start Encode')
            gfn_encode = np.vectorize(partial(GaloisFieldNumber.encode, p=self._p))
            local_weights = [gfn_encode(e.astype(np.float64)) for e in real_local_weights]
            self.logger.info('Start Apply Mask')
            for record in self._peer_dh_pk:
                cid = record['client_id']
                random.seed(record['derived_mask_key'])
                for i in range(len(local_weights)):
                    # self.logger.info('Start Generate Mask')
                    tmp_random_mask = np.vectorize(lambda _: random.randint(0, self._p))(local_weights[i])
                    if cid < self.client_id:
                        # self.logger.info(f'Start - Mask {tmp_random_mask.shape}')
                        local_weights[i] += (self._p - tmp_random_mask)
                    else:
                        # self.logger.info(f'Start + Mask {tmp_random_mask.shape}')
                        local_weights[i] += tmp_random_mask
                    del tmp_random_mask
            # self.logger.info('End Apply Mask')
            del real_local_weights
            return local_weights
