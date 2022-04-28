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
from ..secure_protocols import ShamirSecretSharing, GaloisFieldNumber, aes_gcm_decrypt, \
    aes_gcm_encrypt, GaloisFieldParams

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric.dh import DHPrivateNumbers
# serialization
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat, ParameterFormat
from cryptography.hazmat.primitives.serialization import load_der_parameters
from cryptography.hazmat.primitives.serialization import load_der_public_key


class SAStatus(enum.Enum):
    Init = 'Init'
    DHKeyAgree = 'KeyAgree'
    ApplyMask = 'ApplyMask'
    RemoveMask = 'RemoveMask'
    UpdateWeights = 'UpdateWeights'


class SecureAggregation(FedStrategy):

    def __init__(self, param_parser=ParamParser):
        super().__init__(param_parser)

        config_manager = ConfigurationManager()
        if config_manager.role == Role.Server:
            # default values of p, m, n
            self._p = 2 ** 2203 - 1  # 16th Mersenne prime
            self._n = config_manager.num_of_train_clients_contacted_per_round
            self._m = self._n
            self._server_status = SAStatus.Init
            self._host_params_type = HostParamsType.Personalized
            self._received_peer_dh_pk = None
            self._encrypted_shares = None
            self._aggregated_weights = None

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

        self._survive_clients = None
        self._dropout_clients = None
        self._sss: ShamirSecretSharing = None
        self._total_training_samples = None

    def host_select_train_clients(self, ready_clients):
        if self._server_status is SAStatus.Init:
            return super(SecureAggregation, self).host_select_train_clients(ready_clients)
        else:
            return self.train_selected_clients

    def retrieve_host_download_info(self):
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
        elif self._server_status is SAStatus.RemoveMask:
            return {
                cid: {
                    'status': self._server_status,
                    'received_clients': self._survive_clients, 'missed_clients': self._dropout_clients
                }
                for cid in self.train_selected_clients
            }
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
            self._survive_clients = [e['client_id'] for e in client_params]
            self._dropout_clients = [e for e in self.train_selected_clients if e not in self._survive_clients]
            self._aggregated_weights = []
            for i in range(len(client_params[0]['masked_weights'])):
                self._aggregated_weights.append(np.sum([
                    e['masked_weights'][i] for e in client_params if e['client_id'] in self._survive_clients
                ], axis=0))
            self._server_status = SAStatus.RemoveMask
        elif self._server_status is SAStatus.RemoveMask:
            # Remove local mask of survival clients
            sss = ShamirSecretSharing(m=self._m, n=self._n, p=self._p)
            gfp = GaloisFieldParams(self._p)
            for cid in self._survive_clients:
                current_client_shares = [e['local_mask_shares'][cid] for e in client_params]
                current_client_seed = sss.recon(current_client_shares)
                random.seed(current_client_seed)
                for i in range(len(self._aggregated_weights)):
                    tmp_random_mask = np.vectorize(
                        lambda _: GaloisFieldNumber(
                            encoding=self._p - random.randint(0, self._p),
                            exponent=GaloisFieldNumber.FULL_PRECISION,
                            gfp=gfp
                        )
                    )(self._aggregated_weights[i])
                    self._aggregated_weights[i] += tmp_random_mask
                    del tmp_random_mask
            # Remove the mask of dropout clients
            if len(self._dropout_clients) > 0:
                dh_mask_pk = {e['client_id']: load_der_public_key(e['dh_mask_pk']) for e in self._received_peer_dh_pk}
                for cid in self._dropout_clients:
                    current_client_shares = [e['global_mask_shares'][cid] for e in client_params]
                    current_client_sk_x = sss.recon(current_client_shares)
                    current_client_sk = DHPrivateNumbers(
                        current_client_sk_x, dh_mask_pk[cid].public_numbers()).private_key()
                    for other_cid in dh_mask_pk:
                        if other_cid == cid:
                            continue
                        shared_mask_seed = HKDF(
                            algorithm=hashes.SHA256(), length=32, salt=None, info=None,
                        ).derive(current_client_sk.exchange(dh_mask_pk[other_cid]))
                        # Remove peer mask
                        random.seed(shared_mask_seed)
                        for i in range(len(self._aggregated_weights)):
                            if cid < other_cid:
                                tmp_random_mask = np.vectorize(
                                    lambda _: GaloisFieldNumber(
                                        encoding=self._p - random.randint(0, self._p),
                                        exponent=GaloisFieldNumber.FULL_PRECISION,
                                        gfp=gfp
                                    )
                                )(self._aggregated_weights[i])
                                self._aggregated_weights[i] += tmp_random_mask
                            else:
                                tmp_random_mask = np.vectorize(
                                    lambda _: GaloisFieldNumber(
                                        encoding=random.randint(0, self._p),
                                        exponent=GaloisFieldNumber.FULL_PRECISION,
                                        gfp=gfp
                                    )
                                )(self._aggregated_weights[i])
                                self._aggregated_weights[i] += tmp_random_mask
                            del tmp_random_mask
            decode = np.vectorize(lambda x: x.decode())
            self.ml_model.set_weights([decode(e) for e in self._aggregated_weights])
            del self._aggregated_weights
            self._aggregated_weights = None
            self._server_status = SAStatus.UpdateWeights
        elif self._server_status is SAStatus.UpdateWeights:
            self._server_status = SAStatus.Init
        else:
            raise NotImplementedError

    def set_host_params_to_local(self, host_params, current_round: int):
        self._client_status = host_params['status']
        self.logger.info(f'Client {self.client_id} report status {self._client_status}')
        if self._client_status is SAStatus.Init:
            self._m = host_params['m']
            self._n = host_params['n']
            self._p = host_params['p']
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
        elif self._client_status is SAStatus.RemoveMask:
            self._survive_clients = host_params['received_clients']
            self._dropout_clients = host_params['missed_clients']
            assert len([e for e in self._survive_clients if e in self._dropout_clients]) == 0, \
                'The server can only require one share for each client'
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
            gfp = GaloisFieldParams(p=self._p)
            gfn_encode = np.vectorize(partial(GaloisFieldNumber.encode, gfp=gfp))
            local_weights = [gfn_encode(e.astype(np.float64)) for e in real_local_weights]
            masked_local_weights = []
            self.logger.info('Start Apply Mask')
            for record in self._peer_dh_pk:
                cid = record['client_id']
                # Add peer mask
                random.seed(record['derived_mask_key'])
                for i in range(len(local_weights)):
                    if self.client_id < cid:
                        tmp_random_mask = np.vectorize(
                            lambda _: GaloisFieldNumber(
                                encoding=self._p - random.randint(0, self._p),
                                exponent=GaloisFieldNumber.FULL_PRECISION,
                                gfp=gfp
                            )
                        )(real_local_weights[i])
                    else:
                        tmp_random_mask = np.vectorize(
                            lambda _: GaloisFieldNumber(
                                encoding=random.randint(0, self._p),
                                exponent=GaloisFieldNumber.FULL_PRECISION,
                                gfp=gfp
                            )
                        )(real_local_weights[i])
                    masked_local_weights[i] = local_weights[i] + tmp_random_mask
                    del tmp_random_mask
            # Add individual mask
            masked_local_weights_final = []
            random.seed(self._local_mask_seed)
            for i in range(len(masked_local_weights)):
                tmp_random_mask = np.vectorize(
                    lambda _: GaloisFieldNumber(
                        encoding=random.randint(0, self._p),
                        exponent=GaloisFieldNumber.FULL_PRECISION,
                        gfp=gfp
                    )
                )(real_local_weights[i])
                masked_local_weights_final[i] = masked_local_weights[i] + tmp_random_mask
                del tmp_random_mask
            del local_weights
            del masked_local_weights
            del real_local_weights
            return {'client_id': self.client_id, 'masked_weights': masked_local_weights_final}
        elif self._client_status is SAStatus.RemoveMask:
            global_mask_shares = {}
            local_mask_shares = {}
            for share in self._peer_shares:
                src_cid, self_cid, gms, lms = share.split('||')
                if eval(src_cid) in self._survive_clients:
                    local_mask_shares[eval(src_cid)] = eval(lms)
                if eval(src_cid) in self._dropout_clients:
                    global_mask_shares[eval(src_cid)] = eval(gms)
            return {
                'client_id': self.client_id,
                'global_mask_shares': global_mask_shares,
                'local_mask_shares': local_mask_shares
            }
        