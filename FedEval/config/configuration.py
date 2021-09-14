import json
import os
from abc import abstractmethod, abstractproperty, ABC
from copy import deepcopy
from enum import Enum
from threading import Lock
from typing import List, Mapping, Optional, Sequence, TextIO, Tuple, Union

import yaml

from .filename_checker import check_filename
from .singleton import Singleton
from .role import Role

RawConfigurationDict = Mapping[str, Optional[Union[str, int]]]

DEFAULT_D_CFG_FILENAME = '1_data_config.yml'
DEFAULT_MDL_CFG_FILENAME = '2_model_config.yml'
DEFAULT_RT_CFG_FILENAME = '3_runtime_config.yml'

# default data configurations
_D_DIR_KEY = 'data_dir'
_D_NAME_KEY = 'dataset'
_D_NI_ENABLE_KEY = 'non-iid'
_D_NI_CLASS_KEY = 'non-iid-class'
_D_NI_STRATEGY_KEY = 'non-iid-strategy'
_D_NORMALIZE_KEY = 'normalize'
_D_SAMPLE_SIZE_KEY = 'sample_size'
_D_PARTITION_KEY = 'train_val_test'
_DEFAULT_D_CFG: RawConfigurationDict = {
    _D_DIR_KEY: 'data',
    _D_NAME_KEY: 'mnist',
    _D_NI_ENABLE_KEY: False,
    _D_NI_CLASS_KEY: 1,
    _D_NI_STRATEGY_KEY: 'average',
    _D_NORMALIZE_KEY: True,
    _D_SAMPLE_SIZE_KEY: 300,
    _D_PARTITION_KEY: [0.8, 0.1, 0.1]
}

# default model configurations
_STRATEGY_KEY = 'FedModel'
_STRATEGY_NAME_KEY = 'name'
_STRATEGY_ETA_KEY = 'eta'
_STRATEGY_B_KEY = 'B'
_STRATEGY_C_KEY = 'C'
_STRATEGY_E_KEY = 'E'
_STRATEGY_MAX_ROUND_NUM_KEY = 'max_rounds'
_STRATEGY_TOLERANCE_NUM_KEY = 'num_tolerance'
_STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY = 'rounds_between_val'
_STRATEGY_FEDSTC_SPARSITY_KEY = 'sparsity'
_STRATEGY_FEDPROX_MU_KEY = 'mu'
_STRATEGY_FEDOPT_TAU_KEY = 'tau'
_STRATEGY_FEDOPT_BETA1_KEY = 'beta1'
_STRATEGY_FEDOPT_BETA2_KEY = 'beta2'
_STRATEGY_FEDOPT_NAME_KEY = 'opt_name'
_STRATEGY_FETCHSGD_COL_NUM_KEY = 'num_col'
_STRATEGY_FETCHSGD_ROW_NUM_KEY = 'num_row'
_STRATEGY_FETCHSGD_BLOCK_NUM_KEY = 'num_block'
_STRATEGY_FETCHSGD_TOP_K_KEY = 'top_k'

_ML_KEY = 'MLModel'
_ML_NAME_KEY = 'name'
_ML_ACTIVATION_KEY = 'activation'
_ML_DROPOUT_RATIO_KEY = 'dropout'
_ML_UNITS_SIZE_KEY = 'units'
_ML_OPTIMIZER_KEY = 'optimizer'
_ML_OPTIMIZER_NAME_KEY = 'name'
_ML_OPTIMIZER_LEARNING_RATE_KEY = 'lr'
_ML_OPTIMIZER_MOMENTUM_KEY = 'momentum'
_ML_LOSS_CALC_METHODS_KEY = 'loss'
_ML_METRICS_KEY = 'metrics'
_ML_DEFAULT_METRICS = ['accuracy']

_DEFAULT_MDL_CFG: RawConfigurationDict = {
    _STRATEGY_KEY: {
        _STRATEGY_NAME_KEY: 'FedAvg',
        # shared params
        _STRATEGY_B_KEY: 32,
        _STRATEGY_C_KEY: 0.1,
        _STRATEGY_E_KEY: 1,
        _STRATEGY_MAX_ROUND_NUM_KEY: 3000,
        _STRATEGY_TOLERANCE_NUM_KEY: 100,
        _STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY: 1,
        # FedSTC
        _STRATEGY_FEDSTC_SPARSITY_KEY: 0.01,
        # FedProx
        _STRATEGY_FEDPROX_MU_KEY: 0.01,
        # FedOpt
        _STRATEGY_FEDOPT_TAU_KEY: 1e-4,
        _STRATEGY_FEDOPT_BETA1_KEY: 0.9,
        _STRATEGY_FEDOPT_BETA2_KEY: 0.99,
        _STRATEGY_FEDOPT_NAME_KEY: 'fedyogi',
        # server-side learning rate, used by FedSCA and FedOpt
        _STRATEGY_ETA_KEY: 1.0,
        # FetchSGD
        _STRATEGY_FETCHSGD_COL_NUM_KEY: 5,
        _STRATEGY_FETCHSGD_ROW_NUM_KEY: 1e4,
        _STRATEGY_FETCHSGD_BLOCK_NUM_KEY: 10,
        _STRATEGY_FETCHSGD_TOP_K_KEY: 0.1,
    },
    _ML_KEY: {
        _ML_NAME_KEY: 'MLP',
        _ML_ACTIVATION_KEY: 'relu',
        _ML_DROPOUT_RATIO_KEY: 0.2,
        _ML_UNITS_SIZE_KEY: [512, 512],
        _ML_OPTIMIZER_KEY: {
            _ML_OPTIMIZER_NAME_KEY: 'sgd',
            _ML_OPTIMIZER_LEARNING_RATE_KEY: 0.1,
            _ML_OPTIMIZER_MOMENTUM_KEY: 0,
            # _ML_OPTIMIZER_MOMENTUM_KEY: 0.9,    # FetchSGD
        },
        _ML_LOSS_CALC_METHODS_KEY: 'categorical_crossentropy',
        _ML_METRICS_KEY: _ML_DEFAULT_METRICS,
    },
}

# default runtime configurations
_RT_CLIENTS_KEY = 'clients'
_RT_C_BANDWIDTH_KEY = 'bandwidth'

_RT_SERVER_KEY = 'server'
_RT_S_HOST_KEY = 'host'
_RT_S_LISTEN_KEY = 'listen'
_RT_S_PORT_KEY = 'port'
_RT_S_CLIENTS_NUM_KEY = 'num_clients'
_RT_S_SECRET_KEY = 'secret'

_RT_DOCKER_KEY = 'docker'
_RT_D_IMAGE_LABEL_KEY = 'image'
_RT_D_CONTAINER_NUM_KEY = 'num_containers'
_RT_D_GPU_ENABLE_KEY = 'enable_gpu'
_RT_D_GPU_NUM_KEY = 'num_gpu'

_RT_MACHINES_KEY = 'machines'
_RT_M_ADDRESS_KEY = 'host'
_RT_M_PORT_KEY = 'port'
_RT_M_USERNAME_KEY = 'username'
_RT_M_WORK_DIR_KEY = 'dir'
_RT_M_SK_FILENAME_KEY = 'key'
_RT_M_CAPACITY_KEY = 'capacity'
_RT_M_SERVER_NAME = 'server'

_RT_LOG_DIR_PATH_KEY = 'log_dir'
_DEFAULT_RT_CFG: RawConfigurationDict = {
    _RT_LOG_DIR_PATH_KEY: 'log/quickstart',
    _RT_DOCKER_KEY: {
        _RT_D_IMAGE_LABEL_KEY: 'fedeval:gpu',
        _RT_D_CONTAINER_NUM_KEY: 10,
        _RT_D_GPU_ENABLE_KEY: False,
        _RT_D_GPU_NUM_KEY: 0,
    },
    _RT_CLIENTS_KEY: {
        _RT_C_BANDWIDTH_KEY: '100Mbit',
    },
    _RT_SERVER_KEY: {
        _RT_S_HOST_KEY: 'server',
        _RT_S_LISTEN_KEY: 'server',
        _RT_S_CLIENTS_NUM_KEY: 10,
        _RT_S_PORT_KEY: 8000,
        _RT_S_SECRET_KEY: 'secret!',
    },
    _RT_MACHINES_KEY: {
        _RT_M_SERVER_NAME: {
            _RT_M_ADDRESS_KEY: '10.173.1.22',
            _RT_M_PORT_KEY: 22,
            _RT_M_USERNAME_KEY: 'ubuntu',
            _RT_M_WORK_DIR_KEY: '/ldisk/chaidi/FedEval',
            _RT_M_SK_FILENAME_KEY: 'id_rsa',
        },
        'm1': {
            _RT_M_ADDRESS_KEY: '10.173.1.22',
            _RT_M_PORT_KEY: 22,
            _RT_M_USERNAME_KEY: 'ubuntu',
            _RT_M_WORK_DIR_KEY: '/ldisk/chaidi/FedEval',
            _RT_M_SK_FILENAME_KEY: 'id_rsa',

            _RT_M_CAPACITY_KEY: 100,
        },
    },
}

# --- Configuration Entities ---
class _Configuraiton(object):
    def __init__(self, config: RawConfigurationDict) -> None:
        self._inner: RawConfigurationDict = config

    @property
    def inner(self) -> RawConfigurationDict:
        """return a deep copy of its inner configuraiton data, presented as a dict.
        Noticed that modifications on the returned object will NOT affect the original
        configuration.

        Returns:
            RawConfigurationDict: a deep copy of the inner data representaiton
            of this config object.
        """
        return deepcopy(self._inner)


class _DataConfig(_Configuraiton):
    _IID_EXCEPTiON_CONTENT = 'The dataset is configured as iid.'

    def __init__(self, data_config: RawConfigurationDict = _DEFAULT_D_CFG) -> None:
        super().__init__(data_config)

        # non-iid
        self._non_iid: bool = self._inner.get(_D_NI_ENABLE_KEY, False)
        if self._non_iid:
            self._non_iid_class_num: int = int(
                self._inner.get(_D_NI_CLASS_KEY, 1))
            self._non_iid_strategy_name: str = self._inner.get(
                _D_NI_STRATEGY_KEY, 'average')

        # partition
        partition = self._inner[_D_PARTITION_KEY].copy()
        if len(partition) != 3:
            raise ValueError(
                f'there should be 3 values in {_D_PARTITION_KEY}.')
        for i in partition:
            if i < 0:
                raise ValueError(
                    f'values in {_D_PARTITION_KEY} should not be negetive.')
        summation = sum(partition)
        if summation <= 1e-6:
            raise ValueError(f'values in {_D_PARTITION_KEY} are too small.')
        partition = [i / summation for i in partition]
        self._partition = partition

    @property
    def dir_name(self) -> str:
        """The output directory of the clients' data.

        Returns:
            str: the name of the data directory.
        """
        return self._inner[_D_DIR_KEY]

    @property
    def dataset_name(self) -> str:
        """the name of the dataset, chosen from mnist, cifar10, cifar100, femnist, and mnist.

        Returns:
            str: the name of chosen dataset.
        """
        return self._inner[_D_NAME_KEY]

    @property
    def iid(self) -> bool:
        """if the dataset would be used in an i.i.d. manner.

        Returns:
            bool: True if the dataset is sampled in an i.i.d. manner; otherwise, False.
        """
        return not self._non_iid

    @property
    def non_iid_class_num(self) -> int:
        """return the number of classes hold by each client.
        Only avaliable when the dataset is sampled in a non-i.i.d. form.

        Raises:
            AttributeError: raised when called without non-i.i.d. setting.

        Returns:
            int: the number of classes hold by each client.
        """
        if self._non_iid:
            return self._non_iid_class_num
        else:
            raise AttributeError(_DataConfig._IID_EXCEPTiON_CONTENT)

    @property
    def non_iid_strategy_name(self) -> str:
        """return the name of non-i.i.d. data partition strategy.
        Two choices are given:
        1. "natural" strategy for femnist and celebA dataset
        2. "average" for mnist, cifar10 and cifar100

        Raises:
            AttributeError: raised when called without non-i.i.d. setting.

        Returns:
            str: the name of non-i.i.d. data partition strategy.
        """
        if self._non_iid:
            if self._non_iid_strategy_name_check():
                raise AttributeError(
                    f'unregistered non-iid data partition srategy name: {self._non_iid_strategy_name}')
            return self._non_iid_strategy_name
        else:
            raise AttributeError(_DataConfig._IID_EXCEPTiON_CONTENT)

    def _non_iid_strategy_name_check(self) -> bool:
        """check if the non-i.i.d. data partition strategy is known.

        Returns:
            bool: True if the data partition strategy name is registered as followed; otherwise, False.
        """
        return self._non_iid_strategy_name in ['natural', 'average']

    @property
    def normalized(self) -> bool:
        """whether the image pixel data point will be normalized to [0, 1].

        Returns:
            bool: True if data points would be normalized; otherwise, False.
        """
        return self._inner[_D_NORMALIZE_KEY]

    @property
    def sample_size(self) -> int:
        """return the number of samples owned by each client."""
        return self._inner[_D_SAMPLE_SIZE_KEY]

    @property
    def data_partition(self) -> Sequence[float]:
        """get the data partition proportion, ordered as
        [train data ratio, test data ration, validation data ration].
        
        Constraints met by the return value:
            1. all the ratios in the returned list sum up to 1.
            2. all the ratios in the returned list are non-negative.

        Returns:
            Sequence[float]: [train data ratio, test data ration, validation data ration]
        """
        return self._partition


class _ModelConfig(_Configuraiton):
    def __init__(self, model_config: RawConfigurationDict = _DEFAULT_MDL_CFG) -> None:
        _ModelConfig.__check_raw_config(model_config)
        super().__init__(model_config)
        self._strategy_cfg = model_config[_STRATEGY_KEY]
        self._ml_cfg = model_config[_ML_KEY]

        self._unit_size: List[int] = [
            int(i) for i in self._ml_cfg[_ML_UNITS_SIZE_KEY]]

    @staticmethod
    def __check_raw_config(config: RawConfigurationDict) -> None:
        _ModelConfig.__check_runtime_config_shallow_structure(config)
        _ModelConfig.__check_ML_model_params(config[_ML_KEY])

    @staticmethod
    def __check_runtime_config_shallow_structure(config: RawConfigurationDict) -> None:
        assert config.get(
            _ML_KEY) != None, f'model_config should have `{_ML_KEY}`'
        assert config.get(
            _STRATEGY_KEY) != None, f'model_config should have `{_STRATEGY_KEY}`'

    @staticmethod
    def __check_ML_model_params(ml_config: RawConfigurationDict) -> None:
        dropout_ratio = ml_config.get(_ML_DROPOUT_RATIO_KEY)
        if dropout_ratio:
            assert dropout_ratio >= 0 and dropout_ratio <= 1, 'dropout ration out of range.'

    @property
    def strategy_config(self) -> RawConfigurationDict:
        """a variant of inner method: return a copy of inner strategy raw dict.

        Returns:
            RawConfigurationDict: a deep copy of the strategy-related configuration dict.
        """
        return deepcopy(self._strategy_cfg)

    @property
    def ml_config(self) -> RawConfigurationDict:
        """a variant of inner method: return a copy of inner machine learning raw dict.

        Returns:
            RawConfigurationDict: a deep copy of the ML model-related configuration dict.
        """
        return deepcopy(self._ml_cfg)

    @property
    def strategy_name(self) -> str:
        """get the class name of the federated strategy (i.e., the main controller of federated
        process). Notice that the strategy class with this name (case sensitive and whole word
        matching) should have been implemented in this library (specifically, in strategy module),
        otherwise a TypeNotFound exception would be raised in the following steps.

        Returns:
            str: the classname/typename of the federated strategy.
        """
        return self._strategy_cfg[_STRATEGY_NAME_KEY]

    @property
    def ml_method_name(self) -> str:
        """get the class name of the machine learning model (i.e., the kernel of the whole
        calculation process). Notice that the strategy class with this name (case sensitive
        and whole word matching) should have been implemented in this library (specifically,
        in model module), otherwise a TypeNotFound exception would be raised in the
        following steps.

        Returns:
            str: the classname/typename of the inner machine learning model.
        """
        return self._ml_cfg[_ML_NAME_KEY]

    @property
    def server_learning_rate(self) -> float:
        """get the learning rate on the server side.
        Only available in FedOpt and FedSCA.

        Raises:
            AttributeError: called in a in proper federated strategy.

        Returns:
            float: the learning rate on the server side.
        """
        if self.strategy_name != 'FedOpt' or self.strategy_name != 'FedSCA':
            raise AttributeError
        return float(self._strategy_cfg[_STRATEGY_ETA_KEY])

    @property
    def B(self) -> int:
        """the local minibatch size used for the updates on the client side."""
        return int(self._strategy_cfg[_STRATEGY_B_KEY])

    @property
    def C(self) -> float:
        """the fraction of clients that perform computation in each round.

        Examples:
            if there are 100 available clients in a test network with a C of 0.2,
            then there should be (100*0.2=)20 clients in each round of iterations.
        """ 
        return float(self._strategy_cfg[_STRATEGY_C_KEY])

    @property
    def E(self) -> int:
        """the number of training passes that each client makes over its local dataset
        in each round.
        """
        return int(self._strategy_cfg[_STRATEGY_E_KEY])

    @property
    def max_round_num(self) -> int:
        """the total/maximum number of the iteration rounds."""
        return int(self._strategy_cfg[_STRATEGY_MAX_ROUND_NUM_KEY])

    @property
    def tolerance_num(self) -> int:
        """the patience for early stopping"""
        return int(self._strategy_cfg[_STRATEGY_TOLERANCE_NUM_KEY])

    @property
    def num_of_rounds_between_val(self) -> int:
        """the number of rounds between test or validation"""
        return int(self._strategy_cfg[_STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY])

    @property
    def stc_sparsity(self) -> float:
        """TODO(fgh): the origin of FedSTC"""
        return float(self._strategy_cfg[_STRATEGY_FEDSTC_SPARSITY_KEY])

    @property
    def prox_mu(self) -> float:
        """the /mu parameter in FedProx, a scaler that measures the approximation
        between the local model and the global model.
        More info available in Federated Optimization in Heterogeneous Networks(arXiv:1812.06127).
        """
        return float(self._strategy_cfg[_STRATEGY_FEDPROX_MU_KEY])

    @property
    def opt_tau(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_TAU_KEY])

    @property
    def opt_beta_1(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_BETA1_KEY])

    @property
    def opt_beta_2(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_BETA2_KEY])

    @property
    def activation(self) -> str:
        """the name of activation mechanism in tensorflow layers.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/activations.
        """
        return self._ml_cfg[_ML_ACTIVATION_KEY]

    @property
    def dropout(self) -> float:
        """the dropout fraction of Dropout layer in the DL model."""
        return float(self._ml_cfg[_ML_DROPOUT_RATIO_KEY])

    @property
    def unit_size(self) -> Sequence[int]:
        """the size of sequential neural network components.

        Returns:
            Sequence[int]: the size of network components
            (ordered the same with data flow direction)
        """
        return self._unit_size.copy()

    @property
    def optimizer_name(self) -> str:
        """the name of the optimizer in tensorflow network.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers.
        """
        return self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_NAME_KEY]

    @property
    def learning_rate(self) -> float:
        """the learning rate of model training in tensorlflow."""
        return float(self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_LEARNING_RATE_KEY])

    @property
    def momentum(self) -> float:
        """the momentum of the optimizer."""
        return float(self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_MOMENTUM_KEY])

    @property
    def loss_calc_method(self) -> str:
        """the identifier of a loss function in tensorflow.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/losses.

        Returns:
            str: the string name of the loss function during model training.
        """
        return self._ml_cfg[_ML_LOSS_CALC_METHODS_KEY]

    @property
    def metrics(self) -> Sequence[str]:
        """names of the metrics used in model training and validation in tensorflow.
        More info in https://tensorflow.google.cn/api_docs/python/tf/keras/metrics.

        Returns:
            Sequence[str]: a copy of metric names.
        """
        return self._ml_cfg[_ML_METRICS_KEY].copy()

    @property
    def col_num(self) -> int:
        """the number of columns in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_COL_NUM_KEY])

    @property
    def row_num(self) -> int:
        """the number of rows in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_ROW_NUM_KEY])

    @property
    def block_num(self) -> int:
        """the number of blocks in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_BLOCK_NUM_KEY])

    @property
    def top_k(self) -> int:
        """the number of top items in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_TOP_K_KEY])


class _RT_Machine(_Configuraiton):
    __ITEM_CHECK_VALUE_ERROR_PATTERN = 'machine configuraitons should have {}.'

    def __init__(self, machine_config: RawConfigurationDict, is_server: bool = False) -> None:
        _RT_Machine.__check_items(machine_config, is_server)
        super().__init__(machine_config)
        self._is_server = is_server

    @staticmethod
    def __check_items(config: RawConfigurationDict, is_server: bool = False) -> None:
        required_keys = [_RT_M_ADDRESS_KEY, _RT_M_WORK_DIR_KEY,
                         _RT_M_PORT_KEY, _RT_M_USERNAME_KEY, _RT_M_SK_FILENAME_KEY]
        for k in required_keys:
            assert k in config, ValueError(
                _RT_Machine.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(k))
        if not is_server:
            assert _RT_M_CAPACITY_KEY in config, ValueError(
                _RT_Machine.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(_RT_M_CAPACITY_KEY))

    @property
    def is_server(self) -> bool:
        """if the machine is a central server."""
        return self._is_server

    @property
    def addr(self) -> str:
        """the IP address of this machine or the name of this container in docker."""
        return self._inner[_RT_M_ADDRESS_KEY]

    @property
    def port(self) -> int:
        """the port of this virtual machine on the physical machine."""
        return int(self._inner[_RT_M_PORT_KEY])

    @property
    def username(self) -> str:
        """the username of this machine."""
        return self._inner[_RT_M_USERNAME_KEY]

    @property
    def work_dir_path(self) -> str:
        """the path of this machine's working diretory."""
        return self._inner[_RT_M_WORK_DIR_KEY]

    @property
    def key_filename(self) -> str:
        """the name of ssh connection secret key file."""
        return self._inner[_RT_M_SK_FILENAME_KEY]

    @property
    def capacity(self) -> int:
        """the number of container that this machine can handle.
        Only available on the client side.

        Raises:
            AttributeError: called from the server side.
        """
        if self._is_server:
            raise AttributeError(
                'capacity is inaccessible for the server side.')
        return int(self._inner[_RT_M_CAPACITY_KEY])


class _RuntimeConfig(_Configuraiton):
    __ITEM_CHECK_VALUE_ERROR_PATTERN = 'runtime configurations should have {}.'

    def __init__(self, runtime_config: RawConfigurationDict = _DEFAULT_RT_CFG) -> None:
        _RuntimeConfig.__check_items(runtime_config)
        super().__init__(runtime_config)
        self.__init_machines()

    @staticmethod
    def __check_items(config: RawConfigurationDict) -> None:
        required_keys = [_RT_CLIENTS_KEY, _RT_DOCKER_KEY,
                         _RT_SERVER_KEY, _RT_LOG_DIR_PATH_KEY]
        for k in required_keys:
            assert k in config, ValueError(
                _RuntimeConfig.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(k))

    def _has_machines(self) -> bool:
        return _RT_MACHINES_KEY in self._inner

    def __init_machines(self) -> bool:
        if not self._has_machines():
            return False
        self._machines: dict[_RT_Machine] = dict()
        for name in self._inner[_RT_MACHINES_KEY]:
            self._machines[name] = _RT_Machine(
                self._inner[_RT_MACHINES_KEY][name], name == _RT_M_SERVER_NAME)
        return True

    @property
    def machines(self) -> Optional[Mapping[str, _RT_Machine]]:
        """return a deep copy of all the machines in the configuration.

        Returns:
            Optional[Mapping[str, _RT_Machine]]: None if there is no machine setting.
        """
        if not self._has_machines():
            return None
        return deepcopy(self._machines)

    @property
    def client_machines(self) -> Optional[Mapping[str, _RT_Machine]]:
        """return a deep copy of all the client machines in the configuration.

        Returns:
            Optional[Mapping[str, _RT_Machine]]: None if there is no client machine setting.
        """
        if not self._has_machines():
            return None
        return deepcopy({name: v for name, v in self._machines if not v.is_server})

    @property
    def client_bandwidth(self) -> str:
        """the bandwidth of each client."""
        return self._inner[_RT_CLIENTS_KEY][_RT_C_BANDWIDTH_KEY]

    @property
    def image_label(self) -> str:
        """the label of the docker image used in this experiment."""
        return self._inner[_RT_DOCKER_KEY][_RT_D_IMAGE_LABEL_KEY]

    @property
    def container_num(self) -> int:
        """the number of total docker containers in this experiment."""
        return self._inner[_RT_DOCKER_KEY][_RT_D_CONTAINER_NUM_KEY]

    @property
    def central_server_addr(self) -> str:
        """the IP address of the central server."""
        return self._inner[_RT_SERVER_KEY][_RT_S_HOST_KEY]

    @property
    def central_server_listen_at(self) -> str:
        """the listening IP address of the flask services on the cetral server side."""
        return self._inner[_RT_SERVER_KEY][_RT_S_LISTEN_KEY]

    @property
    def central_server_port(self) -> int:
        """the port that the central server occupies."""
        return int(self._inner[_RT_SERVER_KEY][_RT_S_PORT_KEY])

    @property
    def client_num(self) -> int:
        """the total number of the clients."""
        return int(self._inner[_RT_SERVER_KEY][_RT_S_CLIENTS_NUM_KEY])

    @property
    def log_dir_path(self) -> str:
        """the path of the base of log directory."""
        return self._inner[_RT_LOG_DIR_PATH_KEY]

    @property
    def secret_key(self) -> str:
        """the secret key of the flask service on the central server side.

        Returns:
            str: the secret key as a string.
        """
        return self._inner[_RT_SERVER_KEY][_RT_S_SECRET_KEY]

    @property
    def gpu_enabled(self) -> bool:
        """whether the GPU is enabled in this experiment."""
        return bool(self._inner[_RT_DOCKER_KEY][_RT_D_GPU_ENABLE_KEY])

    @property
    def gpu_num(self) -> int:
        """the number of GPUs.

        Raises:
            AttributeError: called without GPUs enabled.
        """
        if not self.gpu_enabled:
            raise AttributeError('GPU is not enabled.')
        return int(self._inner[_RT_DOCKER_KEY][_RT_D_GPU_NUM_KEY])


# --- Configuration Manager Interfaces ---
class ConfigurationManagerInterface(ABC):
    @abstractproperty
    def data_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def model_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def runtime_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def data_config(self) -> RawConfigurationDict:
        raise NotImplementedError

    @abstractproperty
    def model_config(self) -> _ModelConfig:
        raise NotImplementedError

    @abstractproperty
    def runtime_config(self) -> RawConfigurationDict:
        raise NotImplementedError


class ClientConfigurationManagerInterface(ABC):
    """an interface of ConfigurationManager from the client side,
    regulating the essential functions as clients.

    Raises:
        NotImplementedError: called without implementation.
    """
    pass

class ServerConfigurationManagerInterface(ABC):
    """an interface of ConfigurationManager from the central server side,
    regulating the essential functions as clients.

    Raises:
        NotImplementedError: called without implementation.
    """
    @abstractproperty
    def num_of_clients_contacted_per_round(self) -> int:
        raise NotImplementedError


# --- Configuration Serilizer Interfaces ---
_DEFAULT_ENCODING = 'utf-8'
_Stream = Union[str, bytes, TextIO]

class _CfgYamlInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons in YAML.
    """
    @staticmethod
    @abstractmethod
    def from_yamls(data_cfg_stream: _Stream,
                   model_cfg_stream: _Stream,
                   runtime_cfg_stream: _Stream) -> ConfigurationManagerInterface:
        raise NotImplementedError

    @abstractmethod
    def to_yamls(self) -> Tuple[str, str, str]:
        """convert self into YAML strings.

        Returns:
            Tuple[str, str, str]: [data_config_string,
                                   model_config_string,
                                   runtime_config_string]
        """
        raise NotImplementedError

    @staticmethod
    def load_yaml_configs(src_path,
                           data_cofig_filename: str = DEFAULT_D_CFG_FILENAME,
                           model_config_filename: str = DEFAULT_MDL_CFG_FILENAME,
                           runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME,
                           encoding=_DEFAULT_ENCODING) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        _d_cfg_path = os.path.join(src_path, data_cofig_filename)
        _mdl_cfg_path = os.path.join(src_path, model_config_filename)
        _rt_cfg_path = os.path.join(src_path, runtime_config_filename)
        return _CfgYamlInterface.load_yaml_configs_from_files(
            _d_cfg_path, _mdl_cfg_path, _rt_cfg_path, encoding=encoding)

    @staticmethod
    def load_yaml_configs_from_files(data_cfg_path: str,
                                      model_cfg_path: str,
                                      runtime_cfg_path: str,
                                      encoding=_DEFAULT_ENCODING) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        with open(data_cfg_path, 'r', encoding=encoding) as f:
            d_cfg = yaml.load(f)
        with open(model_cfg_path, 'r', encoding=encoding) as f:
            mdl_cfg = yaml.load(f)
        with open(runtime_cfg_path, 'r', encoding=encoding) as f:
            rt_cfg = yaml.load(f)
        return d_cfg, mdl_cfg, rt_cfg

    @staticmethod
    def save_yaml_configs_to_files(data_cfg: RawConfigurationDict,
                           model_cfg: RawConfigurationDict,
                           runtime_cfg: RawConfigurationDict,
                           dst_path,
                           data_cofig_filename: str = DEFAULT_D_CFG_FILENAME,
                           model_config_filename: str = DEFAULT_MDL_CFG_FILENAME,
                           runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME,
                           encoding=_DEFAULT_ENCODING) -> None:
        os.makedirs(dst_path, exist_ok=True)
        _d_cfg_path = os.path.join(dst_path, data_cofig_filename)
        _mdl_cfg_path = os.path.join(dst_path, model_config_filename)
        _rt_cfg_path = os.path.join(dst_path, runtime_config_filename)
        with open(_d_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(data_cfg, f)
        with open(_mdl_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(model_cfg, f)
        with open(_rt_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(runtime_cfg, f)


class _CfgJsonInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons in JSON.
    """
    @staticmethod
    @abstractmethod
    def from_jsons(data_cfg_stream: _Stream,
                   model_cfg_stream: _Stream,
                   runtime_cfg_stream: _Stream) -> ConfigurationManagerInterface:
        raise NotImplementedError

    @abstractmethod
    def to_jsons(self) -> Tuple[str, str, str]:
        """convert self into JSON strings.

        Returns:
            Tuple[str, str, str]: [data_config_string,
                                   model_config_string,
                                   runtime_config_string]
        """
        raise NotImplementedError

    @staticmethod
    def load_json_configs_from_files(data_cfg_path: str,
                                      model_cfg_path: str,
                                      runtime_cfg_path: str,
                                      encoding=_DEFAULT_ENCODING) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        with open(data_cfg_path, 'r', encoding=encoding) as f:
            d_cfg = json.load(f)
        with open(model_cfg_path, 'r', encoding=encoding) as f:
            mdl_cfg = json.load(f)
        with open(runtime_cfg_path, 'r', encoding=encoding) as f:
            rt_cfg = json.load(f)
        return d_cfg, mdl_cfg, rt_cfg

    @staticmethod
    def save_json_configs_to_files(data_cfg: RawConfigurationDict,
                                    model_cfg: RawConfigurationDict,
                                    runtime_cfg: RawConfigurationDict,
                                    dst_path,
                                    data_cofig_filename: str = DEFAULT_D_CFG_FILENAME,
                                    model_config_filename: str = DEFAULT_MDL_CFG_FILENAME,
                                    runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME,
                                    encoding=_DEFAULT_ENCODING) -> None:
        os.makedirs(dst_path, exist_ok=True)
        _d_cfg_path = os.path.join(dst_path, data_cofig_filename)
        _mdl_cfg_path = os.path.join(dst_path, model_config_filename)
        _rt_cfg_path = os.path.join(dst_path, runtime_config_filename)
        with open(_d_cfg_path, 'w', encoding=encoding) as f:
            json.dump(data_cfg, f)
        with open(_mdl_cfg_path, 'w', encoding=encoding) as f:
            json.dump(model_cfg, f)
        with open(_rt_cfg_path, 'w', encoding=encoding) as f:
            json.dump(runtime_cfg, f)


class _CfgSerializer(Enum):
    """types of serializer for configurations."""
    YAML = 'yaml'
    JSON = 'json'


class _CfgFileInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons from the file system.
    """
    @staticmethod
    @abstractmethod
    def from_files(data_cfg_path: str,
                   model_cfg_path: str,
                   runtime_cfg_path: str,
                   serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                   encoding=_DEFAULT_ENCODING) -> ConfigurationManagerInterface:
        raise NotImplementedError

    @abstractmethod
    def to_files(self,
                 dst_dir_path: str,
                 serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                 encoding: Optional[str] = None) -> None:
        raise NotImplementedError

    def serializer2enum(serializer: Union[str, _CfgSerializer]) -> _CfgSerializer:
        """convert serializer name(string) into enum type"""
        if isinstance(serializer, str):
            try:
                serializer = _CfgSerializer(serializer)
            except:
                raise ValueError(f'{serializer} is not supported currently.')
        if not isinstance(serializer, _CfgSerializer):
            raise ValueError(f'invalid serializer type: {serializer.__class__.__name__}.')
        return serializer

# --- Role-related Configuration Interface --- 
class _RoledConfigurationInterface(ABC):
    @abstractproperty
    def role(self) -> Role:
        raise NotImplementedError

# --- Configuration Manager ---
class ConfigurationManager(Singleton,
                           ConfigurationManagerInterface,
                           ClientConfigurationManagerInterface,
                           ServerConfigurationManagerInterface,
                           _CfgYamlInterface,
                           _CfgJsonInterface,
                           _CfgFileInterface,
                           _RoledConfigurationInterface):
    def __init__(self,
                 data_config: RawConfigurationDict = _DEFAULT_D_CFG,
                 model_config: RawConfigurationDict = _DEFAULT_MDL_CFG,
                 runtime_config: RawConfigurationDict = _DEFAULT_RT_CFG,
                 thread_safe: bool = False) -> None:
        self._d_cfg: _DataConfig = _DataConfig(data_config)
        self._mdl_cfg: _ModelConfig = _ModelConfig(model_config)
        self._rt_cfg: _RuntimeConfig = _RuntimeConfig(runtime_config)

        self._lock: Optional[Lock] = Lock() if thread_safe else None
        self._init_file_names()
        self._encoding = _DEFAULT_ENCODING
        self.__init_role()

    def _init_file_names(self,
                         data_config_filename: str = DEFAULT_D_CFG_FILENAME,
                         model_config_filename: str = DEFAULT_MDL_CFG_FILENAME,
                         runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME) -> None:
        self._d_cfg_filename = data_config_filename
        self._mdl_cfg_filename = model_config_filename
        self._rt_cfg_filename = runtime_config_filename
        # TODO(fgh) add unit tests for this method in test_config.py

    @property
    def encoding(self) -> str:
        """the encoding scheme during (de)serialization."""
        return self._encoding

    @encoding.setter
    def encoding(self, encoding) -> None:
        self._encoding = encoding

    @property
    def data_config_filename(self) -> str:
        return self._d_cfg_filename

    @data_config_filename.setter
    @check_filename(1)
    def data_config_filename(self, filename: str) -> None:
        self._d_cfg_filename = filename

    @property
    def model_config_filename(self) -> str:
        return self._mdl_cfg_filename

    @model_config_filename.setter
    @check_filename(1)
    def model_config_filename(self, filename: str) -> None:
        self._mdl_cfg_filename = filename

    @property
    def runtime_config_filename(self) -> str:
        return self._rt_cfg_filename

    @runtime_config_filename.setter
    @check_filename(1)
    def runtime_config_filename(self, filename: str) -> None:
        self._rt_cfg_filename = filename

    def _thread_safe(self) -> bool:
        # TODO(fgh) add thread safety to self._x_cfg attributes.
        # Skip this if there's no modification to self._x_cfg.
        # Currently, except for config filenames, there's no
        # modification towards a constructed ConfiguraitonManger object.
        return self._lock is not None

    @property
    def data_config(self) -> _DataConfig:
        return self._d_cfg

    @property
    def model_config(self) -> _ModelConfig:
        return self._mdl_cfg

    @property
    def runtime_config(self) -> _RuntimeConfig:
        return self._rt_cfg

    @property
    def num_of_clients_contacted_per_round(self) -> int:
        """the number of clients selected to participate the main
        federated process in each round.
        """
        return int(self._rt_cfg.client_num * self._mdl_cfg.C)

    @staticmethod
    def from_yamls(data_cfg_stream: _Stream,
                   model_cfg_stream: _Stream,
                   runtime_cfg_stream: _Stream) -> ConfigurationManagerInterface:
        d_cfg = yaml.load(data_cfg_stream)
        mdl_cfg = yaml.load(model_cfg_stream)
        rt_cfg = yaml.load(runtime_cfg_stream)
        return ConfigurationManager(d_cfg, mdl_cfg, rt_cfg)

    def to_yamls(self) -> Tuple[str, str, str]:
        d_cfg = self.data_config.inner
        mdl_cfg = self.model_config.inner
        rt_cfg = self.runtime_config.inner
        return yaml.dump(d_cfg), yaml.dump(mdl_cfg), yaml.dump(rt_cfg)

    @staticmethod
    def from_jsons(data_cfg_stream: _Stream,
                   model_cfg_stream: _Stream,
                   runtime_cfg_stream: _Stream) -> ConfigurationManagerInterface:
        d_cfg = json.loads(data_cfg_stream)
        mdl_cfg = json.loads(model_cfg_stream)
        rt_cfg = json.loads(runtime_cfg_stream)
        return ConfigurationManager(d_cfg, mdl_cfg, rt_cfg)

    def to_jsons(self) -> Tuple[str, str, str]:
        d_cfg = self.data_config.inner
        mdl_cfg = self.model_config.inner
        rt_cfg = self.runtime_config.inner
        return json.dumps(d_cfg), json.dumps(mdl_cfg), json.dumps(rt_cfg)

    @staticmethod
    def from_files(data_cfg_path: str,
                   model_cfg_path: str,
                   runtime_cfg_path: str,
                   serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                   encoding=_DEFAULT_ENCODING) -> ConfigurationManagerInterface:
        serializer = _CfgFileInterface.serializer2enum(serializer)
        if serializer == _CfgSerializer.YAML:
            return _CfgYamlInterface.load_yaml_configs_from_files(
                data_cfg_path, model_cfg_path, runtime_cfg_path, encoding=encoding)
        elif serializer == _CfgSerializer.JSON:
            return _CfgJsonInterface.load_json_configs_from_files(
                data_cfg_path, model_cfg_path, runtime_cfg_path, encoding=encoding)
        else:
            raise NotImplementedError

    def to_files(self,
                 dst_dir_path: str,
                 serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                 encoding: Optional[str] = None) -> None:
        serializer = _CfgFileInterface.serializer2enum(serializer)
        d_cfg = self.data_config.inner
        mdl_cfg = self.model_config.inner
        rt_cfg = self.runtime_config.inner

        d_filname = self.data_config_filename
        mdl_filename = self.model_config_filename
        rt_filename = self.runtime_config_filename
        encoding = encoding or self.encoding

        if serializer == _CfgSerializer.YAML:
            return _CfgYamlInterface.save_yaml_configs_to_files(
                d_cfg, mdl_cfg, rt_cfg,
                dst_dir_path,
                d_filname, mdl_filename, rt_filename,
                encoding=encoding)
        elif serializer == _CfgSerializer.JSON:
            return _CfgJsonInterface.save_json_configs_to_files(
                d_cfg, mdl_cfg, rt_cfg,
                dst_dir_path,
                d_filname, mdl_filename, rt_filename,
                encoding=encoding)
        else:
            raise NotImplementedError

    def __init_role(self) -> None:
        self._role_set = False  # whether the role of this entity has been set.

    @property
    def role(self) -> Role:
        """return the role of this runtime entity.

        Raises:
            AttributeError: called without role configured.

        Returns:
            Role: the role of this runtime entity.
        """
        if not self._role_set:
            raise AttributeError('the role of this node has not been set yet.')
        return self._role

    @role.setter
    def role(self, role: Role) -> None:
        """set the role of this runtime entity.
        This method should be called only once.
        It is recommoned to be set as soon as the role of this runtime could be known.

        Args:
            role (Role): the role which this entity should be.

        Raises:
            AttributeError: called more than once.
        """
        if self._role_set:
            raise AttributeError('the role of a node can only be set once.')
        self._role = role
        self._role_set = True
