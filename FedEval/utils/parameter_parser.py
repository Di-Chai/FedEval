import os
import pickle
from typing import Any, Mapping, Tuple, TypeVar
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

from ..dataset import get_data_shape
from ..model import *
from ..role import Role

Data = TypeVar("Data", Any)
XYData = Mapping[str, Data] # {'x': Data, 'y': Data}

class ParamParserInterface(metaclass=ABCMeta):
    """ Abstract class of ParamParser.

    Raises:
        NotImplementedError: raised when methods in this class was not implemented.
    """
    @abstractmethod
    def parse_server_addr(self, role: Role) -> Tuple[str, str]:
        """parse the address of central server from the configurations given during the
        the construction of this obejct.

        Args:
            role (Role): the role of the holder of this parameter parser.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Tuple[str, str]: [IP address, port] in a string tuple.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_model(self) -> tf.keras.Model:
        """construct a tensorflow model according to the model configuration.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            tf.keras.Model: the model constructed.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_data(self) -> Tuple[XYData, XYData, XYData]:
        """load data for train/test/validation purpose according to the data configuration.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Tuple[XYData, XYData, XYData]: [data for train, data for test, data for validation]
        """
        raise NotImplementedError

    @abstractmethod
    def parse_run_config(self) -> Mapping[str, Any]:
        """return configurations conluded from the model configuraion and the runtime configuration.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Mapping[str, Any]: consists of: 
                num_clients: the total number of clients,
                max_num_rounds: the maximum rounds of federated iterations,
                num_tolerance: TODO(fgh),
                num_clients_contacted_per_round: C * num_clients,
                rounds_between_val: the number of rounds between each validation.
        """
        raise NotImplementedError


class ParamParser(ParamParserInterface):
    """an implentation of ParamParserInterface."""
    def __init__(self, data_config, model_config, runtime_config, role:Role):
        self._role: Role = role
        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config

        self.x_size, self.y_size = get_data_shape(dataset=self.data_config.get('dataset'))

    # Basic parse functions
    def parse_server_addr(self) -> Tuple[str, str]:
        if self._role == Role.Server:
            addr, port = self.runtime_config['server']['listen'], self.runtime_config['server']['port']
        elif self._role == Role.Client:
            addr, port = self.runtime_config['server']['host'], self.runtime_config['server']['port']
        else:
            raise NotImplementedError
        return str(addr), str(port)

    # Basic parse functions
    def parse_model(self) -> tf.keras.Model:

        # (0) Test, Config the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
                )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e) # TODO expose this exception

        ml_model_config = self.model_config['MLModel']
        ml_model_name = ml_model_config.get('name')

        optimizer_config = ml_model_config.get('optimizer')
        optimizer = tf.keras.optimizers.get(optimizer_config.get('name'))
        for key, value in optimizer_config.items():
            if key != 'name' and hasattr(optimizer, key):
                setattr(optimizer, key, value)
                print('Set attribute %s=%s in optimizer' % (key, value), optimizer)

        loss = ml_model_config.get('loss')
        metrics = ml_model_config.get('metrics')

        ml_model: tf.keras.Model = eval(ml_model_name)(target_shape=self.y_size, **ml_model_config)
        ml_model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        if ml_model_name == 'MLP':
            self.x_size = (None, int(np.prod(self.x_size[1:])))
        ml_model.build(input_shape=self.x_size)

        # Run predict output, such that the model could be saved.
        # And this should be a issue of TF
        ml_model.compute_output_shape(self.x_size)

        return ml_model

    # Basic parse functions
    def parse_data(self) -> Tuple[XYData, XYData, XYData]:
        client_id = os.environ.get('CLIENT_ID', '0')
        with open(os.path.join(self.data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
            data = pickle.load(f)
        train_data = {'x': data['x_train'], 'y': data['y_train']}
        val_data = {'x': data['x_val'], 'y': data['y_val']}
        test_data = {'x': data['x_test'], 'y': data['y_test']}
        return train_data, val_data, test_data

    # Basic parse functions
    def parse_run_config(self) -> Mapping[str, Any]:
        return {
            'num_clients': self.runtime_config['server']['num_clients'],
            'max_num_rounds': self.model_config['FedModel']['max_rounds'],
            'num_tolerance': self.model_config['FedModel']['num_tolerance'],
            'num_clients_contacted_per_round': int(self.runtime_config['server']['num_clients']
                                                   * self.model_config['FedModel']['C']),
            'rounds_between_val': self.model_config['FedModel']['rounds_between_val'],
        }
