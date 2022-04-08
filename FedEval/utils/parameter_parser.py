import os
import pickle
import hickle
from abc import ABCMeta, abstractmethod
from typing import Any, Mapping, Tuple

import numpy as np
import tensorflow as tf

from ..config.configuration import ConfigurationManager
from ..dataset import get_data_shape
from ..model import *

Data = Any
XYData = Mapping[str, Data] # {'x': Data, 'y': Data}

class ParamParserInterface(metaclass=ABCMeta):
    """ Abstract class of ParamParser, containing basic params parse functions.

    Raises:
        NotImplementedError: raised when methods in this class was not implemented.
    """

    @staticmethod
    @abstractmethod
    def parse_model() -> tf.keras.Model:
        """construct a tensorflow model according to the model configuration.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            tf.keras.Model: the model constructed.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_data(client_id) -> Tuple[XYData, XYData, XYData]:
        """load data for train/test/validation purpose according to the data configuration.

        Attributes:
            client_id: the id of the client which issued this data parse procedure.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Tuple[XYData, XYData, XYData]: [data for train, data for test, data for validation]
        """
        raise NotImplementedError


class ParamParser(ParamParserInterface):
    """an implentation of ParamParserInterface."""

    @staticmethod
    def parse_model():
        x_size, y_size = get_data_shape(ConfigurationManager().data_config.dataset_name)
        cfg_mgr = ConfigurationManager()
        # (0) Test, Config the GPU
        if cfg_mgr.runtime_config.gpu_enabled:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e) # TODO(fgh) expose this exception

        mdl_cfg = cfg_mgr.model_config
        mdl_cfg_inner = mdl_cfg.inner['MLModel']
        optimizer = tf.keras.optimizers.get(mdl_cfg.optimizer_name)
        for key, value in mdl_cfg_inner.get('optimizer', {}).items():
            if key != 'name' and hasattr(optimizer, key):
                setattr(optimizer, key, value)
                print('Set attribute %s=%s in optimizer' % (key, value), optimizer)

        ml_model: tf.keras.Model = eval(mdl_cfg.ml_method_name)(target_shape=y_size, **mdl_cfg_inner)
        ml_model.compile(loss=mdl_cfg.loss_calc_method,
                         metrics=mdl_cfg.metrics, optimizer=optimizer, run_eagerly=True)

        if mdl_cfg.ml_method_name == 'MLP':
            x_size = (None, int(np.prod(x_size[1:])))
        ml_model.build(input_shape=x_size)

        # Run predict output, such that the model could be saved.
        # And this should be a issue of TF
        ml_model.compute_output_shape(x_size)

        return ml_model

    @staticmethod
    def parse_data(client_id) -> Tuple[XYData, XYData, XYData]:
        d_cfg = ConfigurationManager().data_config
        data_path = os.path.join(d_cfg.dir_name, f'client_{client_id}.pkl')
        data = hickle.load(data_path)

        train_data = {'x': data.get('x_train', []), 'y': data.get('y_train', [])}
        val_data = {'x': data.get('x_val', []), 'y': data.get('y_val', [])}
        test_data = {'x': data.get('x_test', []), 'y': data.get('y_test', [])}
        return train_data, val_data, test_data
