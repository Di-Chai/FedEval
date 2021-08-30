import random
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Mapping, Optional

from callbacks import *
from model import *
from role import ContainerId, Role

from utils import ParamParser, ParamParserInterface

from .utils import aggregate_weighted_average

ModelWeights = Any  # weights of DL model

class FedStrategyInterface(metaclass=ABCMeta):
    """the interface of federated strategies.

    This class should be inherited instead of being instantiated.

    Raises:
        NotImplementedError: raised when called but not overriden.
    """

    @abstractproperty
    def param_parser(self) -> ParamParserInterface:
        """the getter of param_parser.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            ParamParserInterface: self._param_parser
        """
        raise NotImplementedError

    @param_parser.setter
    def param_parser(self, value: ParamParserInterface):
        """the setter of param_parser.

        Args:
            value (ParamParserInterface): the new param_parser.

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    # Host functions
    @abstractmethod
    def host_get_init_params(self) -> ModelWeights:
        """get the initial model params/weights from its machine/deep learning model.

        Called by the central server.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            ModelWeights: the weights/params of its inner machine/deep learning model.
        """
        raise NotImplementedError

    # Host functions
    @abstractmethod
    def update_host_params(self, client_params, aggregate_weights) -> ModelWeights:
        """update central server's model params/weights with
        the aggregated params received from clients.

        Called by the central server.

        Args:
            client_params: TODO(fgh)
            aggregate_weights: TODO(fgh)

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            ModelWeights: the updated model params/weights, equals to params of self.
        """
        raise NotImplementedError

    # Host functions
    @abstractmethod
    def host_exit_job(self, host):
        """do self-defined finishing jobs before the shutdown of the central server.

        Called by the central server.

        Args:
            host: TODO(fgh)

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    # Host functions
    @abstractmethod
    def host_select_train_clients(self, ready_clients: List[ContainerId]) -> List[ContainerId]:
        """select clients from the given ones for training purpose.

        Called by the central server.

        Args:
            ready_clients (List[ContainerId]): the id list of clients that are ready for training.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            List[ContainerId]: the id list of the selected clients.
        """
        raise NotImplementedError

    # Host functions
    @abstractmethod
    def host_select_evaluate_clients(self, ready_clients: List[ContainerId]) -> List[ContainerId]:
        """select clients from the given ones for evaluation purpose.

        Called by the central server.

        Args:
            ready_clients (List[ContainerId]): the id list of clients that are ready for evaluaion.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            List[ContainerId]: the id list of the selected clients.
        """
        raise NotImplementedError

    # Client functions
    @abstractmethod
    def set_host_params_to_local(self, host_params: ModelWeights, current_round: int):
        """update local ML/DL model params with params received from the central server.

        Called by clients.

        Args:
            host_params (ModelWeights): params received from the central server.
            current_round (int): the current round number

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    # Client functions
    @abstractmethod
    def fit_on_local_data(self):
        """fit model with local data at client side.

        Called by the selected clients.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            TODO(fgh)
        """
        raise NotImplementedError

    # Client functions
    @abstractmethod
    def retrieve_local_upload_info(self) -> ModelWeights:
        """return the information aggregated from local model
        for uploading to the central server.

        Called by the selected clients.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            ModelWeights: the local model weights/params.
        """
        raise NotImplementedError

    # Client functions
    @abstractmethod
    def local_evaluate(self) -> Mapping[str, Any]:
        """evaluate and test the model received from the central server.

        Called by the selected clients.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Mapping[str, Any]: evaluation & test metrics. 
        """
        raise NotImplementedError

    # Client functions
    @abstractmethod
    def client_exit_job(self, client):
        """do self-defined finishing jobs before the shutdown of the local clients.

        Called by one of the clients.

        Args:
            client: TODO(fgh)

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    @abstractmethod
    def set_logger(self, logger) -> None:
        """the setter of logger of this federated learning.

        Args:
            logger: an external logger.

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError


class FedStrategy(FedStrategyInterface):
    """the basic class of federated strategies."""

    def __init__(self, role: Role, data_config, model_config, runtime_config, param_parser_class=ParamParser, logger=None):
        self._init_configs(role, data_config, model_config, runtime_config, param_parser_class)
        self._init_states()
        self._config_callback()
        self.logger = logger

    def set_logger(self, logger) -> None:
        self.logger = logger

    def _init_configs(self, role: Role, data_config, model_config, runtime_config, param_parser_type: type):
        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.role: Role = role
        self._param_parser: ParamParserInterface = param_parser_type(
            data_config=self.data_config, model_config=self.model_config,
            runtime_config=self.runtime_config, role=self.role
        )
        if not isinstance(self._param_parser, ParamParserInterface):
            raise ValueError(f"param_parser_class({type(param_parser_type)})" 
                             + f"should implement {type(ParamParserInterface)}")

    def _init_model(self):
        self.ml_model = self.param_parser.parse_model()

    def _init_states(self):
        self.current_round: Optional[int] = None

        if self.role == Role.Server:
            self.params = None
            self.gradients = None
            # TMP
            run_config = self.param_parser.parse_run_config()
            self.num_clients_contacted_per_round = run_config['num_clients_contacted_per_round']
            self.train_selected_clients = None
        if self.role == Role.Client:
            # only clients parse data
            self.train_data, self.val_data, self.test_data = self.param_parser.parse_data()
            self.train_data_size = len(self.train_data['x'])
            self.val_data_size = len(self.val_data['x'])
            self.test_data_size = len(self.test_data['x'])
            self.local_params_pre = None
            self.local_params_cur = None

    def _config_callback(self):
        # TODO: Add the callback model for implementing attacks
        self.callback: Optional[CallBack] = self.model_config['FedModel'].get('callback')
        if self.callback:
            self.callback = eval(self.callback)()

    def _has_callback(self) -> bool:
        return self.callback is not None and isinstance(self.callback, CallBack)

    @property
    def param_parser(self) -> ParamParserInterface:
        return self._param_parser

    @param_parser.setter
    def param_parser(self, value: ParamParserInterface):
        self._param_parser = value

    def host_get_init_params(self) -> ModelWeights:
        self.params = self.ml_model.get_weights()
        return self.params

    def update_host_params(self, client_params, aggregate_weights) -> ModelWeights:
        if self._has_callback():
            client_params = self.callback.on_host_aggregate_begin(client_params)
        self.params = aggregate_weighted_average(client_params, aggregate_weights)
        return self.params

    def host_exit_job(self, host):
        if self._has_callback():
            self.callback.on_host_exit()

    def host_select_train_clients(self, ready_clients: List[ContainerId]) -> List[ContainerId]:
        self.train_selected_clients = random.sample(list(ready_clients), self.num_clients_contacted_per_round)
        return self.train_selected_clients

    def host_select_evaluate_clients(self, ready_clients: List[ContainerId]) -> List[ContainerId]:
        return [e for e in self.train_selected_clients if e in ready_clients]

    def set_host_params_to_local(self, host_params: ModelWeights, current_round: int):
        if self._has_callback():
            host_params = self.callback.on_setting_host_to_local(host_params)
        self.current_round = current_round
        self.ml_model.set_weights(host_params)

    def fit_on_local_data(self):
        if self._has_callback():
            self.train_data, model = self.callback.on_client_train_begin(
                data=self.train_data, model=self.ml_model.get_weights()
            )
            self.ml_model.set_weights(model)
        self.local_params_pre = self.ml_model.get_weights()
        train_log = self.ml_model.fit(
            x=self.train_data['x'], y=self.train_data['y'],
            epochs=self.model_config['FedModel']['E'],
            batch_size=self.model_config['FedModel']['B']
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self.ml_model.get_weights()
        return train_loss, self.train_data_size

    def retrieve_local_upload_info(self) -> ModelWeights:
        model_weights = self.ml_model.get_weights()
        if self._has_callback():
            model_weights = self.callback.on_client_upload_begin(model_weights)
        return model_weights

    def local_evaluate(self) -> Mapping[str, Any]:
        # val and test
        val_result = self.ml_model.evaluate(
            x=self.val_data['x'], y=self.val_data['y'])
        test_result = self.ml_model.evaluate(
            x=self.test_data['x'], y=self.test_data['y'])
        metrics_names = self.ml_model.metrics_names
        # Reformat
        evaluate = {
            'val_' + metrics_names[i]: float(val_result[i]) for i in range(len(metrics_names))}
        evaluate.update(
            {'test_' + metrics_names[i]: float(test_result[i]) for i in range(len(metrics_names))})
        # TMP
        evaluate['val_size'] = self.val_data_size
        evaluate['test_size'] = self.test_data_size
        return evaluate

    def client_exit_job(self, client):
        if self._has_callback():
            self.callback.on_client_exit()
