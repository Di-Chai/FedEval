import random
from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import Iterable, List, Mapping, Optional, Tuple, Union

from ..aggregater import ModelWeights, aggregate_weighted_average
from ..callbacks import *
from ..config import ClientId, ConfigurationManager, Role
from ..model import *
from ..utils import ParamParser, ParamParserInterface


class HostParamsType(Enum):
    Uniform = 'uniform'
    Personalized = 'personalized'


class FedStrategyHostInterface(metaclass=ABCMeta):

    @abstractproperty
    def host_params_type(self):
        """
        Returns the same parameters for all parties or personalized ML model
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_host_download_info(self) -> Tuple[ModelWeights, str]:
        """get the host download information,
           e.g., model params/weights from its machine/deep learning model.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            ModelWeights: the weights/params of its inner machine/deep learning model.
            TODO(fgh): the meaning of the str in the return type/
        """
        raise NotImplementedError

    @abstractmethod
    def update_host_params(self, client_params: Iterable[ModelWeights], aggregate_weights: Iterable[Union[float, int]]) -> None:
        """update central server's model params/weights with
        the aggregated params received from clients.

        Args:
            client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
            aggregate_weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
                clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
                aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    @abstractmethod
    def host_exit_job(self, host):
        """do self-defined finishing jobs before the shutdown of the central server.

        Args:
            host: TODO(fgh)

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    @abstractmethod
    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        """select clients from the given ones for training purpose.

        Args:
            ready_clients (List[ClientId]): the id list of clients that are ready for training.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            List[ClientId]: the id list of the selected clients.
        """
        raise NotImplementedError

    @abstractmethod
    def host_select_evaluate_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        """select clients from the given ones for evaluation purpose.

        Args:
            ready_clients (List[ClientId]): the id list of clients that are ready for evaluaion.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            List[ClientId]: the id list of the selected clients.
        """
        raise NotImplementedError


class FedStrategyPeerInterface(metaclass=ABCMeta):
    @abstractmethod
    def set_host_params_to_local(self, host_params: ModelWeights, current_round: int):
        """update the current local ML/DL model's params with params received
        from the central server.

        Args:
            host_params (ModelWeights): params received from the central server.
            current_round (int): the current round number

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

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

    @abstractmethod
    def local_evaluate(self) -> Mapping[str, Union[int, float]]:
        """evaluate and test the model received from the central server.

        Called by the selected clients.

        Raises:
            NotImplementedError: raised when called but not overriden.

        Returns:
            Mapping[str, Union[int, float]]: evaluation & test metrics. 
        """
        raise NotImplementedError

    @abstractmethod
    def client_exit_job(self, client):
        """do self-defined finishing jobs before the shutdown of the local clients.

        Args:
            client: TODO(fgh)

        Raises:
            NotImplementedError: raised when called but not overriden.
        """
        raise NotImplementedError

    @abstractmethod
    def load_data_with(self, client_id) -> None:
        """load data with respect to the id of this client.

        TODO(fgh): move this duty into data related modules.

        Args:
            client_id: the id of this client.

        Raises:
            NotImplementedError: raised when called but not implemented.
        """
        raise NotImplementedError


class FedStrategyInterface(FedStrategyHostInterface, FedStrategyPeerInterface):
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

    def __init__(self, param_parser_type: type = ParamParser, logger=None):
        self._param_parser: ParamParserInterface = param_parser_type()
        if not isinstance(self._param_parser, ParamParserInterface):
            raise ValueError(f"param_parser_class({type(param_parser_type)})" 
                             + f"should implement {type(ParamParserInterface)}")
        self._init_states()
        self._init_model()
        self._config_callback()
        self._init_host_params_type()
        self.logger = logger

    def _init_host_params_type(self):
        self._host_params_type = HostParamsType.Uniform

    def set_logger(self, logger) -> None:
        self.logger = logger

    def _init_model(self):
        self.ml_model = self.param_parser.parse_model()

    def _init_states(self):
        self.current_round: Optional[int] = None
        cfg_mgr = ConfigurationManager()
        role = cfg_mgr.role
        if role == Role.Server:
            self.params = None
            self.gradients = None
            # TMP
            self.num_clients_contacted_per_round = cfg_mgr.num_of_clients_contacted_per_round
            self.train_selected_clients = None
        elif role == Role.Client:
            self.local_params_pre = None
            self.local_params_cur = None
        else:
            raise NotImplementedError

    def _config_callback(self):
        # TODO(chaidi): Add the callback model for implementing attacks
        # TODO(fgh): add 'callback' in configurations
        strategy = ConfigurationManager().model_config.strategy_config
        callback: Optional[CallBack] = strategy.get('callback')
        self.callback = eval(callback)() if callback else None

    def _has_callback(self) -> bool:
        return self.callback is not None and isinstance(self.callback, CallBack)

    @property
    def param_parser(self) -> ParamParserInterface:
        return self._param_parser

    @property
    def host_params_type(self):
        return self._host_params_type

    @param_parser.setter
    def param_parser(self, value: ParamParserInterface):
        self._param_parser = value

    def load_data_with(self, client_id) -> None:
        if ConfigurationManager().role != Role.Client:
            raise TypeError(f"This {self.__class__.__name__}'s role is not a {Role.Client.value}.")
        self._client_id = client_id
        self.train_data, self.val_data, self.test_data = self.param_parser.parse_data(self._client_id)

    def retrieve_host_download_info(self) -> ModelWeights:
        if self.params is None:
            return self.ml_model.get_weights()
        else:
            return self.params

    def update_host_params(self, client_params, aggregate_weights) -> None:
        if self._has_callback():
            client_params = self.callback.on_host_aggregate_begin(client_params)
        self.params = aggregate_weighted_average(client_params, aggregate_weights)

    def host_exit_job(self, host):
        if self._has_callback():
            self.callback.on_host_exit()

    def host_select_train_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
        self.train_selected_clients = random.sample(list(ready_clients), self.num_clients_contacted_per_round)
        return self.train_selected_clients

    def host_select_evaluate_clients(self, ready_clients: List[ClientId]) -> List[ClientId]:
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
        mdl_cfg = ConfigurationManager().model_config
        train_log = self.ml_model.fit(
            x=self.train_data['x'], y=self.train_data['y'],
            epochs=mdl_cfg.E,
            batch_size=mdl_cfg.B,
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self.ml_model.get_weights()
        return train_loss, len(self.train_data['x'])

    def _retrieve_local_params(self):
        return self.ml_model.get_weights()

    def retrieve_local_upload_info(self) -> ModelWeights:
        model_weights = self._retrieve_local_params()
        if self._has_callback():
            model_weights = self.callback.on_client_upload_begin(model_weights)
        return model_weights

    def local_evaluate(self) -> Mapping[str, Union[int, float]]:
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
        evaluate['val_size'] = len(self.val_data['x'])
        evaluate['test_size'] = len(self.test_data['x'])
        return evaluate

    def client_exit_job(self, client):
        if self._has_callback():
            self.callback.on_client_exit()
