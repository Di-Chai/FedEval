from abc import ABCMeta

from ..strategy import *
from .role import Role


class Node(metaclass=ABCMeta):
    """the basic of a node in federated learning network.
    This class should be inherited instead of directly instantiate. 

    Attributes:
        name (str): the name of this node instance.
        data_config (Dict[str, Any]): cache of the given data configuration.
        model_config (Dict[str, Any]): cache of the given model configuration.
        runtime_config (Dict[str, Any]): cache of the given runtime configuration.
        fed_model (FedStrategyInterface): federated strategy instance
            constructed according to the given configurations.
    """

    def __init__(self, name: str, data_config, model_config, runtime_config, role: Role) -> None:
        super().__init__()
        self._bind_configs(name, data_config, model_config, runtime_config, role)
        if role == Role.Server:
            self._construct_fed_model()
        # client-side fed_model construction is defined in container init procedure.

    def _bind_configs(self, name: str, data_config, model_config, runtime_config, role: Role):
        """bind the given configurations with self.

        Args:
            name (str): the name of this node.
            data_config: the given data configuration.
            model_config: the given model configuraion.
            runtime_config: the given runtime configuration.
            role (Role): the role of this node.
        """
        self.name = name
        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self._role: Role = role

    def _construct_fed_model(self):
        """Construct a federated model according to `self.model_config` and bind it to `self.fed_model`.
            This method only works after `self._bind_configs()`.
        """
        fed_model_type: type = eval(self.model_config['FedModel']['name'])
        self.fed_model: FedStrategyInterface = fed_model_type(
            role=self._role, data_config=self.data_config,
            model_config=self.model_config, runtime_config=self.runtime_config)
