from abc import ABCMeta

from ..config.configuration import ConfigurationManager
from ..strategy import *
from .role import Role


class Node(metaclass=ABCMeta):
    """the basic of a node in federated learning network.
    This class should be inherited instead of directly instantiate. 

    Attributes:
        name (str): the name of this node instance.
        fed_model (FedStrategyInterface): federated strategy instance
            constructed according to the given configurations.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        role = ConfigurationManager().role
        if role == Role.Server:
            self._construct_fed_model()
        # client-side fed_model construction is defined in container init procedure.

    def _construct_fed_model(self):
        """Construct a federated model according to `self.model_config` and bind it to `self.fed_model`.
            This method only works after `self._bind_configs()`.
        """
        cfg_mgr = ConfigurationManager()
        fed_model_type: type = eval(cfg_mgr.model_config.strategy_name)
        self.fed_model: FedStrategyInterface = fed_model_type()
