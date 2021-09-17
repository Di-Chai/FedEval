from abc import ABCMeta


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
