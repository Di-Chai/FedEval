from typing import Iterable
from numpy import median

from .trim import trim_params
from .ModelWeight import ModelWeights
from .utils import layerwise_aggregate

def coordinate_wise_median(client_params: Iterable[ModelWeights]) -> ModelWeights:
    """return the coordinate-wise median of the given client-side params.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """
    return layerwise_aggregate(client_params, median)


def trimmed_coordinate_wise_median(client_params: Iterable[ModelWeights], ratio: float = 0.05) -> ModelWeights:
    """
    Return the coordinate-wise median of the given client-side params after trimming a certain ratio
    of the extreme parameter values.

    Args:
        client_params (Iterable[ModelWeights]): The weights from different clients, ordered like [params1, params2, ...].
        ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 0.5.
            Defaults to 0.05.

    Raises:
        ValueError: If trim_ratio is in [0, 0.5).

    Returns:
        ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
    """
    trimmed_params = trim_params(client_params, ratio)
    return coordinate_wise_median(trimmed_params)
