from typing import Iterable, Union
from functools import partial
from numpy import sum as mean, average

from .ModelWeight import ModelWeights
from .trim import trim_params
from .utils import layerwise_aggregate


def weighted_average(client_params: Iterable[ModelWeights], weights: Iterable[Union[float, int]]) -> ModelWeights:
    """return the weighted average of the given client-side params according to the given weights.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
            clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
            aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """
    assert len(client_params) == len(
        weights), 'the number of client params and weights should be the same'
    weighted_average = partial(average, weights=weights)
    return layerwise_aggregate(client_params, weighted_average)


def trimmed_mean(client_params: Iterable[ModelWeights], ratio: float = 0.05) -> ModelWeights:
    """
    Return the coordinate-wise mean of the given client-side params after trimming a certain ratio
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
    return layerwise_aggregate(trimmed_params, mean)
