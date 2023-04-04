from typing import Iterable, Union
from numpy import ndarray, array, float as np_float, sum as np_sum, mean

from .ModelWeight import ModelWeights
from .trim import trim_params


def normalize_weights(weights: Iterable[Union[float, int]]) -> ndarray:
    """normalize the given weights so that its summation equals to 1.0

    Args:
        weights (Iterable[Union[float, int]]): a Iterable object filled with non-negative numbers.

    Returns:
        np.ndarray: an non-negative array that sums up to 1.0
    """
    weights = array(weights).astype(np_float)
    assert (weights >= 0).all(
    ), 'all the numbers in the given weight should be non-negative'
    return weights / weights.sum()


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
    weights = normalize_weights(weights)
    return np_sum([client_params[i] * weights[i] for i in range(len(client_params))], axis=0)


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
    return mean(trimmed_params, axis=0)
