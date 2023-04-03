from typing import Iterable, Union
from numpy import ndarray, array, float as np_float, sum as np_sum

from .ModelWeight import ModelWeights


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
