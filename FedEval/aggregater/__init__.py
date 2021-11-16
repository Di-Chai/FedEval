from typing import Iterable, Union, Any

import numpy as np


ModelWeights = Any  # weights of DL model

def normalize_weights(weights: Iterable[Union[float, int]]) -> np.ndarray:
    """normalize the given weights so that its summation equals to 1.0

    Args:
        weights (Iterable[Union[float, int]]): a Iterable object filled with non-negative numbers.

    Returns:
        np.ndarray: an non-negative array that sums up to 1.0
    """
    weights = np.array(weights).astype(np.float)
    assert (weights >= 0).all(), 'all the numbers in the given weight should be non-negative'
    return weights / weights.sum()


def aggregate_weighted_average(client_params: Iterable[ModelWeights], aggregate_weights: Iterable[Union[float, int]]) -> ModelWeights:
    """return the weighted average of the given client-side params according to the given weights.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        aggregate_weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
            clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
            aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """

    aggregate_weights = normalize_weights(aggregate_weights)
    new_param = []
    for i in range(len(client_params[0])):
        for j in range(len(client_params)):
            if j == 0:
                new_param.append(client_params[j][i] * aggregate_weights[j])
            else:
                new_param[i] += client_params[j][i] * aggregate_weights[j]
    return new_param
