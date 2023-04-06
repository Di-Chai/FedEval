"""
Krum Implementations

Original implementations are from git@github.com:LPD-EPFL/AggregaThor.git.
It is rewritten in numpy and scipy.

"""

from typing import Iterable, Optional
from numpy import sum as np_sum, mean, argpartition
from scipy.spatial.distance import pdist, squareform

from .ModelWeight import ModelWeights
from .utils import layerwise_aggregate, stack_layers


def _check_select(select: int, num_params: int) -> int:
    if select is None or select == num_params:
        return num_params
    assert isinstance(select, int), 'select must be an integer'
    if not(select > 0 and select <= num_params):
        raise ValueError("Invalid number of selected params. It should be in range (0, len(params)].")
    return select

def krum_select_params(params: Iterable[ModelWeights], select: Optional[int] = 1, dist_metric: str = 'euclidean') -> Iterable[ModelWeights]:
    _check_select(select, len(params))
    layerwise_distances = list()
    for layers in zip(*params):
        condensed_distances = pdist(stack_layers(layers), metric=dist_metric)
        layerwise_distances.append(squareform(condensed_distances))

    distances = np_sum(layerwise_distances, axis=0)
    scores = np_sum(distances, axis=0)
    selected_idx = argpartition(scores, select)[:select]
    return [params[idx] for idx in selected_idx]

def krum(params: Iterable[ModelWeights], select: Optional[int] = 1, dist_metric: str = 'euclidean') -> ModelWeights:
    selected_params = krum_select_params(params, select, dist_metric)
    return layerwise_aggregate(selected_params, mean)
