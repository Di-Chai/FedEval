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


def krum(params: Iterable[ModelWeights], select: Optional[int] = 1, dist_metric: str = 'euclidean') -> ModelWeights:
    if select == None or select == len(params):
        return layerwise_aggregate(params, mean)
    assert isinstance(select, int), 'select must be an integer'
    if not(select > 0 and select <= len(params)):
        raise ValueError("Invalid number of selected params. It should be in range (0, len(params)].")

    layerwise_distances = list()
    for layers in zip(*params):
        condensed_distances = pdist(stack_layers(layers), metric=dist_metric)
        layerwise_distances.append(squareform(condensed_distances))

    distances = np_sum(layerwise_distances, axis=0)
    scores = np_sum(distances, axis=0)
    selected_idx = argpartition(scores, select)[:select]
    selected_params = [params[idx] for idx in selected_idx]
    return layerwise_aggregate(selected_params, mean)
