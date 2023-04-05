"""Norm clipping aggregator.
Proposed in Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, and H Brendan McMahan. 2019. Can you really backdoor federated learning?. In NeurIPS FL Workshop.
"""


from typing import Iterable, Optional

from numpy import sum as np_sum
from numpy.linalg import norm

from .ModelWeight import ModelWeights


def _weight_delta(param: ModelWeights, prev_param: ModelWeights) -> ModelWeights:
    return [p - prev_p for p, prev_p in zip(param, prev_param)]


def _norm_clip(model_updates: Iterable[ModelWeights], threshold: Optional[float] = 0.5) -> ModelWeights:
    """Clip the model updates by norm.

    $$\delta_{w_{t+1}} = \sum_{k\in S_t} \frac{\delta_{w_{t+1}^k}}{max(1, ||\delta_{w_{t+1}^k}||_2/Threshold)}$$

    Args:
        model_updates (Iterable[ModelWeights]): the updates from different clients, ordered like [update1, update2, ...]
        threshold (Optional[float], optional): the threshold of the norm. Defaults to 0.5.

    Raises:
        ValueError: If threshold is invalid.

    Returns:
        ModelWeights: the clipped and aggregated update
    """

    if threshold == None:
        threshold = 1
    if threshold <= 0:
        raise ValueError("Invalid threshold. It should be positive.")

    return [np_sum([layer / max(1, norm(layer, ord=2) / threshold) for layer in layers], axis=0)
            for layers in zip(*model_updates)]


def norm_clip(client_params: Iterable[ModelWeights], server_param: ModelWeights, threshold: Optional[float] = 0.5) -> ModelWeights:
    """Aggregate the given client-side params by norm clipping.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        server_param (ModelWeights): the weights at the server-side
        threshold (Optional[float], optional): the threshold of the norm. Defaults to 0.5.

    Raises:
        ValueError: If threshold is invalid.

    Returns:
        ModelWeights: the aggregated model weights.
    """
    model_updates = [_weight_delta(param, server_param)
                     for param in client_params]
    clipped_updates = _norm_clip(model_updates, threshold)
    return [p + u for p, u in zip(server_param, clipped_updates)]
