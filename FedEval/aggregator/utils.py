from typing import Callable, Iterable
from numpy import ndarray, concatenate, expand_dims

from .ModelWeight import ModelWeights


def stack_layers(layers: Iterable[ndarray]) -> ndarray:
    """stack the given layers into a single array

    Args:
       layers (Iterable[ndarray]): the layers to stack, ordered like [layer1, layer2, ...]

    Returns:
       ndarray: the stacked layers, with shape (len(layers), *layers[0].shape)

    Example:
        >>> _stack_layers([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
        array([[[1, 2],
                [3, 4]],
                [[5, 6],
                [7, 8]]])
    """
    return concatenate([expand_dims(layer, axis=0) for layer in layers], axis=0)


def layerwise_aggregate(params: Iterable[ModelWeights], func: Callable[[ndarray], ndarray]) -> ModelWeights:
    """
    Aggregate the given client-side params layer-wise.

    Args:
        params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        func (Callable[[ndarray], ndarray]): the aggregation function to apply to stacked layers from each client. And they will be called with axis=0.

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the params

    Example:
        >>> params = [[np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])], [np.array([[9, 10], [11, 12]]), np.array([[13, 14], [15, 16]])]]
        >>> _layerwise_aggregate(params, np.mean)
        [array([[5., 6.],
               [7., 8.]]),
        array([[9., 10.],
               [11., 12.]])]

        >>> _layerwise_aggregate(params, np.median)
        [array([[1., 2.],
               [3., 4.]]),
        array([[5., 6.],
               [7., 8.]])]
    """
    return [func(stack_layers(layers), axis=0) for layers in zip(*params)]
