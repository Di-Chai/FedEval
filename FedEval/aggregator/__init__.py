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


def aggregate_weighted_average(client_params: Iterable[ModelWeights], weights: Iterable[Union[float, int]]) -> ModelWeights:
    """return the weighted average of the given client-side params according to the given weights.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
            clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
            aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """
    assert len(client_params) == len(weights), 'the number of client params and weights should be the same'
    weights = normalize_weights(weights)
    return np.sum([client_params[i] * weights[i] for i in range(len(client_params))], axis=0)


def coordinate_wise_median(client_params: Iterable[ModelWeights])-> ModelWeights:
    """return the coordinate-wise median of the given client-side params.
    
    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]
        
    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """
    return np.median(client_params, axis=0)


def trimmed_coordinate_wise_median(client_params: Iterable[ModelWeights], trim_ratio: float = 0.05) -> ModelWeights:
    """
    Return the coordinate-wise median of the given client-side params after trimming a certain ratio
    of the extreme parameter values.

    Args:
        client_params (Iterable[ModelWeights]): The weights from different clients, ordered like [params1, params2, ...].
        trim_ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 1.
            Defaults to 0.05.

    Raises:
        ValueError: If trim_ratio is in [0, 0.5).
            
    Returns:
        ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
    """
    if not (0 <= trim_ratio and trim_ratio < 0.5):
        raise ValueError('Trim ratio must be in [0, 0.5).')

    stacked_params = np.stack(client_params)
    num_params = stacked_params.shape[1]
    num_trim = int(num_params * trim_ratio)

    # Check if all but one parameter will be trimmed, and adjust num_trim accordingly
    if num_trim*2 >= num_params:
        num_trim = 0 if num_params <= 2 else (num_params-1) // 2 
  
    # Sort the parameter values along each axis (coordinate)
    sorted_params = np.sort(stacked_params, axis=0)
    # Trim the specified ratio of lowest and highest parameters along each axis
    trimmed_params = sorted_params[num_trim:-num_trim, :]
    return np.median(trimmed_params, axis=0)


class ParamAggregator:
    """A class to aggregate the parameters from different clients."""
    def __init__(self, client_params: Iterable[ModelWeights]) -> None:
        self._client_params: Iterable[ModelWeights] = client_params
    
    def average(self) -> ModelWeights:
        """return the average of the given client-side params.
        
        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return np.stack(self._client_params).mean(axis=0)

    def weighted_average(self, weights: Iterable[Union[float, int]]) -> ModelWeights:
        """return the weighted average of the given client-side params according to the given weights.

        Args:
            weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
                clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
                aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return aggregate_weighted_average(self._client_params, weights)
    
    def coordinate_wise_median(self) -> ModelWeights:
        """return the coordinate-wise median of the given client-side params.
        
        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return coordinate_wise_median(self._client_params)
    
    def trimmed_coordinate_wise_median(self, trim_ratio: float = 0.05) -> ModelWeights:
        """
        Return the coordinate-wise median of the given client-side params after trimming a certain ratio
        of the extreme parameter values.

        Args:
            trim_ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 1.
                Defaults to 0.05.

        Raises:
            ValueError: If trim_ratio is in [0, 0.5).
                
        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return trimmed_coordinate_wise_median(self._client_params, trim_ratio)