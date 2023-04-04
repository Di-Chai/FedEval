from typing import Iterable, Union
import numpy as np

from .ModelWeight import ModelWeights
from .mean import weighted_average, trimmed_mean
from .median import coordinate_wise_median, trimmed_coordinate_wise_median


class ParamAggregator:
    """A class to aggregate the parameters from different clients."""

    def __init__(self, params: Iterable[ModelWeights]) -> None:
        self._params: Iterable[ModelWeights] = params

    def average(self) -> ModelWeights:
        """return the average of the given client-side params.

        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return np.stack(self._params).mean(axis=0)

    def weighted_average(self, weights: Iterable[Union[float, int]]) -> ModelWeights:
        """return the weighted average of the given client-side params according to the given weights.

        Args:
            weights (Iterable[Union[float, int]]): aggregate weights of different clients, usually set according to the
                clients' training sample size. E.g., A, B, and C have 10, 20, and 30 images, then the
                aggregate_weights can be `[1/6, 1/3, 1/2]` or `[10, 20, 30]`. 

        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return weighted_average(self._params, weights)

    def median(self) -> ModelWeights:
        """return the coordinate-wise median of the given client-side params.

        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return coordinate_wise_median(self._params)

    def trimmed_median(self, ratio: float = 0.05) -> ModelWeights:
        """
        Return the coordinate-wise median of the given client-side params after trimming a certain ratio
        of the extreme parameter values.

        Args:
            ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 1.
                Defaults to 0.05.

        Raises:
            ValueError: If trim_ratio is in [0, 0.5).

        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return trimmed_coordinate_wise_median(self._params, ratio)

    def trimmed_mean(self, ratio: float = 0.05) -> ModelWeights:
        """
        Return the coordinate-wise mean of the given client-side params after trimming a certain ratio
        of the extreme parameter values.

        Args:
            ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 1.
                Defaults to 0.05.

        Raises:
            ValueError: If trim_ratio is in [0, 0.5).

        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return trimmed_mean(self._params, ratio)
