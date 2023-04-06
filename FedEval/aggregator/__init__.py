from typing import Iterable, Optional, Union
from numpy import mean as np_mean

from .ModelWeight import ModelWeights
from .utils import layerwise_aggregate as _layerwise_aggregate
from .mean import weighted_average, trimmed_mean
from .median import coordinate_wise_median, trimmed_coordinate_wise_median
from .krum import krum
from .norm_clipping import norm_clip
from .bulyan import bulyan


class ParamAggregator:
    """A class to aggregate the parameters from different clients."""

    def __init__(self, params: Iterable[ModelWeights]) -> None:
        self._params: Iterable[ModelWeights] = params

    def average(self) -> ModelWeights:
        """return the average of the given client-side params.

        Returns:
            ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
        """
        return _layerwise_aggregate(self._params, np_mean)

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
            ValueError: If trim_ratio is not in [0, 0.5).

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
            ValueError: If trim_ratio is not in [0, 0.5).

        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return trimmed_mean(self._params, ratio)

    def krum(self, select: Optional[int] = 1) -> ModelWeights:
        """
        Return the krum aggregate of the given client-side params.

        Args:
            select (int, optional): The number of clients to select to support multi-krum.
                If set to None, it will return an averaged one of all the params. Defaults to 1.

        Raises:
            ValueError: If select is invalid.

        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return krum(self._params, select)
    
    def bulyan(self, ratio: float = 0.05, select: Optional[int] = 1, dist_metric: str = 'euclidean') -> ModelWeights:
        """
        Return the bulyan aggregate of the given client-side params.

        Args:
            ratio (float, optional): The ratio of extreme parameter values to trim. Should be in [0, 0.5).
                Defaults to 0.05.
            select (int, optional): The number of clients to select to support multi-krum. Defaults to 1.
            dist_metric (str, optional): The distance metric to use. Defaults to 'euclidean'.

        Raises:
            ValueError: Invalid number of selected params or trim ratio.

        Returns:
            ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
        """
        return bulyan(self._params, ratio, select, dist_metric)

    def norm_clip(self, server_param: ModelWeights, threshold: Optional[float] = 0.5) -> ModelWeights:
        """Aggregate the given client-side params by norm clipping.

        Args:
            server_param (ModelWeights): the weights at the server-side
            threshold (Optional[float], optional): the threshold of the norm. Defaults to 0.5.

        Raises:
            ValueError: If threshold is invalid.

        Returns:
            ModelWeights: the aggregated model weights.
        """
        return norm_clip(self._params, server_param, threshold)
