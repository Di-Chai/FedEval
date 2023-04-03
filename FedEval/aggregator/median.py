from typing import Iterable
from .ModelWeight import ModelWeights
from numpy import median, stack, sort


def coordinate_wise_median(client_params: Iterable[ModelWeights]) -> ModelWeights:
    """return the coordinate-wise median of the given client-side params.

    Args:
        client_params (Iterable[ModelWeights]): the weights form different clients, ordered like [params1, params2, ...]

    Returns:
        ModelWeights: the aggregated parameters which have the same format with any instance from the client_params
    """
    return median(client_params, axis=0)


def trimmed_coordinate_wise_median(client_params: Iterable[ModelWeights], ratio: float = 0.05) -> ModelWeights:
    """
    Return the coordinate-wise median of the given client-side params after trimming a certain ratio
    of the extreme parameter values.

    Args:
        client_params (Iterable[ModelWeights]): The weights from different clients, ordered like [params1, params2, ...].
        ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 1.
            Defaults to 0.05.

    Raises:
        ValueError: If trim_ratio is in [0, 0.5).

    Returns:
        ModelWeights: The aggregated parameters which have the same format with any instance from the client_params.
    """
    if not (0 <= ratio and ratio < 0.5):
        raise ValueError('Trim ratio must be in [0, 0.5).')

    stacked_params = stack(client_params)
    num_params = stacked_params.shape[1]
    num_trim = int(num_params * ratio)

    # Check if all but one parameter will be trimmed, and adjust num_trim accordingly
    if num_trim*2 >= num_params:
        num_trim = 0 if num_params <= 2 else (num_params-1) // 2

    # Sort the parameter values along each axis (coordinate)
    sorted_params = sort(stacked_params, axis=0)
    # Trim the specified ratio of lowest and highest parameters along each axis
    trimmed_params = sorted_params[num_trim:-num_trim, :]
    return median(trimmed_params, axis=0)
