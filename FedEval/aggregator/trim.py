from typing import Iterable
from .ModelWeight import ModelWeights
from numpy import stack, sort


def trim_params(params: Iterable[ModelWeights], ratio: float = 0.05) -> Iterable[ModelWeights]:
    """
    Return the params after trimming a certain ratio of the extreme parameter values.

    Args:
        params (Iterable[ModelWeights]): The weights of models, ordered like [params1, params2, ...].
        ratio (float, optional): The ratio of extreme parameter values to trim. Should be between 0 and 0.5.
            Defaults to 0.05.

    Raises:
        ValueError: If trim_ratio is in [0, 0.5).

    Returns:
        Iterable[ModelWeights]: trimmed params
    """
    if not (0 <= ratio and ratio < 0.5):
        raise ValueError('Trim ratio must be in [0, 0.5).')

    stacked_params = stack(params)
    num_params = stacked_params.shape[1]
    num_trim = int(num_params * ratio)

    # Check if all but one parameter will be trimmed, and adjust num_trim accordingly
    if num_trim*2 >= num_params:
        num_trim = 0 if num_params <= 2 else (num_params-1) // 2

    sorted_params = sort(stacked_params, axis=0)
    return sorted_params[num_trim:-num_trim, :]
