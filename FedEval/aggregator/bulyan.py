from typing import Iterable, Optional

from .ModelWeight import ModelWeights
from .mean import trimmed_mean
from .krum import krum_select_params


def bulyan(params: Iterable[ModelWeights], ratio: float = 0.05, select: Optional[int] = 1, dist_metric: str = 'euclidean') -> ModelWeights:
    """Bulyan aggregation method.
    
    Args:
        params (Iterable[ModelWeights]): List of model weights.
        ratio (float): Ratio of params to be trimmed. Defaults to 0.05.
        select (Optional[int], optional): Number of clients to be selected. Defaults to 1.
        dist_metric (str, optional): Distance metric. Defaults to 'euclidean'.

    Raises:
        ValueError: Invalid number of selected params, or invalid trim ratio.
            
    Returns:
        ModelWeights: Aggregated model weights.
    """
    selected_params = krum_select_params(params, select, dist_metric)
    return trimmed_mean(selected_params, ratio=ratio)
