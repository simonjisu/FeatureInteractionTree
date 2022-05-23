from typing import Dict, Tuple, Any, List, Set, Callable

def g_sum(scores: Dict[Tuple[Any] | int, float]) -> Tuple[Tuple[Any] | int, float]:
    """_summary_

    Args:
        scores (Dict[Tuple[Any], float]): _description_

    Returns:
        Tuple[Tuple[Any], float]: _description_
    """    
    l = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return l[0]

def g_abs_sum(scores: Dict[Tuple[Any] | int, float]) -> Tuple[Tuple[Any] | int, float]:
    pass