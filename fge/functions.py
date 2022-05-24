from typing import Dict, Tuple, Any, List, Set, Callable

import itertools
import numpy as np
from .utils import flatten

def g_template(
        nodes_to_run: List[Tuple[Any] | int], 
        siv_scores: np.ndarray,
        siv: np.ndarray,
        scores: Dict[Tuple[Any], float],
        values: Dict[Tuple[Any], float],
    ):
    # should return `scores` and `values`
    
    return NotImplementedError('Template')


def g_base(
        nodes_to_run: List[Tuple[Any] | int], 
        siv_scores: np.ndarray,
        siv: np.ndarray,
        scores: Dict[Tuple[Any], float],
        values: Dict[Tuple[Any], float],
    ):
    for cmbs in itertools.combinations(nodes_to_run, 2):
        if cmbs not in scores.keys():
            r, c = list(zip(*itertools.product(flatten(cmbs), flatten(cmbs))))
            scores[cmbs] = siv_scores[r, c].sum()
            values[cmbs] = siv[r, c].sum()
    return scores, values