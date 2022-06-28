from typing import List, Any, Generator

def flatten(li: List[Any]) -> Generator:
    """flatten nested list

    ```python
    x = [[[1], 2], [[[[3]], 4, 5], 6], 7, [[8]], [9], 10]

    print(type(flatten(x)))
    # <generator object flatten at 0x00000212BF603CC8>
    print(list(flatten(x)))
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ```
    Args:
        li (List[Any]): any kinds of list

    Yields:
        Generator: flattened list generator
    """
    for ele in li:
        if isinstance(ele, list) or isinstance(ele, tuple):
            yield from flatten(ele)
        else:
            yield ele


def c_statistic_harrell(y_true, y_pred):
    """
    Ref: https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
    The C-statistic measures how well we can order people by their survival time (1.0 is a perfect ordering).
    """
    total = 0
    matches = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if y_true[j] > 0 and abs(y_true[i]) > y_true[j]:
                total += 1
                if y_pred[j] > y_pred[i]:
                    matches += 1
    return matches/total


