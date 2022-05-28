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