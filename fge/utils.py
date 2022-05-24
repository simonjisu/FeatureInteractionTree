from typing import List, Any

def flatten(li: List[Any]):
    for ele in li:
        if isinstance(ele, list) or isinstance(ele, tuple):
            yield from flatten(ele)
        else:
            yield ele