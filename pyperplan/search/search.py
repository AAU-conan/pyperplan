from typing import Optional

from pyperplan.task import Operator, Task


class Search:
    """
    Interface for search algorithms.
    """

    def __init__(self, task: Task):
        pass

    def search(self) -> Optional[list[str]]:
        raise NotImplementedError("Base class does not implement this method.")
