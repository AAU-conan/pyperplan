from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pyperplan.cli import cli_register


if TYPE_CHECKING:
    from pyperplan.search.searchspace import SearchNode
    from pyperplan.task import Task


class Pruning(ABC):
    def __init__(self, task: "Task"):
        self.task = task

    @abstractmethod
    def prune(self, node: "SearchNode") -> bool:
        pass


@cli_register()
class NonePruning(Pruning):
    """
    A pruning method that does not prune any nodes.
    """

    def __init__(self, task: "Task"):
        super().__init__(task)

    def prune(self, node: "SearchNode"):
        return False
