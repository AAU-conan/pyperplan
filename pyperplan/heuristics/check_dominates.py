import logging

from pyperplan.cli import cli_register
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.task import Task


@cli_register()
class CheckDominates(Heuristic):
    """
    A heuristic that checks if the heuristics in the list dominate each other.
    Uses last heuristic's value as return value.
    """

    def __init__(self, task: Task, *heuristics: Heuristic):
        super().__init__(task)
        self.heuristics = [h(task) for h in heuristics]

    def __call__(self, node):
        values = [h(node) for h in self.heuristics]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if i != j and values[i] < values[j]:
                    logging.error(f"{self.heuristics[i].__class__.__name__} does not dominate {self.heuristics[j].__class__.__name__}: {values[i]} < {values[j]}")
        return values[-1]
