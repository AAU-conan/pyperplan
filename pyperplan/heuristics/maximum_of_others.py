from pyperplan.cli import cli_register
from pyperplan.heuristics.heuristic_base import Heuristic


@cli_register()
class MaximumOf(Heuristic):
    """
    A heuristic that returns the maximum value of a list of other heuristics.
    """

    def __init__(self, task, *heuristics: type[Heuristic]):
        super().__init__(task)
        self.heuristics = [h(task) for h in heuristics]

    def __call__(self, node):
        return max(h(node) for h in self.heuristics)
