import copy
import logging
import time

from pyperplan.cli import cli_register
from pyperplan.pruning.dominance_analysis import DominanceAnalysis
from pyperplan.pruning.pruning import Pruning
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactoredTaskState
from pyperplan.task_transformation.noop_task_transformation import (
    NoopTaskTransformation,
)


class DominanceDatabase:

    def __init__(self, task: FactoredTask, dominance_analysis: DominanceAnalysis):
        self.task = task
        self.dominance_analysis = dominance_analysis

    def is_dominated(self, state: FactoredTaskState, g: int) -> bool:
        raise NotImplementedError()

    def register_state(self, state: FactoredTaskState, g: int) -> None:
        raise NotImplementedError()


class ListDominanceDatabase(DominanceDatabase):
    """
    A dominance database that stores all seen states in a list and checks for dominance by iterating over the list.
    """

    def __init__(self, task: FactoredTask, dominance_analysis: DominanceAnalysis):
        super().__init__(task, dominance_analysis)
        self.seen_states: list[tuple[FactoredTaskState, int]] = []

    def is_dominated(self, state: FactoredTaskState, g: int) -> bool:
        for previous_state, previous_g in self.seen_states:
            if g >= previous_g:
                if self.dominance_analysis.dominates(state, previous_state):
                    return True
        return False

    def register_state(self, state: FactoredTaskState, g: int) -> None:
        self.seen_states.append((state, g))


@cli_register("pdominance")
class DominancePruning(Pruning):
    """
    A pruning method that prunes nodes based on dominance relations. The dominance relations are computed using
    label-dominance.
    """

    def __init__(self, task: FactoredTask, database: type[DominanceDatabase] = ListDominanceDatabase, noop_transformation: bool = True):
        if noop_transformation:
            task = NoopTaskTransformation().transform(copy.deepcopy(task))

        super().__init__(task)
        self.task: FactoredTask = task
        self.dominance_analysis = DominanceAnalysis(task)
        self.database = database(task, self.dominance_analysis)

    def prune(self, node: SearchNode) -> bool:
        res = self.database.is_dominated(node.state, node.g)
        if not res:
            self.database.register_state(node.state, node.g)
        return res
