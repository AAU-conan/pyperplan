import copy

from pyperplan.cli import cli_register
from pyperplan.task import FactoredTask, LabelledTransitionSystem
from pyperplan.task_transformation.task_transformation import TaskTransformation


@cli_register("ttmerge")
class MergeTaskTransformation(TaskTransformation):
    """
    Task transformation that merges factors.
    """

    def __init__(self, factor_size: int = 15):
        self.factor_size = factor_size
        pass

    def transform(self, task: FactoredTask) -> FactoredTask:
        """
        Merge factors until no two factors can be merged into a factor of size <= self.factor_size.
        """
        task = copy.copy(task)
        while True:
            merged = False
            for i in range(len(task.factors)):
                for j in range(i + 1, len(task.factors)):
                    if len(task.factors[i].states) * len(task.factors[j].states) <= self.factor_size:
                        product_factor = LabelledTransitionSystem.merge(task.factors[i], task.factors[j])
                        if len(product_factor.states) <= self.factor_size:
                            task.factors.pop(j)
                            task.factors[i] = product_factor
                            merged = True
                            break
                if merged:
                    break
            if not merged:
                break
        return task
