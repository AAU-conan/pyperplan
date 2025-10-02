import copy

from pyperplan.cli import cli_register
from pyperplan.task import LabelledTransitionSystem
from pyperplan.task_transformation.task_transformation import TaskTransformation


@cli_register()
class CheckAreBisimilar(TaskTransformation):

    def __init__(self, tt1: type[TaskTransformation], tt2: type[TaskTransformation]) -> None:
        super().__init__()
        self.tt1 = tt1()
        self.tt2 = tt2()

    def transform(self, task):
        task1 = self.tt1.transform(copy.deepcopy(task))
        task2 = self.tt2.transform(copy.deepcopy(task))

        if not all(LabelledTransitionSystem.are_bisimilar(f1, f2) for f1, f2 in zip(task1.factors, task2.factors)):
            raise ValueError("The two transformed tasks are not bisimilar")
        return task1
