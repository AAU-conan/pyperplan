from pyperplan.cli import cli_register
from pyperplan.task import FactoredTask
from pyperplan.task_transformation.task_transformation import TaskTransformation


@cli_register()
class NoopTaskTransformation(TaskTransformation):
    """
    A task transformation that adds a new label 'noop' with a self-loop transition in all states of all factors.
    """

    def transform(self, task: FactoredTask):
        assert "noop" not in task.labels
        task.labels.add("noop")
        task._ordered_labels = sorted(task.labels)
        task.label_costs["noop"] = 0
        for factor in task.factors:
            for state in factor.states:
                factor.transitions.append((state, "noop", state))
            factor._compute_cached_values()
        return task
