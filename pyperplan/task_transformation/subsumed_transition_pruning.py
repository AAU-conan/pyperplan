import copy
import logging

from pyperplan.cli import cli_register
from pyperplan.pruning.dominance_analysis import DominanceAnalysis
from pyperplan.pruning.dominance_pruning import DominancePruning
from pyperplan.task import FactoredTask
from pyperplan.task_transformation.noop_task_transformation import (
    NoopTaskTransformation,
)
from pyperplan.task_transformation.task_transformation import TaskTransformation


@cli_register("ttstp")
class SubsumedTransitionPruning(TaskTransformation):
    """
    This class implements the Subsumed Transition Pruning Task Transformation for planning tasks.
    A transition s-l->s' is subsumed if there exists another transition s-l'->s'' such that l' dominates l in all other
    factors and s'' dominates s'. After removing a subsumed transition, we need to update the dominance relations, since
    the label dominance might change.

    If preserve_label_dominance is set to True, we only remove a subsumed transition if the label dominance is preserved,
    i.e. we can only remove s-l->s' if for all s -l''->s''' then l does not dominate l''.
    """

    def __init__(self, preserve_label_dominance: bool = True) -> None:
        super().__init__()
        self.preserve_label_dominance = preserve_label_dominance

    def transform(self, task: FactoredTask):
        """ """
        dominance_analysis = DominanceAnalysis(NoopTaskTransformation().transform(copy.deepcopy(task)))
        removed = True
        while removed:
            removed = False
            dom_rels = dominance_analysis.dominance_relations
            label_rel = dominance_analysis.label_relation
            for i, factor in enumerate(task.factors):
                for src, label, tgt in factor.transitions:
                    if self.preserve_label_dominance:
                        # Check if label dominates any other label in the same factor
                        if any(label_rel.dominates_in_factor(i, other_label, label) and other_label != label for other_label, _ in factor.transitions_of_state(src)):
                            continue

                    for label2, tgt2 in factor.transitions_of_state(src):
                        if not (tgt == tgt2 and label == label2):
                            if label_rel.dominates_in_all_other_factors(i, label, label2) and dom_rels[i].dominates(tgt, tgt2):
                                logging.debug(f"Removing subsumed transition {src} -{label}-> {tgt} because {src} -{label2}-> {tgt2} dominates it")
                                factor.transitions.remove((src, label, tgt))
                                factor._compute_cached_values()
                                removed = True
                                break

                if removed:
                    if not self.preserve_label_dominance:
                        dominance_analysis._refine_dominance_relations()
                    break

        for factor in task.factors:
            factor.remove_deadends_and_unreachable()

        return task
