from pyperplan.pruning.dominance_analysis import DominanceAnalysis
from pyperplan.pruning.dominance_pruning import DominancePruning
from pyperplan.task import FactoredTask, FactorState, LabelledTransitionSystem


def test_one_factor_dominance():
    task = FactoredTask(
        "simple",
        LabelledTransitionSystem("factor_0", ["a", "b", "c"], [("a", "0", "c"), ("b", "1", "c"), ("c", "2", "c")], "a", ["c"]),
    )
    a = FactorState("a", 0)
    b = FactorState("b", 1)
    c = FactorState("c", 2)

    dominance_analysis = DominanceAnalysis(task)

    assert not dominance_analysis.dominance_relations[0].dominates(c, a)
    assert not dominance_analysis.dominance_relations[0].dominates(c, b)
    assert dominance_analysis.dominance_relations[0].dominates(a, b)
    assert dominance_analysis.dominance_relations[0].dominates(b, a)
    assert dominance_analysis.dominance_relations[0].dominates(a, c)
    assert dominance_analysis.dominance_relations[0].dominates(b, c)


def test_two_factor_dominance():
    task = FactoredTask(
        "simple",
        LabelledTransitionSystem("factor_0", ["a", "b", "c"], [("a", "l1", "c"), ("b", "l2", "c"), ("c", "l1", "c"), ("c", "l2", "c")], "a", ["c"]),
        LabelledTransitionSystem("factor_1", ["a", "b", "c"], [("a", "l1", "c"), ("a", "l2", "c"), ("b", "l2", "c")], "a", ["c"]),
    )
    a = FactorState("a", 0)
    b = FactorState("b", 1)
    c = FactorState("c", 2)

    dominance_analysis = DominanceAnalysis(task)

    assert not dominance_analysis.dominance_relations[0].dominates(c, a)
    assert not dominance_analysis.dominance_relations[0].dominates(c, b)
    assert dominance_analysis.dominance_relations[0].dominates(a, b)
    assert not dominance_analysis.dominance_relations[0].dominates(b, a)
    assert dominance_analysis.dominance_relations[0].dominates(a, c)
    assert dominance_analysis.dominance_relations[0].dominates(b, c)

    assert not dominance_analysis.dominance_relations[1].dominates(c, a)
    assert not dominance_analysis.dominance_relations[1].dominates(c, b)
    assert dominance_analysis.dominance_relations[1].dominates(b, a)
    assert not dominance_analysis.dominance_relations[1].dominates(b, c)
    assert not dominance_analysis.dominance_relations[1].dominates(a, b)
    assert not dominance_analysis.dominance_relations[1].dominates(a, c)

    assert not dominance_analysis.label_relation.dominates_in_factor(0, "l1", "l2")
    assert not dominance_analysis.label_relation.dominates_in_factor(0, "l2", "l1")
    assert dominance_analysis.label_relation.dominates_in_factor(1, "l1", "l2")
    assert not dominance_analysis.label_relation.dominates_in_factor(1, "l2", "l1")
