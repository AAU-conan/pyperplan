from pyperplan.pruning.dominance_pruning import DominancePruning
from pyperplan.task import FactoredTask, LabelledTransitionSystem, FactorState


def test_one_factor_dominance():
    task = FactoredTask("simple",
        LabelledTransitionSystem("factor_0", ['a', 'b', 'c'], [('a', 'ac', 'c'), ('b', 'bc', 'c')], 'a', ['c']),
    )
    a = FactorState('a', 0)
    b = FactorState('b', 1)
    c = FactorState('c', 2)

    dominance_pruning = DominancePruning(task)

    assert (c, a) not in dominance_pruning.dominance_relations[0]
    assert (c, b) not in dominance_pruning.dominance_relations[0]
    assert (a, b) in dominance_pruning.dominance_relations[0]
    assert (b, a) in dominance_pruning.dominance_relations[0]
    assert (a, c) in dominance_pruning.dominance_relations[0]
    assert (b, c) in dominance_pruning.dominance_relations[0]

def test_two_factor_dominance():
    task = FactoredTask("simple",
                        LabelledTransitionSystem("factor_0", ['a', 'b', 'c'], [('a', 'l1', 'c'), ('b', 'l2', 'c')], 'a', ['c']),
                        LabelledTransitionSystem("factor_1", ['a', 'b', 'c'], [('a', 'l1', 'c'), ('a', 'l2', 'c'), ('b', 'l2', 'c')], 'a', ['c']),
                        )
    a = FactorState('a', 0)
    b = FactorState('b', 1)
    c = FactorState('c', 2)

    dominance_pruning = DominancePruning(task)

    assert (c, a) not in dominance_pruning.dominance_relations[0]
    assert (c, b) not in dominance_pruning.dominance_relations[0]
    assert (a, b) in dominance_pruning.dominance_relations[0]
    assert (b, a) not in dominance_pruning.dominance_relations[0]
    assert (a, c) in dominance_pruning.dominance_relations[0]
    assert (b, c) in dominance_pruning.dominance_relations[0]
