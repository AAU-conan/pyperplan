import logging
import time
from typing import Callable

from pyperplan.task import (
    FactoredTask,
    FactoredTaskState,
    FactorState,
    LabelledTransitionSystem,
)


Label = str


class StateDominanceRelation:
    """
    A class to represent a dominance relation between states of a factor.
    """

    def __init__(self):
        self.relation: set[tuple[FactorState, FactorState]] = set()

    def initialize(self, factor: LabelledTransitionSystem, trivial_condition: Callable[[FactorState, FactorState], bool]):
        """
        Initializes the dominance relation with all pairs (s, t) where trivial_condition(s, t) is True.
        """
        self.relation = {(s, t) for s in factor.states for t in factor.states if trivial_condition(s, t)}

    def add(self, s: FactorState, t: FactorState):
        self.relation.add((s, t))

    def remove(self, s: FactorState, t: FactorState):
        self.relation.remove((s, t))

    def dominates(self, s: FactorState, t: FactorState) -> bool:
        """
        Checks if state t dominates state s.
        """
        return (s, t) in self.relation

    def __iter__(self):
        return iter(self.relation)


class LabelDominanceRelation:
    """
    A class to represent a dominance relation between labels.
    """

    class AllNoneFactor:
        _ALL = -1
        _NONE = -2

        def __init__(self, value: int):
            self._not_value = value

        def remove(self, factor: int) -> bool:
            if self._not_value == self._NONE:
                return False
            elif self._not_value == self._ALL:
                self._not_value = factor
                return True
            elif self._not_value != factor:
                self._not_value = self._NONE
                return True
            return False

        def is_none(self) -> bool:
            return self._not_value == self._NONE

        def is_all(self) -> bool:
            return self._not_value == self._ALL

        def contains(self, factor: int):
            return self.is_all() or (self._not_value != factor and not self.is_none())

        def contains_all_but(self, factor: int):
            return self.is_all() or self._not_value == factor

        def intersect(self, other: "LabelDominanceRelation.AllNoneFactor") -> "LabelDominanceRelation.AllNoneFactor":
            if self.is_none() or other.is_none():
                return LabelDominanceRelation.AllNoneFactor.none()
            elif self.is_all():
                return other
            elif other.is_all():
                return self
            elif self._not_value == other._not_value:
                return LabelDominanceRelation.AllNoneFactor(self._not_value)
            else:
                return LabelDominanceRelation.AllNoneFactor.none()

        @staticmethod
        def all() -> "LabelDominanceRelation.AllNoneFactor":
            return LabelDominanceRelation.AllNoneFactor(LabelDominanceRelation.AllNoneFactor._ALL)

        @staticmethod
        def none() -> "LabelDominanceRelation.AllNoneFactor":
            return LabelDominanceRelation.AllNoneFactor(LabelDominanceRelation.AllNoneFactor._NONE)

        def __eq__(self, other):
            return self._not_value == other._not_value

        def __lt__(self, other: "LabelDominanceRelation.AllNoneFactor") -> bool:
            return (not self.is_all() and other.is_all()) or (self.is_none() and not other.is_none())

        def __repr__(self):
            if self.is_all():
                return "ALL"
            elif self.is_none():
                return "NONE"
            else:
                return f"ALL BUT {self._not_value}"

        def __str__(self):
            return self.__repr__()

        def __hash__(self):
            return hash(self._not_value)

    def __init__(self):
        self._relation: dict[tuple[Label, Label], "LabelDominanceRelation.AllNoneFactor"] = {}

    def initialize(self, labels: set[Label], trivial_condition: Callable[[Label, Label], int]):
        """
        Initializes the label dominance relation with all pairs (l1, l2) where trivial_condition(l1, l2) is True.
        """
        self._relation = {(l1, l2): LabelDominanceRelation.AllNoneFactor.all() for l1 in labels for l2 in labels if trivial_condition(l1, l2)}

    def set_not_dominates_in(self, i: int, l1: Label, l2: Label) -> bool:
        """
        Sets the label relation for l1 and l2 in factor i to not dominate.
        """
        if (l1, l2) in self._relation:
            f = self._relation[(l1, l2)]
            if f.remove(i):
                if f.is_none():
                    self._relation.pop((l1, l2))
                return True
        return False

    def dominates_in_all_other_factors(self, i: int, l1: Label, l2: Label) -> bool:
        """
        Returns if label l2 dominates label l1 in all factors except factor i.
        """
        f = self._relation.get((l1, l2), LabelDominanceRelation.AllNoneFactor.none())
        return f.contains_all_but(i)

    def dominates_in_factor(self, i: int, l1: Label, l2: Label) -> bool:
        """
        Lookup if label l2 dominates label l1 in factor i.
        """
        r = self._relation.get((l1, l2), LabelDominanceRelation.AllNoneFactor.none())
        return r.contains(i)

    def __getitem__(self, item: tuple[Label, Label]) -> "LabelDominanceRelation.AllNoneFactor":
        return self._relation.get(item, LabelDominanceRelation.AllNoneFactor.none())

    def size(self) -> int:
        return len(self._relation)

    def __iter__(self):
        return iter(self._relation.items())


class DominanceAnalysis:
    """
    A class to perform label-dominance analysis on a FactoredTask.
    """

    def __init__(self, task: FactoredTask):
        self.task = task
        self.dominance_relations: list[StateDominanceRelation] = [StateDominanceRelation() for _ in range(self.task.size())]
        self.label_relation: LabelDominanceRelation = LabelDominanceRelation()
        compute_start_time = time.process_time()
        self._compute_dominance_relations()
        logging.info(f"Dominance pre-computation time: {time.process_time() - compute_start_time}s")

    @staticmethod
    def _goal_condition_met(factor: LabelledTransitionSystem, s: FactorState, t: FactorState):
        """
        The trivial condition for dominance: s is not a goal state or t is a goal state.
        t cannot dominate s if s is a goal state and t is not.
        """
        return s not in factor.goal_states or t in factor.goal_states

    def _initialize_dominance_relations(self):
        for i in range(self.task.size()):
            self._initialize_dominance_relation(i)

        # Initialize label relation. Trivial condition is that l2 cannot be more expensive than l1.
        self.label_relation.initialize(self.task.labels, lambda l1, l2: self.task.label_costs[l2] <= self.task.label_costs[l1])

    def _initialize_dominance_relation(self, i: int):
        factor = self.task.factors[i]
        self.dominance_relations[i].initialize(factor, lambda s, t: self._goal_condition_met(factor, s, t))

    def _compute_dominance_relations(self):
        """
        Computes the dominance relations for the task.
        Initialize the dominance relations and then refine them until a fixpoint is reached.
        """
        self._initialize_dominance_relations()
        logging.info("Computing dominance relations...")
        self._refine_dominance_relations()
        # logging.debug(self.dominance_relations)
        # logging.debug(self.label_relation)

    def _refine_dominance_relations(self):
        """
        Refines the dominance relations until a fixpoint is reached.
        """
        changes = True
        iterations = 0
        while changes:
            changes = False
            iterations += 1

            logging.debug(
                f"Iteration {iterations}, sum non-identity state-pairs {sum(sum(1 if s != t else 0 for s, t in f) for f in self.dominance_relations)}, sum label-pairs {self.label_relation.size() - len(self.task.labels)}"
            )
            for i in range(self.task.size()):
                changes |= self._update_factor(i)
            changes |= self._update_label_relation()

    def _transition_dominated_by_transition(self, i: int, l: Label, s_prime: FactorState, l_prime: Label, t_prime: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by the transition t -l'-> t' in factor i.
        """
        return (s_prime, t_prime) in self.dominance_relations[i] and self.label_relation.dominates_in_all_other_factors(i, l, l_prime)

    def _transition_dominated_by_any_transition(self, i: int, l: Label, s_prime: FactorState, t: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by any transition of t in factor i.
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for l_prime, t_prime in factor.transitions_of_state(t):
            if self._transition_dominated_by_transition(i, l, s_prime, l_prime, t_prime):
                return True
        return False

    def _state_dominates_state(self, i: int, s: FactorState, t: FactorState) -> bool:
        """
        Checks if state t dominates state s in factor i.
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for l, s_prime in factor.transitions_of_state(s):
            if not self._transition_dominated_by_any_transition(i, l, s_prime, t):
                return False

        return True

    def _update_factor(self, i: int) -> bool:
        changes = False
        for s, t in list(self.dominance_relations[i]):
            # t dominates s if
            # ∀ s -l-> s'. (∃ t -l'-> t'. l' dominates l in all other factors than i AND t' dominates s')
            if not self._state_dominates_state(i, s, t):
                self.dominance_relations[i].remove(s, t)
                changes = True

        return changes

    def _transition_with_label_that_dominates(self, i: int, s: FactorState, l: Label, t: FactorState):
        """
        Checks if there is a transitions t -l->t' such that t' dominates s', in factor i.
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for t_prime in factor.transitions_of_label_state(t, l):
            if (s, t_prime) in self.dominance_relations[i]:
                return True
        return False

    def _label_dominates_label_in_factor(self, i: int, l1: Label, l2: Label) -> bool:
        """
        Checks if label l2 dominates label l1 in factor i.
        l2 dominates l1 if
        ∀ s -l1-> s'. ∃ s -l2-> s''. s'' dominates s'
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for s, s_prime in factor.transitions_of_label(l1):
            if not self._transition_with_label_that_dominates(i, s_prime, l2, s):
                return False
        return True

    def _update_label_relation(self) -> bool:
        changes = False
        for (l1, l2), f in list(self.label_relation):
            for i in range(self.task.size()):
                if not f.contains(i):
                    continue

                if not self._label_dominates_label_in_factor(i, l1, l2):
                    changes |= self.label_relation.set_not_dominates_in(i, l1, l2)
                    if self.label_relation[(l1, l2)].is_none():
                        break

        return changes

    def dominates(self, s: FactoredTaskState, t: FactoredTaskState) -> bool:
        """
        Checks if state t dominates state s.
        """
        for i in range(self.task.size()):
            if not self.dominance_relations[i].dominates(s.states[i], t.states[i]):
                return False
        return True

    def validate(self):
        """
        Validates that the dominance relations are correct. Checks that the relations are reflexive and transitive.
        """
        for i in range(self.task.size()):
            factor: LabelledTransitionSystem = self.task.factors[i]
            # Validate dominance relation
            for s in factor.states:
                if (s, s) not in self.dominance_relations[i]:
                    raise ValueError(f"Dominance relation for factor {i} is not reflexive: {s} does not dominate itself.")
            for s, t in self.dominance_relations[i]:
                for u, v in self.dominance_relations[i]:
                    if t == u and (s, v) not in self.dominance_relations[i]:
                        raise ValueError(f"Dominance relation for factor {i} is not transitive: {s} dominates {t} and {u} dominates {v}, but {s} does not dominate {v}.")

            # Validate label relation
            for l in self.task.labels:
                f = self.label_relation[(l, l)]
                if not f.is_all():
                    raise ValueError(f"Label relation is not reflexive: {l} does not dominate itself, but only in {f}.")
            for (l1, l2), f1 in self.label_relation:
                if l1 == l2:
                    continue
                for (l3, l4), f2 in self.label_relation:
                    if l3 == l4:
                        continue
                    if l2 == l3:
                        f3 = self.label_relation[(l1, l4)]
                        if f3 >= f1.intersect(f2):
                            raise ValueError(
                                f"Label relation is not transitive: {l1} dominates {l2} in {f1} and {l2} dominates {l4} in {f2}, but {l1} only dominates {l4} in {f3}."
                            )
