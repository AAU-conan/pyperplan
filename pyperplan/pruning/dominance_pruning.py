import copy
import logging
import time

from pyperplan.pruning.pruning import Pruning
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactorState, LabelledTransitionSystem, FactoredTaskState

DOMINATES_IN_ALL = -1
DOMINATES_IN_NONE = -2

class DominancePruning(Pruning):
    """
    A pruning method that prunes nodes based on dominance relations. The dominance relations are computed using
    label-dominance.
    """
    def __init__(self, task: FactoredTask):
        super().__init__(task)
        self.dominance_relations: list[set[tuple[FactorState, FactorState]]]
        self.task: FactoredTask = task
        compute_start_time = time.process_time()
        self._compute_dominance_relations()
        logging.info(f"Dominance pre-computation time: {time.process_time() - compute_start_time}s")

        # Stores the lowest g-value seen for a state
        self.seen_states: dict[FactoredTaskState, int] = dict()

    def prune(self, node: SearchNode) -> bool:
        # Check if it is dominated
        state: FactoredTaskState = node.state
        for previous_state, previous_g in self.seen_states.items():
            if node.g >= previous_g:
                is_dominated = True
                for i in range(self.task.size()):
                    if (state.states[i], previous_state.states[i]) not in self.dominance_relations[i]:
                        is_dominated = False
                        break
                if is_dominated:
                    return True

        # If we don't prune, add state to seen states
        previous_lowest_g = self.seen_states.get(node.state, float('inf'))
        if node.g < previous_lowest_g:
            self.seen_states[node.state] = node.g
        return False

    def _initialize_dominance_relations(self):
        self.dominance_relations = [{(s, t) for s in factor.states for t in factor.states if s not in factor.goal_states or t in factor.goal_states} for factor in self.task.factors]
        self.label_relation = {(l1, l2): DOMINATES_IN_ALL for l1 in self.task.labels for l2 in self.task.labels if self.task.get_action_cost(l2) <= self.task.get_action_cost(l1)}
        self.label_relation.update({(l, "noop"): DOMINATES_IN_ALL for l in self.task.labels})

    def _compute_dominance_relations(self):
        """
        Computes the dominance relations for the task.
        Returns a set of tuples (state1, state2) where state1 dominates state2.
        """
        self._initialize_dominance_relations()
        assert "noop" not in self.task.labels, "The label 'noop' should not be in the task labels."

        changes = True
        iterations = 0
        logging.info("Computing dominance relations...")
        while changes:
            changes = False
            iterations += 1

            logging.debug(f"Iteration {iterations}, sum non-identity state-pairs {sum(sum(1 if s != t else 0 for s, t in f) for f in self.dominance_relations)}, sum label-pairs {len(self.label_relation) - len(self.task.labels)}")
            for i in range(self.task.size()):
                changes |= self._update_factor(i)
            changes |= self._update_label_relation()

        # logging.debug(self.dominance_relations)
        # logging.debug(self.label_relation)


    def _dominates_in_all_other_factors(self, i: int, l1: str, l2: str) -> bool:
        """
        Returns if label l2 dominates l1 in all factors except factor i.
        """
        f = self.label_relation.get((l1, l2), DOMINATES_IN_NONE)
        return f == DOMINATES_IN_ALL or f == i

    def _set_label_not_dominates_in(self, i: int, l1: str, l2: str) -> bool:
        """
        Sets the label relation for l1 and l2 in factor i to not dominate.
        """
        if (l1, l2) in self.label_relation:
            f = self.label_relation[(l1, l2)]
            if f == DOMINATES_IN_ALL:
                self.label_relation[(l1, l2)] = i
                return True
            elif f != i:
                self.label_relation.pop((l1, l2))
                return True
            else:
                # If it was already set to not dominate in this factor, do nothing
                return False
        return False

    def _transition_dominated_by_noop(self, i: int, l: str, s_prime: FactorState, t: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by a noop transition of t in factor i.
        """
        return (s_prime, t) in self.dominance_relations[i] and self._dominates_in_all_other_factors(i, l, "noop")

    def _transition_dominated_by_transition(self, i: int, l: str, s_prime: FactorState, l_prime: str, t_prime: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by the transition t -l'-> t' in factor i.
        """
        return (s_prime, t_prime) in self.dominance_relations[i] and self._dominates_in_all_other_factors(i, l, l_prime)

    def _transition_dominated_by_any_transition(self, i: int, l: str, s_prime: FactorState, t: FactorState) -> bool:
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
            if not (self._transition_dominated_by_noop(i, l, s_prime, t) or self._transition_dominated_by_any_transition(i, l, s_prime, t)):
                return False

        return True

    def _update_factor(self, i: int) -> bool:
        factor: LabelledTransitionSystem = self.task.factors[i]
        changes = False
        for s, t in self.dominance_relations[i].copy():
            # t dominates s if
            # ∀ s -l-> s'. (∃ t -l'-> t'. l' dominates l in all other factors than i AND t' dominates s') OR (noop dominates l in all other factors than i AND t dominates s')
            if not self._state_dominates_state(i, s, t):
                self.dominance_relations[i].remove((s, t))
                changes = True

        return changes


    def _transition_with_label_that_dominates(self, i: int, s: FactorState, l: str, t: FactorState):
        """
        Checks if there is a transitions t -l->t' such that t' dominates s', in factor i.
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for t_prime in factor.transitions_of_label_state(t, l):
            if (s, t_prime) in self.dominance_relations[i]:
                return True
        return False

    def _label_dominates_label_in_factor(self, i: int, l1: str, l2: str) -> bool:
        """
        Checks if label l2 dominates label l1 in factor i.
        l2 dominates l1 if
        ∀ s -l1-> s'. ∃ s -l2-> s''. s'' dominates s'
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for s, s_prime in factor.transitions_of_label(l1):
            if not (l2 == 'noop' and (s_prime, s) in self.dominance_relations[i]) and not self._transition_with_label_that_dominates(i, s_prime, l2, s):
                return False
        return True
    
    def label_dominates_label_in_factor(self, i: int, l1: str, l2: str):
        """
        Lookup if label l2 dominates label l1 in factor i.
        """
        r = self.label_relation.get((l1, l2), DOMINATES_IN_NONE)
        return r == DOMINATES_IN_ALL or r != i

    def _update_label_relation(self) -> bool:
        changes = False
        for (l1, l2), f in self.label_relation.copy().items():
            for i in range(self.task.size()):
                if i == f:
                    continue

                if not self._label_dominates_label_in_factor(i, l1, l2):
                    changes |= self._set_label_not_dominates_in(i, l1, l2)
                    if (l1, l2) not in self.label_relation:
                        break

        return changes


class GoalDistanceDominancePruning(DominancePruning):
    """
    Initializes dominance relations based on goal distance.
    """
    def __init__(self, task: FactoredTask):
        super().__init__(task)

    def _compute_goal_distances(self, factor: LabelledTransitionSystem) -> dict[FactorState, int]:
        goal_distances = {state: float('inf') if state not in factor.goal_states else 0 for state in factor.states}

        # Do dijkstra-like search to compute the goal distances
        queue = [(0, state) for state in factor.goal_states]
        import heapq
        heapq.heapify(queue)
        while queue:
            cost, state = heapq.heappop(queue)
            if cost > goal_distances[state]:
                continue

            for prev_state, label in factor.transitions_to_state(state):
                prev_cost = cost + 1
                if prev_cost < goal_distances[prev_state]:
                    goal_distances[prev_state] = prev_cost
                    heapq.heappush(queue, (prev_cost, prev_state))

        return goal_distances

    def _initialize_dominance_relations(self):
        self.dominance_relations = []
        for factor in self.task.factors:
            if len(factor.goal_states) == len(factor.states):
                # All states are goal states, everything dominates everything
                self.dominance_relations.append({(s, t) for s in factor.states for t in factor.states})
            else:
                goal_distances = self._compute_goal_distances(factor)
                dominance_relation = set()
                for s in factor.states:
                    for t in factor.states:
                        if goal_distances[s] >= goal_distances[t]:
                            dominance_relation.add((s, t))
                self.dominance_relations.append(dominance_relation)
        self.label_relation = {(l1, l2): DOMINATES_IN_ALL for l1 in self.task.labels for l2 in self.task.labels if self.task.get_action_cost(l2) <= self.task.get_action_cost(l1)}
        self.label_relation.update({(l, "noop"): DOMINATES_IN_ALL for l in self.task.labels})
        self._update_label_relation()


class EvidenceCacheDominancePruning(DominancePruning):
    """
    Remembers the reason for dominance for the checks.
    """
    def __init__(self, task: FactoredTask):
        self.state_evidence_cache: dict[tuple[int, str, FactorState, FactorState], tuple[str, FactorState]] = dict()
        self.label_evidence_cache: dict[tuple[int, FactorState, str, FactorState], FactorState] = dict()
        super().__init__(task)

    def _transition_dominated_by_any_transition(self, i: int, l: str, s_prime: FactorState, t: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by any transition of t in factor i.
        """
        cache_entry = (i, l, s_prime, t)
        if cache_entry in self.state_evidence_cache:
            l_prime, t_prime = self.state_evidence_cache[cache_entry]
            if self._transition_dominated_by_transition(i, l, s_prime, l_prime, t_prime):
                return True
            else:
                self.state_evidence_cache.pop(cache_entry)

        factor: LabelledTransitionSystem = self.task.factors[i]
        for l_prime, t_prime in factor.transitions_of_state(t):
            if self._transition_dominated_by_transition(i, l, s_prime, l_prime, t_prime):
                self.state_evidence_cache[cache_entry] = (l_prime, t_prime)
                return True

        return False

    # def _transition_with_label_that_dominates(self, i: int, s: FactorState, l: str, t: FactorState):
    #     """
    #     Checks if there is a transitions t -l->t' such that t' dominates s', in factor i.
    #     """
    #     cache_entry = (i, s, l, t)
    #     if cache_entry in self.label_evidence_cache:
    #         t_prime = self.label_evidence_cache[cache_entry]
    #         if (s, t_prime) in self.dominance_relations[i]:
    #             return True
    #         else:
    #             self.label_evidence_cache.pop(cache_entry)
    #
    #     factor: LabelledTransitionSystem = self.task.factors[i]
    #     for t_prime in factor.transitions_of_label_state(t, l):
    #         if (s, t_prime) in self.dominance_relations[i]:
    #             self.label_evidence_cache[cache_entry] = t_prime
    #             return True
    #     return False

    # def _label_dominates_label_in_factor(self, i: int, l1: str, l2: str) -> bool:
    #     """
    #     Checks if label l2 dominates label l1 in factor i.
    #     l2 dominates l1 if
    #     ∀ s -l1-> s'. ∃ s -l2-> s''. s'' dominates s'
    #     """
    #     cache_entry = (i, l1, l2)
    #     if cache_entry in self.label_evidence_cache:
    #         # Verify that for all (s,t) pairs in the evidence, t dominates s
    #         if all((s,t) in self.dominance_relations[i] for s,t in self.label_evidence_cache[cache_entry]):
    #             return True
    #         else:
    #             self.label_evidence_cache.pop(cache_entry)
    #
    #     factor: LabelledTransitionSystem = self.task.factors[i]
    #     evidence = []
    #     for s, s_prime in factor.transitions_of_label(l1):
    #         if l2 == 'noop':
    #             if (s_prime, s) in self.dominance_relations[i]:
    #                 evidence.append((s, s_prime))
    #             else:
    #                 return False
    #         else:
    #             r, t_prime = self._transition_with_label_that_dominates(i, s_prime, l2, s)
    #             if r:
    #                 evidence.append((s_prime, t_prime))
    #             else:
    #                 return False
    #
    #     self.label_evidence_cache[cache_entry] = evidence
    #     return True

class IrrelevantLabelsDominancePruning(DominancePruning):
    """
    A label l is irrelevant in factor i if for all states s, s -l-> s. I.e., this labels is a self-loop everywhere and
    never a non-self-loop.
    Two irrelevant labels in factor i l1 and l2, always have the property that l1 dominates l2 and l2 dominates l1 in factor i.
    An irrelevant label in factor i l1 dominates l2 in factor i if c(l1) <= c(l2) and noop dominates l2 in factor i.
    """
    def __init__(self, task: FactoredTask):
        self.irrelevant_labels: list[set[str]] = []
        self.relevant_labels: list[set[str]] = []
        self._compute_irrelevant_labels(task)
        super().__init__(task)

    def _compute_irrelevant_labels(self, task: FactoredTask):
        for factor in task.factors:
            irrelevant = set()
            for l in task.labels:
                if all(src == tgt for src, tgt in factor.transitions_of_label(l)) and len(factor.transitions_of_label(l)) == len(factor.states):
                    irrelevant.add(l)
            self.relevant_labels.append(task.labels - irrelevant)
            self.irrelevant_labels.append(irrelevant)

    def _update_label_relation(self) -> bool:
        changes = False
        lr = self.label_relation.copy()
        for (l1, l2), f in lr.items():
            for i in range(self.task.size()):
                if i == f:
                    continue # If it was already set to not dominate in this factor, do nothing

                if l2 in self.irrelevant_labels[i]:
                    if l1 in self.irrelevant_labels[i]:
                        continue # l1 and l2 are both irrelevant, so they dominate each other
                    else:
                        if self.label_dominates_label_in_factor(i, l1, 'noop'):
                            continue
                        else:
                            changes |= self._set_label_not_dominates_in(i, l1, l2)
                            if (l1, l2) not in self.label_relation:
                                break
                            continue

                if not self._label_dominates_label_in_factor(i, l1, l2):
                    changes |= self._set_label_not_dominates_in(i, l1, l2)
                    if (l1, l2) not in self.label_relation:
                        break

        return changes


class StateOrderedDominancePruning(DominancePruning):
    """
    A dominance pruning that orders the states in each factor and only allows dominance from higher to lower ordered states.
    """
    def __init__(self, task: FactoredTask):
        super().__init__(task)

    def _initialize_dominance_relations(self):
        super()._initialize_dominance_relations()
        self.dominance_order = [{(s,t): 0 for s,t in rel} for rel in self.dominance_relations]


    def _state_dominates_state(self, i: int, s: FactorState, t: FactorState) -> bool:
        """
        Checks if state t dominates state s in factor i.
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for l, s_prime in factor.transitions_of_state(s):
            self.c_state = s
            if not (self._transition_dominated_by_noop(i, l, s_prime, t) or self._transition_dominated_by_any_transition(i, l, s_prime, t)):
                return False

        return True

    def _transition_dominated_by_any_transition(self, i: int, l: str, s_prime: FactorState, t: FactorState) -> bool:
        """
        Checks if the transition s -l-> s' is dominated by any transition of t in factor i.
        Whenever t -l'-> t' dominates s -l-> s', ensure that (t', s') have lower order than (t, s).
        """
        factor: LabelledTransitionSystem = self.task.factors[i]
        for l_prime, t_prime in factor.transitions_of_state(t):
            if self._transition_dominated_by_transition(i, l, s_prime, l_prime, t_prime):
                # If (t', s') have a higher order than (t, s), increase (t,s)'s order
                if self.dominance_order[i][(s_prime, t_prime)] >= self.dominance_order[i][(self.c_state, t)]:
                    self.dominance_order[i][(self.c_state, t)] += 1
                return True
        return False

    def _update_factor(self, i: int) -> bool:
        changes = False
        st_order = sorted(self.dominance_relations[i], key=lambda x: self.dominance_order[i][x])
        for s, t in st_order:
            # t dominates s if
            # ∀ s -l-> s'. (∃ t -l'-> t'. l' dominates l in all other factors than i AND t' dominates s') OR (noop dominates l in all other factors than i AND t dominates s')
            if not self._state_dominates_state(i, s, t):
                self.dominance_relations[i].remove((s, t))
                changes = True

        return changes

