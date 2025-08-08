import logging

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
        self.dominance_relations: tuple[set[FactorState, FactorState]]
        self._compute_dominance_relations()
        self.task: FactoredTask = task

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


    def _compute_dominance_relations(self):
        """
        Computes the dominance relations for the task.
        Returns a set of tuples (state1, state2) where state1 dominates state2.
        """
        self.dominance_relations = [{(s, t) for s in factor.states for t in factor.states if s not in factor.goal_states or t in factor.goal_states} for factor in self.task.factors]
        self.label_relation = {(l1, l2): DOMINATES_IN_ALL for l1 in self.task.labels for l2 in self.task.labels}
        assert "noop" not in self.task.labels, "The label 'noop' should not be in the task labels."
        self.label_relation.update({(l, "noop"): DOMINATES_IN_ALL for l in self.task.labels})

        changes = True
        iterations = 0
        logging.info("Computing dominance relations...")
        while changes:
            changes = False
            iterations += 1

            logging.debug(f"Iteration {iterations}, sum state-pairs {sum(len(f) for f in self.dominance_relations)}, sum label-pairs {len(self.label_relation)}")
            for i in range(self.task.size()):
                changes |= self._update_factor(i)
            changes |= self._update_label_relation()

        logging.debug(self.dominance_relations)
        logging.debug(self.label_relation)


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

    def _update_factor(self, i: int) -> bool:
        factor: LabelledTransitionSystem = self.task.factors[i]
        changes = False
        for s, t in self.dominance_relations[i].copy():
            # t dominates s if
            # ∀ s -l-> s'. (∃ t -l'-> t'. l' dominates l in all other factors than i AND t' dominates s') OR (noop dominates l in all other factors than i AND t dominates s')
            is_dominated = True
            for l, s_prime in factor.transitions_of_state(s):
                transition_dominated = False
                # NOOP case
                if (s_prime, t) in self.dominance_relations[i] and self._dominates_in_all_other_factors(i, l, "noop"):
                    transition_dominated = True
                    continue

                # Normal case
                for l_prime, t_prime in factor.transitions_of_state(t):
                    if (s_prime, t_prime) in self.dominance_relations[i] and self._dominates_in_all_other_factors(i, l, l_prime):
                        transition_dominated = True
                        break

                if not transition_dominated:
                    is_dominated = False
                    break

            if not is_dominated:
                self.dominance_relations[i].remove((s, t))
                changes = True

        return changes
            

    def _update_label_relation(self) -> bool:
        changes = False
        for (l1, l2), f in self.label_relation.copy().items():
            for i in range(self.task.size()):
                factor = self.task.factors[i]
                # l2 dominates l1 if
                # ∀ s -l1-> s'. ∃ s -l2-> s''. s'' dominates s'
                is_dominated = True
                for s, s_prime in factor.transitions_of_label(l1):
                    transition_dominated = False
                    if l2 == 'noop':
                        if (s_prime, s) in self.dominance_relations[i]:
                            transition_dominated = True
                            continue
                    else:
                        for s_double_prime in factor.transitions_of_label_state(s, l2):
                            if (s_prime, s_double_prime) in self.dominance_relations[i]:
                                transition_dominated = True
                                break

                    if not transition_dominated:
                        is_dominated = False
                        break

                if not is_dominated:
                    changes |= self._set_label_not_dominates_in(i, l1, l2)

        return changes

