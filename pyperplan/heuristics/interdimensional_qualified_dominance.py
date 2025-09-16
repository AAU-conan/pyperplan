import copy
import logging
from functools import cmp_to_key
from pathlib import Path
from typing import Type

from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.heuristics.qualified_dominance_heuristic import complement_lts, ComparisonStrategy, \
    AllComparisonStrategy, ParentComparisonStrategy
from pyperplan.pruning.dominance_pruning import DOMINATES_IN_ALL, DOMINATES_IN_NONE
from pyperplan.pruning.two_non_dominated_label_dominance_pruning import TwoNonDominatedLabelDominancePruning
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactoredTaskState, LabelledTransitionSystem, FactorState


class InterdimensionalQualifiedDominance:
    """
    A heuristic that uses interdimensional qualified dominance to estimate the cost of reaching the goal in contrast to a contrastive set
    of states. In contrast to normal qualified dominance, interdimensional qualified dominance allows to jump from one
    factor to another, it the target state dominates but the label does not dominate in the other factor.
    """
    _AUTOMATA_TR = tuple[tuple[int, FactorState, FactorState], str, tuple[int, FactorState, FactorState]]

    def __init__(self, task: FactoredTask, base_heuristic: Type[Heuristic], approximate_to_deterministic: bool = False, comparison_strategy: type[ComparisonStrategy] = AllComparisonStrategy):
        super().__init__()
        self.dominance_pruning = TwoNonDominatedLabelDominancePruning(task)
        self.heuristic: Type[Heuristic] = base_heuristic
        self.task: FactoredTask = task
        self.qdom_automaton: LabelledTransitionSystem
        self.qdom_factors_state_maps: dict[tuple[int, FactorState,FactorState], FactorState] # Maps from the original factor states to the states in the qualified dominance automaton
        self.approximate_to_deterministic = approximate_to_deterministic


        self._compute_qualified_dominance()

        self.comparison_strategy = comparison_strategy(self)
        self.extended_task = copy.deepcopy(task)

    def _compute_qualified_dominance(self):
        """
        Computes the interdimensional qualified dominance automaton for the task.
        Each state (s_i,t_i) for factor i is accepting iff s_i is not a goal or t_i is a goal.
        For each state (s_i,t_i):
            Add a transition to ⊤ (universal true) if with label l if
                ∄ s_i -l->
            Otherwise, add a transition to the state (s_i',t_i') with label l if
                ∃ s_i -l-> s_i' and ∃ t_i -l'-> t_i' s.t. l' dominates l in all other factors than i
            Otherwise, add a transition to (s_j', s_j'') with label l if
                ∃ s_i -l-> s_i' and ∃ t_i -l'-> t_i' s.t. t_i' dominates s_i' and
                ∃! s_j -l-> s_j' and ∃ s_j -l'-> s_j'' s.t. l' dominates l in all other factors than i and j
            Otherwise, add no transition of label l
        """
        # Path("fts.dot").open("w").write(self.task.to_dot())
        logging.debug("Computing Interdimensional Qualified Dominance Automata...")
        universal_true = "TRUE"
        def state_name(i: int, s: FactorState, t: FactorState):
            if (s,t) in self.dominance_pruning.dominance_relations[i]:
                return universal_true
            else:
                return f"{s}_{i} < {t}_{i}"

        states = [state_name(i, s, t) for i, factor in enumerate(self.task.factors) for s in factor.states for t in factor.states if (s,t) not in self.dominance_pruning.dominance_relations[i]]
        states.append(universal_true)

        transitions = {(universal_true, l, universal_true) for l in self.task.labels}
        goal_states = [universal_true]

        for i, factor in enumerate(self.task.factors):
            logging.debug(f'Factor {i}')
            for s in factor.states:
                for t in factor.states:
                    if (s,t) in self.dominance_pruning.dominance_relations[i]:
                        # This state is universally accepting
                        continue

                    if s not in factor.goal_states or t in factor.goal_states:
                        goal_states.append(state_name(i, s, t))

                    labels_not_applicable_at_s = self.task.labels.copy()
                    for l, s_prime in factor.transitions_of_state(s):
                        labels_not_applicable_at_s.remove(l)
                        candidate_transitions: list[tuple[tuple[int, FactorState, FactorState], str, tuple[int, FactorState, FactorState]]] = []
                        
                        # NOOP
                        if self.dominance_pruning._dominates_in_all_other_factors(i, l, "noop"):
                            candidate_transitions.append(((i, s, t), l, (i, s_prime, t)))

                        # Actual transition
                        for l_prime, t_prime in factor.transitions_of_state(t):
                            if self.dominance_pruning._dominates_in_all_other_factors(i, l, l_prime):
                                candidate_transitions.append(((i, s, t), l, (i, s_prime, t_prime)))
                            elif (s_prime, t_prime) in self.dominance_pruning.dominance_relations[i]:
                                # We have s_i -l-> s_i' and t_i -l'-> t_i' with t_i' dominates s_i'
                                # Now, we look for a different factor j where we can jump to
                                if (l, l_prime) not in self.dominance_pruning.label_relation:
                                    continue
                                f = self.dominance_pruning.label_relation[(l, l_prime)]
                                assert f != DOMINATES_IN_NONE and f != DOMINATES_IN_ALL and not (f[0] == i and f[1] == -1) # These cases do not make sense
                                if f[1] == -1:
                                    j = f[0]
                                elif f[0] == i or f[1] == i:
                                    j = f[1] if f[0] == i else f[0]
                                else:
                                    # There are two possible other factors, we cannot jump
                                    continue

                                l_trs = self.task.factors[j].transitions_of_label(l)
                                if len(l_trs) == 1:
                                    s_j, s_j_prime = l_trs[0]
                                    l_prime_trs = self.task.factors[j].transitions_of_label_state(s_j, l_prime)
                                    for s_j_prime_prime in l_prime_trs:
                                        candidate_transitions.append(((i, s, t), l, (j, s_j_prime, s_j_prime_prime)))

                        for (i,s,t), l, (j, s_prime, t_prime) in self._select_transitions(candidate_transitions):
                            transitions.add((state_name(i, s,t), l, state_name(j, s_prime, t_prime)))

                    for l in labels_not_applicable_at_s:
                        # Map this to a universally true state
                        transitions.add((state_name(i, s,t), l, universal_true))

        logging.debug(f'Creating LTS with {len(states)} states and {len(transitions)} transitions')
        lts = LabelledTransitionSystem(
            'interdimensional_qdom',
            states,
            list(transitions),
            initial_state=states[0],  # Fake initial state, not used
            goal_states=goal_states
        )
        # Path(f"iqdom.dot").open("w").write(lts.to_dot())
        self.qdom_factors_state_maps = {p: lts.states[i] for i, p in enumerate((i, s, t) for i, factor in enumerate(self.task.factors) for s in factor.states for t in factor.states if (s,t) not in self.dominance_pruning.dominance_relations[i])}
        logging.debug(f'Computing complement...')
        lts_comp = complement_lts(lts)

        # Path(f"niqdom.dot").open("w").write(lts_comp.to_dot())
        self.qdom_automaton = lts_comp
        logging.debug(f"Automaton has {len(lts_comp.states)} states and {len(lts_comp.transitions)} transitions.")

    def _select_transitions(self, candidate_transitions: list[_AUTOMATA_TR]) -> list[_AUTOMATA_TR]:
        if len(candidate_transitions) <= 1:
            return candidate_transitions
        elif any((s_prime, t_prime) in self.dominance_pruning.dominance_relations[j] for _, _, (j, s_prime, t_prime) in candidate_transitions):
            (i,s,t), l, _ = candidate_transitions[0]
            return [((i,s,t), l, (i,s,s))] # Add transition to universal true, e.g. identity here
        elif self.approximate_to_deterministic:
            # Only select one of the candidate transitions
            def cmp_transition(tr1: InterdimensionalQualifiedDominance._AUTOMATA_TR, tr2: InterdimensionalQualifiedDominance._AUTOMATA_TR):
                j1, s1, t1 = tr1[2]
                j2, s2, t2 = tr2[2]
                if j1 == j2:
                    assert s1 == s2
                    t2_dom_t1 = (t1, t2) in self.dominance_pruning.dominance_relations[j1]
                    t1_dom_t2 = (t2, t1) in self.dominance_pruning.dominance_relations[j1]
                    if t2_dom_t1 and not t1_dom_t2:
                        return -1
                    elif t1_dom_t2 and not t2_dom_t1:
                        return 1
                    else:
                        if t1 == tr1[0][1]:
                            return 1
                        elif t2 == tr2[0][1]:
                            return -1
                        else:
                            return 0
                else:
                    i = tr1[0][0]
                    if j1 == i and j2 != i:
                        return 1
                    elif j2 == i and j1 != i:
                        return -1
                    else:
                        return 0

            return [max(candidate_transitions, key=cmp_to_key(cmp_transition))]
        else:
            return candidate_transitions

    def __call__(self, node: 'SearchNode'):
        """
        Calls the base heuristic and returns the estimated cost.
        """
        self.extended_task.factors = self.task.factors.copy()
        state: FactoredTaskState = copy.deepcopy(node.state)
        # logging.debug(f"Evaluating state {state} (g={node.g})")

        for prev_state, ndf in self.comparison_strategy.get_compare_states(node):
            if ndf == -1:
                return float('inf')
            # This state is dominated in all but one factor, add the qualified dominance automaton for that factor
            self.extended_task.factors.append(self.qdom_automaton)
            state.states.append(self.qdom_factors_state_maps[(ndf, state.states[ndf], prev_state.states[ndf])])
            self.qdom_automaton.initial_state = self.qdom_factors_state_maps[ (ndf, state.states[ndf], prev_state.states[ndf])]
            # reduced = copy.copy(self.qdom_automaton)
            # reduced.remove_deadends_and_unreachable()
            # Path('st_iqdom.dot').open('w').write(reduced.to_dot())
            # pass

        # h_ref = self.heuristic(self.task)
        h = self.heuristic(self.extended_task)

        # print(extended_task.save_dot(Path("extended_task.dot")))

        # value_ref = h_ref(node)
        value = h(SearchNode(state, node.parent, node.action, node.g))
        # logging.debug(f"h/h_ref={(value + 0.00001) / (value_ref + 0.00001)} (g={node.g})")
        # print(f"h={value}")
        return value