import logging
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from pyperplan.heuristics.qualified_dominance_heuristic import complement_lts
from pyperplan.pruning.dominance_pruning import DominancePruning
from pyperplan.task import FactoredTask, FactorState, LabelledTransitionSystem


class QualifiedDominanceTaskTransformation:
    """
    This class implements the Qualified Dominance Task Transformation for planning tasks.
    """
    def __init__(self, approximate_to_deterministic: bool = False):
        """
        :param approximate_to_deterministic: If true, the transformation will try to make the resulting factors
            deterministic by only keeping one of multiple possible transitions if possible.
        """
        self.approximate_to_deterministic = approximate_to_deterministic

    def compute(self, task: FactoredTask) -> FactoredTask:
        """
        First compute the dominance relations for the task.

        For each factor:
            0. Add a state ⊥ to the factor, no transitions from or to it
            1. Add (s_i, ⊥) to the worklist where s_i is the initial state
            2. Pop (s, t) from the worklist
            3. For each label l:
                a. add a transition to ⊤ (universal true) if ∄ s -l->
                d. Otherwise, add a transition s-l->(s',s) to the new factor if t = ⊥ and noop dominates l in all other factors
                c. Otherwise, add a transition to the state (s',t') with label l if
                    ∃ s -l-> s' and ∃ t -l'-> t' s.t. l' dominates l in all other factors
                d. Otherwise, add a transition to (s', ⊥) with label l
        """
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            Path("pre_transformation_task.dot").write_text(task.to_dot())
        dominance_pruning = DominancePruning(task)
        dominance_relations = dominance_pruning.dominance_relations
        new_factors: list[LabelledTransitionSystem] = []
        logging.debug('Starting Qualified Dominance Task Transformation')
        for i, factor in enumerate(task.factors):
            logging.debug(f'Factor {i}')
            factor.states.append(FactorState("⊥", len(factor.states)))

            universal_true = "TRUE"
            def state_name(s: FactorState, t: FactorState):
                if (s,t) in dominance_relations[i]:
                    return universal_true
                return f"{s.name} < {t.name}"
            initial_state_bot = (factor.initial_state, FactorState("⊥", -1))
            states = {state_name(*initial_state_bot)}
            states.add(universal_true)

            transitions = set((universal_true, l, universal_true) for l in task.labels)
            goal_states = [universal_true]

            worklist = [initial_state_bot]

            while len(worklist) > 0:
                s, t = worklist.pop()
                if (s,t) in dominance_relations[i]:
                    continue

                if s not in factor.goal_states or t in factor.goal_states:
                    goal_states.append(state_name(s, t))

                labels_not_applicable_at_s = task.labels.copy()
                for l, s_prime in factor.transitions_of_state(s):
                    labels_not_applicable_at_s.remove(l)
                    any_transition = False
                    candidate_transitions: list[tuple[tuple[FactorState, FactorState], str, tuple[FactorState, FactorState]]] = []

                    # NOOP
                    if t.name != '⊥':
                        if dominance_pruning._dominates_in_all_other_factors(i, l, "noop"):
                            candidate_transitions.append(((s, t), l, (s_prime, t)))
                            any_transition = True

                        # Actual transition
                        for l_prime, t_prime in factor.transitions_of_state(t):
                            if dominance_pruning._dominates_in_all_other_factors(i, l, l_prime):
                                candidate_transitions.append(((s, t), l, (s_prime, t_prime)))
                                any_transition = True
                    else:
                        if dominance_pruning._dominates_in_all_other_factors(i, l, "noop"):
                            candidate_transitions.append(((s, t), l, (s_prime, s)))
                            any_transition = True

                    if not any_transition:
                        candidate_transitions.append(((s, t), l, (s_prime, FactorState("⊥", -1))))

                    for (s,t), l, (s_prime, t_prime) in self._select_transitions(candidate_transitions, i, dominance_relations[i]):
                        transitions.add((state_name(s,t), l, state_name(s_prime, t_prime)))
                        if state_name(s_prime, t_prime) not in states:
                            states.add(state_name(s_prime, t_prime))
                            worklist.append((s_prime, t_prime))

                for l in labels_not_applicable_at_s:
                    # Map this to a universally true state
                    transitions.add((state_name(s,t), l, universal_true))

            lts = LabelledTransitionSystem(
                'qdom_' + factor.name,
                list(states),
                list(transitions),
                initial_state=state_name(*initial_state_bot),
                goal_states=goal_states
            )
            lts_comp = complement_lts(lts)
            lts_comp.remove_deadends_and_unreachable()
            new_factors.append(lts_comp)

        # Remove dead labels
        dead_labels = {l for l in task.labels if any(len(f.transitions_of_label(l)) == 0 for f in new_factors)}
        for f in new_factors:
            f.transitions = [t for t in f.transitions if t[1] not in dead_labels]
            f._compute_cached_values()

        res = FactoredTask(f'qdom_{task.name}', *new_factors, label_costs={l: c for l,c in task.label_costs.items() if l not in dead_labels})
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            Path("post_transformation_task.dot").write_text(res.to_dot())
        return res


    def _select_transitions(self, candidate_transitions: list, i: int, dom_rel) -> list:
        if len(candidate_transitions) <= 1:
            return candidate_transitions
        elif any((s_prime, t_prime) in dom_rel for _, _, (s_prime, t_prime) in candidate_transitions):
            (s,t), l, _ = candidate_transitions[0]
            return [((s,t), l, (s,s))] # Add transition to universal true, e.g. identity here
        elif self.approximate_to_deterministic:
            # Only select one of the candidate transitions
            def cmp_transition(tr1, tr2):
                s1, t1 = tr1[2]
                s2, t2 = tr2[2]
                assert s1 == s2
                # Prefer transitions to states that dominate the other transition's target state
                t2_dom_t1 = (t1, t2) in dom_rel
                t1_dom_t2 = (t2, t1) in dom_rel
                if t2_dom_t1 and not t1_dom_t2:
                    return -1
                elif t1_dom_t2 and not t2_dom_t1:
                    return 1
                else:
                    # Prefer trasitions that keep one step behind the s state (this might not make that much sense)
                    if t1 == tr1[0][0]:
                        return 1
                    elif t2 == tr2[0][0]:
                        return -1
                    else:
                        return 0

            return [max(candidate_transitions, key=cmp_to_key(cmp_transition))]
        else:
            return candidate_transitions


