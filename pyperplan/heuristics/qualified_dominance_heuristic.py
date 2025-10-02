import copy
from functools import cmp_to_key
import logging
from pathlib import Path
from typing import Generator, Type, TYPE_CHECKING

from pyperplan.cli import cli_register
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.pruning.dominance_analysis import DominanceAnalysis
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import (
    FactoredTask,
    FactoredTaskState,
    FactorState,
    LabelledTransitionSystem,
)
from pyperplan.task_transformation.noop_task_transformation import NoopTaskTransformation


def determinize_lts(lts: LabelledTransitionSystem) -> LabelledTransitionSystem:
    """
    Returns a deterministic version of the labelled transition system. This is done by the subset construction algorithm.
    We need to preserve all states in the original lts.
    """
    # The states in the new transition system are sets of states from the original lts.
    state_set_to_factored_state = {frozenset([state]): FactorState(f"{state.name}", state.value) for state in lts.states}
    worklist = [frozenset([state]) for state in lts.states]
    labels = set(l for _, l, _ in lts.transitions)

    transitions = []

    while worklist:
        current_set = worklist.pop()
        # logging.debug(f'states {len(state_set_to_factored_state)}, worklist {len(worklist)}')

        for label in labels:
            # Compute the target set state by taking the union of all target states
            target_set = frozenset({t for s in current_set for t in lts.transitions_of_label_state(s, label)})

            if target_set not in state_set_to_factored_state:
                state_set_to_factored_state[target_set] = FactorState(f'{{{",".join(s.name for s in target_set)}}}', len(state_set_to_factored_state))
                worklist.append(target_set)

            transitions.append((state_set_to_factored_state[current_set], label, state_set_to_factored_state[target_set]))

    return LabelledTransitionSystem.from_factored(
        lts.name + "_determinized",
        [FactorState(v.name, v.value) for _, v in state_set_to_factored_state.items()],
        transitions,
        initial_state=state_set_to_factored_state[frozenset([lts.initial_state])],
        goal_states={fs for state_set, fs in state_set_to_factored_state.items() if any(s in lts.goal_states for s in state_set)},
    )


def complement_lts(lts: LabelledTransitionSystem) -> LabelledTransitionSystem:
    """
    Returns the complement of a labelled transition system.
    The complement is the transition system where the set of plans is complement. Since the original lts is non-deterministic,
    this requires to first determinize the lts and then swap goal/non-goal states. Additionally, we need to preserve all
    states in the lts.
    """
    det_lts = determinize_lts(lts)
    # Swap goal and non-goal states
    goal_states = set(det_lts.states) - set(det_lts.goal_states)
    det_lts.goal_states = goal_states
    return det_lts


class ComparisonStrategy:
    """
    Interface for comparison strategies used in the qualified dominance heuristic. This decides which states to try
    to compare the current state to.
    """

    def __init__(self, qdom: "QualifiedDominanceHeuristic"):
        self.qdom = qdom

    def compare_factor(self, node: SearchNode, previous_g: int, previous_state: FactoredTaskState) -> int | None:
        """
        Returns the factor index that can be used for qualified dominance comparison, -1 if all are dominated, i if
        only factor i is not dominated, or None if no qualified dominance comparison is possible.
        """
        if previous_g <= node.g:
            not_dom_factors = [
                i for i in range(self.qdom.task.size()) if (node.state.states[i], previous_state.states[i]) not in self.qdom.dominance_analysis.dominance_relations[i]
            ]
            if len(not_dom_factors) == 0:
                return -1
            elif len(not_dom_factors) == 1:
                return not_dom_factors[0]
        return None

    def get_compare_states(self, node: SearchNode) -> Generator[tuple[FactoredTaskState, int], None, None]:
        """
        Yields (state, not-dominated-factor) to compare the current node to.
        """
        raise NotImplementedError()


@cli_register()
class AllComparisonStrategy(ComparisonStrategy):
    """
    A comparison strategy that compares the current state to all previously seen states.
    """

    def __init__(self, qdom: "QualifiedDominanceHeuristic"):
        super().__init__(qdom)
        self.seen_states: list[tuple[int, FactoredTaskState]] = []

    def get_compare_states(self, node: SearchNode) -> Generator[tuple[FactoredTaskState, int], None, None]:
        for previous_g, previous_state in self.seen_states:
            ndf = self.compare_factor(node, previous_g, previous_state)
            if ndf is not None:
                yield previous_state, ndf
        self.seen_states.append((node.g, node.state))


@cli_register()
class ParentComparisonStrategy(ComparisonStrategy):
    """
    A comparison strategy that only compares the current state to its parent state.
    """

    def get_compare_states(self, node: SearchNode) -> Generator[tuple[FactoredTaskState, int], None, None]:
        if node.parent is not None:
            ndf = self.compare_factor(node, node.parent.g, node.parent.state)
            if ndf is not None:
                yield node.parent.state, ndf


@cli_register("hqdom")
class QualifiedDominanceHeuristic(Heuristic):
    """
    A heuristic that uses qualified dominance to estimate the cost of reaching the goal in contrast to a contrastive set
    of states. It uses a dominance relation to compute a qualified dominance finite automaton for each factor. Then, during
    search, it creates a new planning task with added qualified dominance automaton as factors for each previous state that
    it compares to.
    """

    _AUTOMATA_TR = tuple[tuple[FactorState, FactorState], str, tuple[FactorState, FactorState]]

    def __init__(
        self,
        task: FactoredTask,
        base_heuristic: Type[Heuristic],
        comparison_strategy: type[ComparisonStrategy] = AllComparisonStrategy,
        intersect_original_factor: bool = False,
        approximate_to_deterministic: bool = False,
    ):
        super().__init__(task)
        self.dominance_analysis = DominanceAnalysis(NoopTaskTransformation().transform(copy.deepcopy(task)))
        self.heuristic: Type[Heuristic] = base_heuristic
        self.intersect_original_factor = intersect_original_factor
        self.task: FactoredTask = task
        self.qdom_factors: list[LabelledTransitionSystem] = []
        self.qdom_factors_state_maps: list[dict[tuple[FactorState, FactorState], FactorState]] = (
            []
        )  # Maps from the original factor states to the states in the qualified dominance automaton
        self.approximate_to_deterministic = approximate_to_deterministic

        self._compute_qualified_dominance()

        self.comparison_strategy = comparison_strategy(self)
        self.extended_task = copy.deepcopy(task)

    def _compute_qualified_dominance(self):
        """
        Computes the qualified dominance automata for the task.
        For each factor, compute an automaton over the states S×(S \cup ⊥) where S is the set of states of the factor.
        The ⊥ state represents the none state, that has no transitions and is a goal state.
        Each state (s,t) is accepting iff s is not a goal or t is a goal.
        For each state (s,t):
            add a transition to ⊤ (universal true) if ∄ s -l->
            otherwise add a transition to the state (s',t') with label l if
                ∃ s -l-> s' and ∃ t -l'-> t' s.t. l' dominates l in all other factors
            otherwise add a transition to (s',⊥) with label l
        """
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            Path("fts.dot").open("w").write(self.task.to_dot())
        logging.debug("Computing Qualified Dominance Automata...")
        for i, factor in enumerate(self.task.factors):
            universal_true = "TRUE"

            def state_name(s: FactorState, t: FactorState):
                if self.dominance_analysis.dominance_relations[i].dominates(s,t):
                    return universal_true
                return f"{s.name} < {t.name}"

            states = [state_name(s, t) for s in factor.states for t in factor.states if  not self.dominance_analysis.dominance_relations[i].dominates(s, t)] + [
                state_name(s, FactorState("⊥", -1)) for s in factor.states
            ]
            states.append(universal_true)

            transitions = set((universal_true, l, universal_true) for l in self.task.labels)
            goal_states = [universal_true]

            for s in factor.states:
                for t in factor.states + [FactorState("⊥", -1)]:
                    if (s, t) in self.dominance_analysis.dominance_relations[i]:
                        continue

                    if s not in factor.goal_states or (t.name != "⊥" and t in factor.goal_states):
                        goal_states.append(state_name(s, t))

                    labels_not_applicable_at_s = self.task.labels.copy()
                    for l, s_prime in factor.transitions_of_state(s):
                        labels_not_applicable_at_s.remove(l)
                        any_transition = False
                        candidate_transitions: list[tuple[tuple[FactorState, FactorState], str, tuple[FactorState, FactorState]]] = []
                        if t.name != "⊥":
                            # Actual transition
                            for l_prime, t_prime in factor.transitions_of_state(t):
                                if self.dominance_analysis.label_relation.dominates_in_all_other_factors(i, l, l_prime):
                                    candidate_transitions.append(((s, t), l, (s_prime, t_prime)))
                                    any_transition = True

                        if not any_transition and self.intersect_original_factor:
                            candidate_transitions.append(((s, t), l, (s_prime, FactorState("⊥", -1))))

                        for (s, t), l, (s_prime, t_prime) in self._select_transitions(candidate_transitions, i):
                            transitions.add((state_name(s, t), l, state_name(s_prime, t_prime)))

                    for l in labels_not_applicable_at_s:
                        # Map this to a universally true state
                        transitions.add((state_name(s, t), l, universal_true))

            lts = LabelledTransitionSystem("qdom_" + factor.name, states, list(transitions), initial_state=states[0], goal_states=goal_states)  # Fake initial state, not used
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                Path(f"qdom_{i}.dot").open("w").write(lts.to_dot())
            self.qdom_factors_state_maps.append(
                {p: lts.states[i] for i, p in enumerate((s, t) for s in factor.states for t in factor.states if not self.dominance_analysis.dominance_relations[i].dominates(s, t))}
            )
            lts_comp = complement_lts(lts)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                Path(f"nqdom_{i}.dot").open("w").write(lts_comp.to_dot())
            self.qdom_factors.append(lts_comp)
            logging.debug(f"Automaton for {factor.name} has {len(lts_comp.states)} states and {len(lts_comp.transitions)} transitions.")

    def _select_transitions(self, candidate_transitions: list[_AUTOMATA_TR], i: int) -> list[_AUTOMATA_TR]:
        if len(candidate_transitions) <= 1:
            return candidate_transitions
        elif any((s_prime, t_prime) in self.dominance_analysis.dominance_relations[i] for _, _, (s_prime, t_prime) in candidate_transitions):
            (s, t), l, _ = candidate_transitions[0]
            return [((s, t), l, (s, s))]  # Add transition to universal true, e.g. identity here
        elif self.approximate_to_deterministic:
            # Only select one of the candidate transitions
            def cmp_transition(tr1: QualifiedDominanceHeuristic._AUTOMATA_TR, tr2: QualifiedDominanceHeuristic._AUTOMATA_TR):
                s1, t1 = tr1[2]
                s2, t2 = tr2[2]
                assert s1 == s2
                # Prefer transitions to states that dominate the other transition's target state
                t2_dom_t1 = (t1, t2) in self.dominance_analysis.dominance_relations[i]
                t1_dom_t2 = (t2, t1) in self.dominance_analysis.dominance_relations[i]
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

    def __call__(self, node: "SearchNode"):
        """
        Calls the base heuristic and returns the estimated cost.
        """
        self.extended_task.factors = self.task.factors.copy()
        state: FactoredTaskState = node.state.copy()
        for prev_state, ndf in self.comparison_strategy.get_compare_states(node):
            if ndf == -1:
                return float("inf")
            # This state is dominated in all but one factor, add the qualified dominance automaton for that factor
            self.extended_task.factors.append(self.qdom_factors[ndf])
            state.states.append(self.qdom_factors_state_maps[ndf][(state.states[ndf], prev_state.states[ndf])])

        h = self.heuristic(self.extended_task)

        # print(f"Evaluating {state}")
        # print(extended_task.save_dot(Path("extended_task.dot")))

        value = h(SearchNode(state, node.parent, node.action, node.g))
        # print(f"h={value}")
        return value
