import copy
import logging
from pathlib import Path
from typing import Type, TYPE_CHECKING

from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.heuristics.saturated_cost_partitioning import SaturatedCostPartitioningHeuristic
from pyperplan.pruning.dominance_pruning import DominancePruning
from pyperplan.task import FactoredTask, LabelledTransitionSystem, FactorState, FactoredTaskState

from pyperplan.search.searchspace import SearchNode


def determinize_lts(lts: LabelledTransitionSystem) -> LabelledTransitionSystem:
    """
    Returns a deterministic version of the labelled transition system. This is done by the subset construction algorithm.
    We need to preserve all states in the original lts.
    """
    # The states in the new transition system are sets of states from the original lts.
    state_set_to_factored_state = {frozenset([state]): FactorState(f'{state.name}', state.value, -1) for state in lts.states}
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
                state_set_to_factored_state[target_set] = FactorState(
                    f'{{{",".join(s.name for s in target_set)}}}',
                    len(state_set_to_factored_state),
                    -1
                )
                worklist.append(target_set)

            transitions.append((state_set_to_factored_state[current_set], label, state_set_to_factored_state[target_set]))

    return LabelledTransitionSystem.from_factored(
        lts.name + '_determinized',
        [v for _, v in state_set_to_factored_state.items()],
        transitions,
        initial_state=state_set_to_factored_state[frozenset([lts.initial_state])],
        goal_states={fs for state_set, fs in state_set_to_factored_state.items() if any(s in lts.goal_states for s in state_set)}
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



class QualifiedDominanceHeuristic(Heuristic):
    """
    A heuristic that uses qualified dominance to estimate the cost of reaching the goal in contrast to a contrastive set
    of states. It uses a dominance relation to compute a qualified dominance finite automaton for each factor. Then, during
    search, it creates a new planning task with added qualified dominance automaton as factors for each previous state that
    it compares to.
    """

    def __init__(self, task: FactoredTask, base_heuristic: Type[Heuristic], intersect_original_factor: bool = False):
        super().__init__()
        self.dominance_pruning = DominancePruning(task)
        self.heuristic: Type[Heuristic] = base_heuristic
        self.intersect_original_factor = intersect_original_factor
        self.task: FactoredTask = task
        self.qdom_factors: list[LabelledTransitionSystem] = []
        self.qdom_factors_state_maps: list[dict[tuple[FactorState,FactorState], FactorState]] = [] # Maps from the original factor states to the states in the qualified dominance automaton
        self._compute_qualified_dominance()
        self.seen_states: list[tuple[int, FactoredTaskState]] = []

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
        Path("fts.dot").open("w").write(self.task.to_dot())
        logging.debug("Computing Qualified Dominance Automata...")
        for i, factor in enumerate(self.task.factors):
            def state_name(s: FactorState, t: FactorState):
                return f"{s.name} < {t.name}"
            states = [state_name(s,t) for s in factor.states for t in factor.states] + [state_name(s, FactorState("⊥", -1, -1)) for s in factor.states]
            universal_true = "TRUE"
            states.append(universal_true)

            transitions = set((universal_true, l, universal_true) for l in self.task.labels)
            goal_states = [universal_true]

            for s in factor.states:
                for t in factor.states + [FactorState("⊥", -1, -1)]:
                    if s not in factor.goal_states or (t.name != '⊥' and t in factor.goal_states):
                        goal_states.append(state_name(s, t))

                    labels_not_applicable_at_s = self.task.labels.copy()
                    for l, s_prime in factor.transitions_of_state(s):
                        labels_not_applicable_at_s.remove(l)
                        any_transition = False
                        # NOOP
                        if t.name != '⊥':
                            if self.dominance_pruning._dominates_in_all_other_factors(i, l, "noop"):
                                transitions.add((state_name(s, t), l, state_name(s_prime, t)))
                                any_transition = True

                            # Actual transition
                            for l_prime, t_prime in factor.transitions_of_state(t):
                                if self.dominance_pruning._dominates_in_all_other_factors(i, l, l_prime):
                                    transitions.add((state_name(s, t), l, state_name(s_prime, t_prime)))
                                    any_transition = True

                        if not any_transition and self.intersect_original_factor:
                            transitions.add((state_name(s, t), l, state_name(s_prime, FactorState("⊥", -1, -1))))

                    for l in labels_not_applicable_at_s:
                        # Map this to a universally true state
                        transitions.add((state_name(s,t), l, universal_true))


            lts = LabelledTransitionSystem(
                'qdom_' + factor.name,
                states,
                list(transitions),
                initial_state=states[0], # Fake initial state, not used
                goal_states=goal_states
            )
            Path(f"qdom_{i}.dot").open("w").write(lts.to_dot())
            self.qdom_factors_state_maps.append({p: lts.states[i] for i, p in enumerate((s,t) for s in factor.states for t in factor.states)})
            lts_comp = complement_lts(lts)

            Path(f"nqdom_{i}.dot").open("w").write(lts_comp.to_dot())
            self.qdom_factors.append(lts_comp)
            logging.debug(f"Automaton for {factor.name} has {len(lts_comp.states)} states and {len(lts_comp.transitions)} transitions.")

    def __call__(self, node: 'SearchNode'):
        """
        Calls the base heuristic and returns the estimated cost.
        """
        self.extended_task.factors = self.task.factors.copy()
        state: FactoredTaskState = copy.deepcopy(node.state)
        for prev_g, prev_state in self.seen_states:
            if prev_g <= node.g:
                not_dom_factors = [i for i in range(self.task.size()) if (state.states[i], prev_state.states[i]) not in self.dominance_pruning.dominance_relations[i]]
                if len(not_dom_factors) == 0:
                    # This state is completely dominated, assign h=∞
                    return float('inf')
                elif len(not_dom_factors) == 1:
                    # This state is dominated in all but one factor, add the qualified dominance automaton for that factor
                    ndf = not_dom_factors[0]
                    self.extended_task.factors.append(self.qdom_factors[ndf])
                    state.states.append(self.qdom_factors_state_maps[ndf][(state.states[ndf], prev_state.states[ndf])])
                else:
                    # This state is not-dominated in multiple factors, it cannot be used
                    pass

        if self.heuristic == SaturatedCostPartitioningHeuristic:
            h = self.heuristic(self.extended_task,
                               # order=range(extended_task.size())[::-1],
                               only_reachable_from=state
            )
        else:
            h = self.heuristic(self.extended_task)

        # print(f"Evaluating {state}")
        # print(extended_task.save_dot(Path("extended_task.dot")))

        self.seen_states.append((node.g, node.state))
        value = h(SearchNode(state, node.parent, node.action, node.g))
        # print(f"h={value}")
        return value