#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

"""
Classes for representing a STRIPS planning task
"""
from abc import ABC, abstractmethod
import itertools
import logging
from pathlib import Path
import timeit
from typing import Any, List, Optional, Set, Tuple

from pyperplan.string_compactification import StringCompactifier
from pyperplan.translate.sas_tasks import SASTask


class Operator:
    """
    The preconditions represent the facts that have to be true
    before the operator can be applied.
    add_effects are the facts that the operator makes true.
    delete_effects are the facts that the operator makes false.
    """

    def __init__(self, name: str, preconditions: Set[str], add_effects: Set[str], del_effects: Set[str]):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    def applicable(self, state: frozenset) -> bool:
        """
        Operators are applicable when their set of preconditions is a subset
        of the facts that are true in "state".

        @return True if the operator's preconditions is a subset of the state,
                False otherwise
        """
        return self.preconditions <= state

    def apply(self, state: frozenset) -> frozenset:
        """
        Applying an operator means removing the facts that are made false
        by the operator from the set of true facts in state and adding
        the facts made true.

        Note that therefore it is possible to have operands that make a
        fact both false and true. This results in the fact being true
        at the end.

        @param state The state that the operator should be applied to
        @return A new state (set of facts) after the application of the
                operator
        """
        assert self.applicable(state)
        assert type(state) in (frozenset, set)
        return (state - self.del_effects) | self.add_effects

    def __eq__(self, other):
        return self.name == other.name and self.preconditions == other.preconditions and self.add_effects == other.add_effects and self.del_effects == other.del_effects

    def __hash__(self) -> int:
        return hash((self.name, self.preconditions, self.add_effects, self.del_effects))

    def __str__(self):
        s = "%s\n" % self.name
        for group, facts in [
            ("PRE", self.preconditions),
            ("ADD", self.add_effects),
            ("DEL", self.del_effects),
        ]:
            for fact in facts:
                s += f"  {group}: {fact}\n"
        return s

    def __repr__(self):
        return "<Op %s>" % self.name


class Task(ABC):
    STATE_TYPE = Any

    def __init__(self, name: str, initial_state: STATE_TYPE):
        self.name = name
        self.initial_state = initial_state

    @abstractmethod
    def goal_reached(self, state: STATE_TYPE) -> bool:
        pass

    @abstractmethod
    def get_successor_states(self, state: STATE_TYPE) -> List[Tuple[Operator, STATE_TYPE]]:
        pass

    def get_action_cost(self, label: str) -> int:
        return 1


class STRIPSTask(Task):
    """
    A STRIPS planning task
    """

    STATE_TYPE = frozenset

    def __init__(self, name: str, facts: Set[str], initial_state: frozenset, goals: frozenset, operators: List[Operator]):
        """
        @param name The task's name
        @param facts A set of all the fact names that are valid in the domain
        @param initial_state A set of fact names that are true at the beginning
        @param goals A set of fact names that must be true to solve the problem
        @param operators A set of operator instances for the domain
        """
        self.name = name
        self.facts = facts
        self.initial_state = initial_state
        self.goals = goals
        self.operators = operators

    def goal_reached(self, state: frozenset) -> bool:
        """
        The goal has been reached if all facts that are true in "goals"
        are true in "state".

        @return True if all the goals are reached, False otherwise
        """
        return self.goals <= state

    def get_successor_states(self, state: frozenset) -> List[Tuple[Operator, frozenset]]:
        """
        @return A list with (op, new_state) pairs where "op" is the applicable
        operator and "new_state" the state that results when "op" is applied
        in state "state".
        """
        return [(op, op.apply(state)) for op in self.operators if op.applicable(state)]

    def __str__(self):
        s = "Task {0}\n  Vars:  {1}\n  Init:  {2}\n  Goals: {3}\n  Ops:   {4}"
        return s.format(
            self.name,
            ", ".join(self.facts),
            self.initial_state,
            self.goals,
            "\n".join(map(repr, self.operators)),
        )

    def __repr__(self):
        string = "<Task {0}, vars: {1}, operators: {2}>"
        return string.format(self.name, len(self.facts), len(self.operators))


def pretty_name(name: str) -> str:
    return name.replace("NegatedAtom ", "~").replace("Atom ", "").replace("()", "")


class FactorState:
    def __init__(self, name: str, value: int):
        self.name = pretty_name(name)
        self.value = value

    def validate(self):
        return True

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}={self.value}"

    def __eq__(self, other):
        return other.value == self.value

    def __lt__(self, other):
        return self.value < other.value

    def __hash__(self):
        return self.value.__hash__()


class FactoredTaskState:
    def __init__(self, *states: FactorState, task: Optional["FactoredTask"] = None):
        assert all(isinstance(s, FactorState) for s in states), "All states must be instances of FactorState"
        self.states: list[FactorState] = list(states)
        self.task = task

    def validate(self):
        for i, state in enumerate(self.states):
            state.validate()

    def size(self):
        return len(self.states)

    def __str__(self):

        return f"<{', '.join(pretty_name(s.name) for s in self.states)}>"

    def __repr__(self):
        return f"<{', '.join(s.name for s in self.states)}>"

    def __hash__(self):
        return hash(tuple(s.value for s in self.states))

    def __eq__(self, other):
        return all(s1.value == s2.value for s1, s2 in zip(self.states, other.states))

    def copy(self):
        return FactoredTaskState(*self.states, task=self.task)


class LabelledTransitionSystem:
    def __init__(self, name: str, states: list[str], transitions: list[Tuple[str, str, str]], initial_state: str, goal_states: list[str]):
        self.name = name
        state_name_to_factor_state: dict[str, FactorState] = {state: FactorState(state, i) for i, state in enumerate(states)}
        self.states: list[FactorState] = [fs for _, fs in state_name_to_factor_state.items()]
        self.initial_state: Optional[FactorState] = state_name_to_factor_state.get(initial_state, None)
        self.transitions: list[tuple[FactorState, str, FactorState]] = [
            (state_name_to_factor_state[src], label, state_name_to_factor_state[dst]) for src, label, dst in transitions
        ]
        self.goal_states: set[FactorState] = {state_name_to_factor_state[goal] for goal in goal_states}

        self.state_label_to_targets: dict[tuple[FactorState, str], list[FactorState]]
        self.state_to_label_targets: dict[tuple[FactorState], list[tuple[str, FactorState]]]
        self.label_to_source_targets: dict[str, list[tuple[FactorState, FactorState]]]
        self.target_to_source_labels: dict[FactorState, list[tuple[FactorState, str]]]
        self._compute_cached_values()

    def _compute_cached_values(self):
        self.state_label_to_targets = {}
        self.state_to_label_targets = {}
        self.label_to_source_targets = {}
        self.target_to_source_labels = {}
        for src, label, dst in self.transitions:
            if (src, label) not in self.state_label_to_targets:
                self.state_label_to_targets[(src, label)] = []
            self.state_label_to_targets[(src, label)].append(dst)

            if src not in self.state_to_label_targets:
                self.state_to_label_targets[src] = []
            self.state_to_label_targets[src].append((label, dst))

            if label not in self.label_to_source_targets:
                self.label_to_source_targets[label] = []
            self.label_to_source_targets[label].append((src, dst))

            if dst not in self.target_to_source_labels:
                self.target_to_source_labels[dst] = []
            self.target_to_source_labels[dst].append((src, label))

    @staticmethod
    def from_factored(name: str, states: list[FactorState], transitions: list[tuple[FactorState, str, FactorState]], initial_state: FactorState, goal_states: set[FactorState]):
        """
        Create a LabelledTransitionSystem from a factored representation.
        """
        dummy = LabelledTransitionSystem(name, [], [], "", [])
        dummy.states = states
        dummy.initial_state = initial_state
        dummy.transitions = transitions
        dummy.goal_states = goal_states
        dummy._compute_cached_values()
        return dummy

    def apply_label(self, state: FactorState, label: str) -> List[FactorState]:
        """
        Apply a label to a state and return the resulting state if the transition exists.
        """
        return self.state_label_to_targets.get((state, label), [])

    def validate(self):
        """
        Validate the LTS by checking if all transitions are valid and if the initial state is defined.
        """
        assert self.initial_state is not None, "Initial state must be defined"
        for src, label, dst in self.transitions:
            assert src in self.states
            assert dst in self.states

        for s in self.states:
            s.validate()

        if logging.getLogger().isEnabledFor(logging.DEBUG) and len(self.goal_states) == 0:
            logging.warning("No goal state is defined")

    def transitions_of_state(self, s: FactorState):
        """
        Get all transitions starting from a given state.
        Returns a list of tuples (label, target_state).
        """
        return self.state_to_label_targets.get(s, [])

    def transitions_to_state(self, t: FactorState) -> list[tuple[FactorState, str]]:
        """
        Get all transitions leading to a specific target state.
        Returns a list of tuples (source_state, label).
        """
        return self.target_to_source_labels.get(t, [])

    def transitions_of_label(self, l: str):
        """
        Get all transitions with a specific label.
        Returns a list of tuples (source_state, target_state).
        """
        return self.label_to_source_targets.get(l, [])

    def transitions_of_label_state(self, s: FactorState, l: str):
        """
        Get all transitions starting from a given state with a specific label.
        Returns a list of target states.
        """
        return self.state_label_to_targets.get((s, l), [])

    def to_dot(self) -> str:
        """
        Generate a dot representation of the labelled transition system.
        """
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.warning("Doing dot graph generation without debug.")
        dot_lines = []
        dot_lines.append("digraph LTS {")
        dot_lines.append(f'  label="{self.name}";')
        dot_lines.append("  rankdir=LR;")

        label_string_compactifier = StringCompactifier({l[1:-1] for _, l, _ in self.transitions})

        # Add states
        for state in self.states:
            dot_lines.append(
                f'  "{state.name}" [label="{state.name}",peripheries="{2 if state in self.goal_states else 1}",shape="{'cds' if state == self.initial_state else 'ellipse'}"];'
            )

        # Add transitions, group with same src and dst and different labels
        aggregated_edges = {}
        for src, label, dst in sorted(self.transitions):
            if (src, dst) not in aggregated_edges:
                aggregated_edges[(src, dst)] = []
            aggregated_edges[(src, dst)].append(label)

        for (src, dst), labels in sorted(aggregated_edges.items()):
            compact_labels = label_string_compactifier.compactify({l[1:-1] for l in labels})
            dot_lines.append(f'  "{src.name}" -> "{dst.name}" [label="{"\n".join(compact_labels)}"];')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def remove_deadends_and_unreachable(self):
        """
        Remove dead-end states that cannot reach a goal state and states unreachable from the initial state.
        """
        assert self.initial_state is not None, "Initial state must be defined"
        # Find all states that can reach a goal state using reverse graph search
        reachable_from_goal = {s for s in self.states if s in self.goal_states}
        worklist = list(reachable_from_goal)
        while worklist:
            state = worklist.pop()
            for src, _ in self.target_to_source_labels.get(state, []):
                if src not in reachable_from_goal:
                    reachable_from_goal.add(src)
                    worklist.append(src)

        # Find all states reachable from the initial state using forward graph search
        if self.initial_state not in reachable_from_goal:
            # Generate empty LTS
            self.states = [self.initial_state]
            self.transitions = []
            self.goal_states = set()
        else:
            reachable_from_init_and_goal = {self.initial_state}
            worklist = [self.initial_state]
            while worklist:
                state = worklist.pop()
                for _, tgt in self.state_to_label_targets.get(state, []):
                    if tgt not in reachable_from_init_and_goal and tgt in reachable_from_goal:
                        reachable_from_init_and_goal.add(tgt)
                        worklist.append(tgt)

            # Keep only states that are both reachable from the initial state and can reach a goal state
            self.states = list(reachable_from_init_and_goal)
            self.transitions = [(s, l, t) for s, l, t in self.transitions if s in reachable_from_init_and_goal and t in reachable_from_init_and_goal]
            self.goal_states = {s for s in self.goal_states if s in reachable_from_init_and_goal}

        self._compute_cached_values()

    @staticmethod
    def merge(left: "LabelledTransitionSystem", right: "LabelledTransitionSystem") -> "LabelledTransitionSystem":
        """
        Merge two LTS into a new LTS by taking the synchronized product.
        The new states are pairs of states from the two LTS.
        A transition exists if both LTS have a transition with the same label.
        The initial state is the pair of the initial states of the two LTS.
        The goal states are pairs where both states are goal states in their respective LTS.
        """
        assert left.initial_state is not None, "Left LTS must have an initial state"
        assert right.initial_state is not None, "Right LTS must have an initial state"

        def state_name(s1: FactorState, s2: FactorState) -> str:
            return f"({s1.name},{s2.name})"

        old_pair_to_new_state: dict[Tuple[FactorState, FactorState], FactorState] = {}

        def get_new_state(s1: FactorState, s2: FactorState) -> FactorState:
            if (s1, s2) not in old_pair_to_new_state:
                old_pair_to_new_state[(s1, s2)] = FactorState(state_name(s1, s2), len(old_pair_to_new_state))
            return old_pair_to_new_state[(s1, s2)]

        worklist = [(left.initial_state, right.initial_state)]
        states: set[FactorState] = {get_new_state(left.initial_state, right.initial_state)}
        transitions: list[tuple[FactorState, str, FactorState]] = []
        goal_states: set[FactorState] = set()

        while len(worklist) > 0:
            ls, rs = worklist.pop()
            new_state = get_new_state(ls, rs)
            if ls in left.goal_states and rs in right.goal_states:
                goal_states.add(new_state)
            left_transitions = left.transitions_of_state(ls)
            right_transitions = right.transitions_of_state(rs)

            for l_label, lt in left_transitions:
                for r_label, rt in right_transitions:
                    if l_label == r_label:
                        new_target_state = get_new_state(lt, rt)
                        transitions.append((new_state, l_label, new_target_state))
                        if new_target_state not in states:
                            worklist.append((lt, rt))
                            states.add(new_target_state)

        return LabelledTransitionSystem.from_factored(
            f"({left.name})_x_({right.name})", list(states), transitions, get_new_state(left.initial_state, right.initial_state), goal_states
        )

    @staticmethod
    def are_bisimilar(left: "LabelledTransitionSystem", right: "LabelledTransitionSystem") -> bool:
        """
        Check if two LTS are bisimilar.
        Initialize all pairs that agree on goal status as related.
        Iteratively remove pairs that do not simulate each other.
        """
        relation = set((ls, rs) for ls in left.states for rs in right.states if (ls in left.goal_states) == (rs in right.goal_states))

        changed = True
        while changed and (left.initial_state, right.initial_state) in relation:
            changed = False
            for ls, rs in relation.copy():
                removed = False
                for label, lt in left.transitions_of_state(ls):
                    for rt in right.transitions_of_label_state(rs, label):
                        if (lt, rt) in relation:
                            break
                    else:
                        relation.remove((ls, rs))
                        removed = True
                        changed = True
                        break
                if not removed:
                    for label, rt in right.transitions_of_state(rs):
                        for lt in left.transitions_of_label_state(ls, label):
                            if (lt, rt) in relation:
                                break
                        else:
                            relation.remove((ls, rs))
                            changed = True
                            break

        return (left.initial_state, right.initial_state) in relation


class FactoredTask(Task):
    """
    A factored planning task
    Each factor is a labelled transition system, the state-space is the synchronized product
    """

    STATE_TYPE = FactoredTaskState

    def __init__(self, name: str, *factors: LabelledTransitionSystem, label_costs: Optional[dict[str, int]] = None):
        self.factors: list[LabelledTransitionSystem] = list(factors)
        self.labels = {label for factor in self.factors for _, label, _ in factor.transitions}
        self._ordered_labels = list(sorted(self.labels))
        self.label_costs: dict[str, int] = {label: 1 for label in self.labels} if label_costs is None else label_costs
        super().__init__(name, FactoredTaskState(*[FactorState(factor.initial_state.name, factor.initial_state.value) for factor in self.factors], task=self))
        self.validate()

    def size(self):
        return len(self.factors)

    def validate(self):
        assert self.initial_state == FactoredTaskState(*(f.initial_state for f in self.factors))
        for factor in self.factors:
            factor.validate()

    def get_action_cost(self, label: str) -> int:
        return self.label_costs.get(label, 1)

    def goal_reached(self, state: FactoredTaskState) -> bool:
        """
        The goal has been reached if all factors' goal states are reached.
        """
        return all(s in self.factors[i].goal_states for i, s in enumerate(state.states))

    def get_successor_states(self, state: FactoredTaskState) -> List[Tuple[str, FactoredTaskState]]:
        """
        Get successor states for the factored task.
        Each factor's transitions are considered independently, and the resulting states are combined.
        """
        successors = []
        for label in self._ordered_labels:
            # Generate all combinations of successor states for each factor
            factor_successors = [factor.apply_label(state.states[i], label) for i, factor in enumerate(self.factors)]
            if all(len(fs) > 0 for fs in factor_successors):
                ps = list(itertools.product(*factor_successors))
                successors.extend((label, FactoredTaskState(*p, task=self)) for p in ps)
        return successors

    @staticmethod
    def from_sas_task(name: str, sas_task: SASTask):
        assert len(sas_task.axioms) == 0, "FactoredTask does not support axioms"
        factors = []
        label_costs = {op.name: op.cost for op in sas_task.operators}
        for i, v_size in enumerate(sas_task.variables.ranges):
            states: list[str] = [sas_task.variables.value_names[i][j] for j in range(v_size)]
            transitions: list[tuple[str, str, str]] = []
            for op in sas_task.operators:
                transitions_for_op = 0
                for var, val in op.prevail:
                    if var == i:
                        transitions.append((states[val], op.name, states[val]))
                        transitions_for_op += 1
                        assert transitions_for_op <= 1, f"Operator {op.name} has multiple prevail conditions for variable"

                for var, pre, post, cond in op.pre_post:
                    assert len(cond) == 0, "FactoredTask does not support conditional effects"
                    if var == i:
                        assert transitions_for_op == 0, f"Operator {op.name} has both prevail and pre/post conditions for variable {i}, or multiple pre/post conditions"
                        if pre == -1:  # -1 means no precondition
                            for s in states:
                                transitions.append((s, op.name, states[post]))
                        else:
                            transitions.append((states[pre], op.name, states[post]))
                        transitions_for_op += 1

                if transitions_for_op == 0:
                    # If there are no transitions for this operator, we can add a self-loop
                    for s in states:
                        transitions.append((s, op.name, s))

            goals = []
            for var, val in sas_task.goal.pairs:
                if var == i:
                    goals.append(states[val])
            if len(goals) == 0:
                goals = states

            factors.append(LabelledTransitionSystem(f"Factor {i}", states, transitions, states[sas_task.init.values[i]], goals))

        return FactoredTask(name, *factors, label_costs=label_costs)

    def to_dot(self) -> str:
        """
        Generate a DOT representation of the factored task. Use the dot of each factor.
        """
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.warning("Doing dot graph generation without debug.")
        res = "digraph FactoredTask {\n"
        for factor in self.factors:
            factor_dotlines = factor.to_dot().splitlines()[1:-1]  # Remove the first and last line (digraph ... { and })
            # Make subgraph for each factor
            res += f'  subgraph "cluster_{factor.name.replace(" ", "_")}" {{\n'
            for line in factor_dotlines:
                res += f"    {line}\n"
            res += "  }\n"
        res += "}\n"
        return res

    def save_dot(self, path: Path):
        path.open("w").write(self.to_dot())
