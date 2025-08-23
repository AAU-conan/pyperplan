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
import itertools
from pathlib import Path
from typing import List, Set, Tuple, Any, Optional
from abc import ABC, abstractmethod

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
        return (
            self.name == other.name
            and self.preconditions == other.preconditions
            and self.add_effects == other.add_effects
            and self.del_effects == other.del_effects
        )

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


class FactorState:
    def __init__(self, name: str, value: int, bound: int):
        self.name = name
        self.value = value
        self.bound = bound  # Maximum value this factor can take

    def validate(self):
        assert self.value < self.bound, f"Value {self.value} of factor {self.name} must be less than its bound {self.bound}"

    def __str__(self):
        return f"{self.name}={self.value}"

    def __repr__(self):
        return f"{self.name}={self.value}"

    def __eq__(self, other):
        return other.value == self.value

    def __lt__(self, other):
        return self.value < other.value

    def __hash__(self):
        return self.value.__hash__()



class FactoredTaskState:
    def __init__(self, *states: FactorState, task: Optional['FactoredTask']=None):
        assert all(isinstance(s, FactorState) for s in states), "All states must be instances of FactorState"
        self.states: list[FactorState] = list(states)
        self.task = task

    def validate(self):
        for i, state in enumerate(self.states):
            if self.task:
                assert state.bound == self.task.factors[i].size()
            state.validate()

    def size(self):
        return len(self.states)

    def __str__(self):
        return f"<{', '.join(s.name for s in self.states)}>"

    def __repr__(self):
        return f"<{', '.join(s.name for s in self.states)}>"

    def __hash__(self):
        return hash(tuple(s.value for s in self.states))

    def __eq__(self, other):
        return all(s1.value == s2.value for s1, s2 in zip(self.states, other.states))

class LabelledTransitionSystem:
    def __init__(self, name: str, states: list[str], transitions: list[Tuple[str, str, str]], initial_state: str, goal_states: list[str]):
        self.name = name
        state_name_to_factor_state: dict[str, FactorState] = {state: FactorState(state, i, len(states)) for i, state in enumerate(states)}
        self.states: list[FactorState] = [fs for _, fs in state_name_to_factor_state.items()]
        self.initial_state: Optional[FactorState] = state_name_to_factor_state.get(initial_state, None)
        self.transitions: list[tuple[FactorState, str, FactorState]] = [(state_name_to_factor_state[src], label, state_name_to_factor_state[dst]) for src, label, dst in transitions]
        self.goal_states: set[FactorState] = {state_name_to_factor_state[goal] for goal in goal_states}

        self.state_label_to_targets = {(s,l): [t for si, li, t in self.transitions if s == si and l == li] for s, l, _ in self.transitions}

    @staticmethod
    def from_factored(name: str, states: list[FactorState], transitions: list[tuple[FactorState, str, FactorState]], initial_state: FactorState, goal_states: set[FactorState]):
        """
        Create a LabelledTransitionSystem from a factored representation.
        """
        dummy = LabelledTransitionSystem(name, [], [], '', [])
        dummy.states = states
        dummy.initial_state = initial_state
        dummy.transitions = transitions
        dummy.goal_states = goal_states
        dummy.state_label_to_targets = {(s,l): [t for si, li, t in transitions if s == si and l == li] for s, l, _ in transitions}
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

        bound = self.states[0].bound
        for s in self.states:
            assert s.bound == bound, f"All states must have the same bound, but {s.name} has {s.bound} while others have {bound}"
            s.validate()

        assert len(self.goal_states) > 0, "At least one goal state must be defined"

    def transitions_of_state(self, s: FactorState):
        """
        Get all transitions starting from a given state.
        Returns a list of tuples (label, target_state).
        """
        return [(label, target) for src, label, target in self.transitions if src == s]

    def transitions_to_state(self, t: FactorState) -> list[tuple[FactorState, str]]:
        """
        Get all transitions leading to a specific target state.
        Returns a list of tuples (source_state, label).
        """
        return [(src, label) for src, label, target in self.transitions if target == t]

    def transitions_of_label(self, l: str):
        """
        Get all transitions with a specific label.
        Returns a list of tuples (source_state, target_state).
        """
        return [(src, target) for src, label, target in self.transitions if label == l]

    def transitions_of_label_state(self, s: FactorState, l: str):
        """
        Get all transitions starting from a given state with a specific label.
        Returns a list of target states.
        """
        return [target for src, label, target in self.transitions if src == s and label == l]

    def to_dot(self) -> str:
        """
        Generate a dot representation of the labelled transition system.
        """
        dot_lines = []
        dot_lines.append("digraph LTS {")
        dot_lines.append(f'  label="{self.name}";')
        dot_lines.append('  rankdir=LR;')

        # Add states
        for state in self.states:
            dot_lines.append(f'  "{state.name}" [label="{state.name}"];')

        # Add transitions
        for src, label, dst in self.transitions:
            dot_lines.append(f'  "{src.name}" -> "{dst.name}" [label="{label}"];')

        # Add initial state
        if self.initial_state:
            dot_lines.append(f'  "Initial" [shape=point,visible=false];')
            dot_lines.append(f'  "Initial" -> "{self.initial_state.name}";')

        # Add goal states
        for goal in self.goal_states:
            dot_lines.append(f'  "{goal.name}" [peripheries=2];')

        dot_lines.append("}")
        return "\n".join(dot_lines)


class FactoredTask(Task):
    """
    A factored planning task
    Each factor is a labelled transition system, the state-space is the synchronized product
    """
    STATE_TYPE = FactoredTaskState

    def __init__(self, name: str, *factors: LabelledTransitionSystem):
        self.factors: tuple[LabelledTransitionSystem, ...] = factors
        self.labels = {label for factor in self.factors for _, label, _ in factor.transitions}
        self.validate()
        super().__init__(name, FactoredTaskState(*[FactorState(factor.initial_state.name, factor.initial_state.value, factor.initial_state.bound) for factor in self.factors], task=self))

    def size(self):
        return len(self.factors)

    def validate(self):
        for factor in self.factors:
            factor.validate()

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
        for label in self.labels:
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
                        if pre == -1: # -1 means no precondition
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

        return FactoredTask(name, *factors)

    def to_dot(self) -> str:
        """
        Generate a DOT representation of the factored task.
        """
        dot_lines = []
        dot_lines.append("digraph FactoredTask {")
        dot_lines.append(f'  label="{self.name}";')
        dot_lines.append('  rankdir=LR;')

        # Add states
        for factor in self.factors:
            for state in factor.states:
                dot_lines.append(f'  "{state.name}" [label="{state.name}"];')

        # Add transitions
        for factor in self.factors:
            for src, label, dst in factor.transitions:
                dot_lines.append(f'  "{src.name}" -> "{dst.name}" [label="{label}"];')

        # Add initial state
        dot_lines.append(f'  "Initial" [shape=point];')
        for factor in self.factors:
            dot_lines.append(f'  "Initial" -> "{factor.initial_state.name}";')

        # Add goal states
        for factor in self.factors:
            for goal in factor.goal_states:
                dot_lines.append(f'  "{goal.name}" [peripheries=2];')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def save_dot(self, path: Path):
        path.open("w").write(self.to_dot())