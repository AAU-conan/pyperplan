import heapq
import logging
from typing import Optional

from pyperplan.cli import cli_register
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactoredTaskState, FactorState


@cli_register("hscp")
class SaturatedCostPartitioning(Heuristic):
    """
    This is just my implementation of what I think saturated cost partitioning kind of is, without having read about it
    in detail.

    We start by computing the goal cost of each state in a factor. Then we assigned the smallest possible cost to each
    label s.t. each state still has the same goal cost. We then continue with that procedure in all factors. Eventually,
    all labels should have 0-cost. To compute the heuristic value, we just sum the goal distances of each factor.
    """

    def __init__(self, task: FactoredTask, order: Optional[list[int]] = None, only_reachable_from: Optional[FactoredTaskState] = None):
        super().__init__(task)
        self.task = task
        self.factor_goal_costs: list[dict[FactorState, int]]

        if order is None:
            # order = list(sorted(range(task.size()), key=lambda i: -len(task.factors[i].goal_states)))
            order = list(range(task.size()))
        self._compute_saturated_cost_partitioning(order, only_reachable_from)

    def _compute_saturated_cost_partitioning(self, order, only_reachable_from: Optional[FactoredTaskState]):
        label_costs = self.task.label_costs.copy()
        self.factor_goal_costs = [None for _ in range(self.task.size())]

        for i in order:
            factor = self.task.factors[i]
            if only_reachable_from is not None:
                # We only ensure the goal cost is preserved from states reachable from the given state
                reachable = {only_reachable_from.states[i]}
                worklist = list(reachable)
                while worklist:
                    state = worklist.pop()
                    for label, next_state in factor.transitions_of_state(state):
                        if next_state not in reachable:
                            reachable.add(next_state)
                            worklist.append(next_state)

                # logging.debug(f"{len(reachable)} reachable out of {len(factor.states)} states in factor {factor.name}")
            else:
                reachable = set(factor.states)

            goal_costs = {state: float("inf") if state not in factor.goal_states else 0 for state in factor.states}

            # Do dijkstra-like search to compute the goal costs
            queue = [(0, state) for state in factor.goal_states]
            heapq.heapify(queue)
            while queue:
                cost, state = heapq.heappop(queue)
                if cost > goal_costs[state]:
                    continue

                for prev_state, label in factor.transitions_to_state(state):
                    prev_cost = cost + label_costs[label]
                    if prev_cost < goal_costs[prev_state]:
                        goal_costs[prev_state] = prev_cost
                        heapq.heappush(queue, (prev_cost, prev_state))

            self.factor_goal_costs[i] = goal_costs

            # Now, for each label we compute the minimum cost it can have and still preserve the goal costs
            min_label_costs = {label: float("-inf") for label in self.task.labels}

            for label in self.task.labels:
                for s, t in factor.transitions_of_label(label):
                    if s in reachable:
                        min_label_costs[label] = max(min_label_costs[label], goal_costs[s] - goal_costs[t])

            # Subtract the minimum cost from all labels
            label_costs = {label: cost - min_label_costs[label] for label, cost in label_costs.items()}

    def __call__(self, node: "SearchNode"):
        """
        Compute the heuristic value by summing the saturated goal costs for each state.
        """
        state: FactoredTaskState = node.state
        return sum(self.factor_goal_costs[i][state.states[i]] for i in range(self.task.size()))
