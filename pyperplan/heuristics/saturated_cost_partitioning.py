import heapq

from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactorState, FactoredTaskState


class SaturatedCostPartitioningHeuristic:
    """
    This is just my implementation of what I think saturated cost partitioning kind of is, without having read about it
    in detail.

    We start by computing the goal cost of each state in a factor. Then we assigned the smallest possible cost to each
    label s.t. each state still has the same goal cost. We then continue with that procedure in all factors. Eventually,
    all labels should have 0-cost. To compute the heuristic value, we just sum the goal distances of each factor.
    """
    def __init__(self, task: FactoredTask):
        self.task = task
        self.factor_goal_costs: list[dict[FactorState, int]]

        self._compute_saturated_cost_partitioning()

    def _compute_saturated_cost_partitioning(self):
        label_costs = {label: 1 for label in self.task.labels} # We only have unit cost actions in this case
        self.factor_goal_costs = []

        for factor in self.task.factors:
            goal_costs = {state: float('inf') if state not in factor.goal_states else 0 for state in factor.states}

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

            self.factor_goal_costs.append(goal_costs)

            # Now, for each label we compute the minimum cost it can have and still preserve the goal costs
            min_label_costs = {label: float('-inf') for label in self.task.labels}

            for label in self.task.labels:
                for s, t in factor.transitions_of_label(label):
                    min_label_costs[label] = max(min_label_costs[label], goal_costs[s] - goal_costs[t])

            # Subtract the minimum cost from all labels
            label_costs = {label: cost - min_label_costs[label] for label, cost in label_costs.items()}

    def __call__(self, node: 'SearchNode'):
        """
        Compute the heuristic value by summing the saturated goal costs for each state.
        """
        state: FactoredTaskState = node.state
        return sum(self.factor_goal_costs[i][state.states[i]] for i in range(self.task.size()))