import logging

import pulp
from pulp import *

from pyperplan.cli import cli_register
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import FactoredTask, FactoredTaskState


@cli_register("hocp")
class OptimalCostPartitioning(Heuristic):
    def __init__(self, task: FactoredTask):
        super().__init__(task)
        self.task = task

        self.factor_state_to_goal_cost = {}
        self._initialize_lp()

    def _initialize_lp(self):
        """
        Compute the optimal cost partitioning that maximizes the heuristic of the state.
        We use a linear programming approach to solve this problem.

        We have one variable per label per factor, representing the cost assigned to that label in that factor.
        We have one constraint per label, representing that the sum of the costs assigned to that label across all factors
        must be less than or equal to the original cost of that label.
        For each factor, we have one variable per state, representing the goal cost of that state in that factor.
        We have one constraint per transition s -l-> s', which is goal_cost(s) <= goal_cost(s') + cost_in_factor(l).
        We have one constraint per goal state s, which is goal_cost(s) = 0.
        """
        self.prob = LpProblem("OptimalCostPartitioning", LpMaximize)
        # self.solver = CPLEX_CMD(path="/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex", msg=False)
        self.solver = pulp.PULP_CBC_CMD(msg=False)

        label_factor_to_cost_var = {(l, i): LpVariable(f"{l}_in_{i}") for i in range(self.task.size()) for l in self.task.labels}

        for l in self.task.labels:
            self.prob += lpSum(label_factor_to_cost_var[(l, i)] for i in range(self.task.size())) <= self.task.label_costs[l], f"cost of label {l}"

        for i, factor in enumerate(self.task.factors):
            self.factor_state_to_goal_cost.update({(i, s): LpVariable(f"goal_cost({i},{s})") for s in factor.states})
            for s in factor.goal_states:
                self.prob += self.factor_state_to_goal_cost[(i, s)] == 0, f"goal_state_{s}_in_{i}"
            self_loop_labels = set()
            for s, l, t in factor.transitions:
                if s == t:
                    self_loop_labels.add(l)
                else:
                    self.prob += (
                        self.factor_state_to_goal_cost[(i, s)] <= self.factor_state_to_goal_cost[(i, t)] + label_factor_to_cost_var[(l, i)],
                        f"transition_{s}_{l}_{t}_in_{i}",
                    )

            for l in self_loop_labels:
                self.prob += label_factor_to_cost_var[(l, i)] >= 0, f"self_loop_label_{l}_in_{i}"

    def __call__(self, node: "SearchNode") -> float:
        self.prob.setObjective(lpSum(self.factor_state_to_goal_cost[(i, s)] for i, s in enumerate(node.state.states)))

        # for v in prob.variables():
        #     if v.lowBound is None:
        #         v.lowBound = -1e9
        #     if v.upBound is None:
        #         v.upBound = 1e9
        self.solver.solve(self.prob)
        # for v in self.prob.variables():
        #     print(v.name, v.varValue)

        if self.prob.status == LpStatusUnbounded:
            return float("inf")
        elif self.prob.status != LpStatusOptimal:
            logging.error(f"LP is {LpStatus[self.prob.status]}")
            raise ValueError("LP was not solved")
        return math.ceil(self.prob.objective.value() - 0.001)


@cli_register("huocp")
class UpfrontOptimalCostPartitioningHeuristic(Heuristic):
    def __init__(self, task: FactoredTask):
        super().__init__(task)
        self.task = task

        self.factor_state_to_goal_cost = {}
        self._initialize_lp()

    def _initialize_lp(self):
        """
        Compute the optimal cost partitioning that maximizes the sum of goal costs over all states in all factors.
        We use a linear programming approach to solve this problem.

        We have one variable per label per factor, representing the cost assigned to that label in that factor.
        We have one constraint per label, representing that the sum of the costs assigned to that label across all factors
        must be less than or equal to the original cost of that label.
        For each factor, we have one variable per state, representing the goal cost of that state in that factor.
        We have one constraint per transition s -l-> s', which is goal_cost(s) <= goal_cost(s') + cost_in_factor(l).
        We have one constraint per goal state s, which is goal_cost(s) = 0.
        """
        self.prob = LpProblem("UpfrontOptimalCostPartitioning", LpMaximize)
        # self.solver = CPLEX_CMD(path="/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex", msg=False)
        self.solver = pulp.PULP_CBC_CMD(msg=False)

        label_factor_to_cost_var = {(l, i): LpVariable(f"{l}_in_{i}") for i in range(self.task.size()) for l in self.task.labels}
        for l in self.task.labels:
            self.prob += lpSum(label_factor_to_cost_var[(l, i)] for i in range(self.task.size())) <= self.task.label_costs[l], f"cost of label {l}"

        for i, factor in enumerate(self.task.factors):
            self.factor_state_to_goal_cost.update({(i, s): LpVariable(f"goal_cost({i},{s})") for s in factor.states})
            for s in factor.goal_states:
                self.prob += self.factor_state_to_goal_cost[(i, s)] == 0, f"goal_state_{s}_in_{i}"
            self_loop_labels = set()
            for s, l, t in factor.transitions:
                if s == t:
                    self_loop_labels.add(l)
                else:
                    self.prob += (
                        self.factor_state_to_goal_cost[(i, s)] <= self.factor_state_to_goal_cost[(i, t)] + label_factor_to_cost_var[(l, i)],
                        f"transition_{s}_{l}_{t}_in_{i}",
                    )

            for l in self_loop_labels:
                self.prob += label_factor_to_cost_var[(l, i)] >= 0, f"self_loop_label_{l}_in_{i}"

        self.prob.setObjective(lpSum(v for _, v in self.factor_state_to_goal_cost.items()))
        self.solver.solve(self.prob)

        # Extract solution and store goal costs for each factor
        self.factor_goal_costs = [{s: self.factor_state_to_goal_cost[(i, s)].varValue for s in factor.states} for i, factor in enumerate(self.task.factors)]

    def __call__(self, node: "SearchNode") -> float:
        return math.ceil(sum(self.factor_goal_costs[i][s] for i, s in enumerate(node.state.states)) - 0.001)
