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
Implements the A* (a-star) and weighted A* search algorithm.
"""

import heapq
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from pyperplan.heuristics.relaxation import (
    hAddHeuristic,
    hFFHeuristic,
    hMaxHeuristic,
    hSAHeuristic,
)
from pyperplan.search.searchspace import SearchNode
from pyperplan.task import Operator, Task

from . import searchspace
from ..cli import cli_register
from ..heuristics.blind import BlindHeuristic
from ..heuristics.heuristic_base import Heuristic
from ..pruning.pruning import NonePruning, Pruning
from .search import Search
from .search_space_drawer import NoneSearchSpaceDrawer, SearchSpaceDrawer


class BestFirstSearch(Search):
    def __init__(
        self, task: Task, heuristic: type[Heuristic] = BlindHeuristic, pruning: type[Pruning] = NonePruning, search_space_drawer: type[SearchSpaceDrawer] = NoneSearchSpaceDrawer
    ):
        super().__init__(task)
        self.task = task
        self.heuristic = heuristic(task)
        self.pruning = pruning(task)
        self.search_space_drawer = search_space_drawer()
        self.use_relaxed_plan = isinstance(self.heuristic, hFFHeuristic)

    def _make_open_entry(self, node: SearchNode, h: int, node_tiebreaker: int) -> Tuple[int, ..., SearchNode]:
        """
        Creates an ordered search node (basically, a tuple containing the node
        itself and an ordering) for best first search.
        """
        raise NotImplementedError("Base class does not implement this method.")

    def search(self) -> Optional[List[Operator]]:
        """
        Searches for a plan in the given task using A* search.

        @param task The task to be solved
        @param heuristic  A heuristic callable which computes the estimated steps
                          from a search node to reach the goal.
        @param make_open_entry An optional parameter to change the bahavior of the
                               astar search. The callable should return a search
                               node, possible values are ordered_node_astar,
                               ordered_node_weighted_astar and
                               ordered_node_greedy_best_first with obvious
                               meanings.
        """
        open = []
        state_cost = {self.task.initial_state: 0}
        node_tiebreaker = 0

        root = searchspace.make_root_node(self.task.initial_state)
        init_h = self.heuristic(root)
        self.search_space_drawer.set_g_value(root, root.g)
        self.search_space_drawer.set_heuristic(root, init_h)
        if init_h != float("inf"):
            heapq.heappush(open, self._make_open_entry(root, init_h, node_tiebreaker))
        logging.info("Initial h value: %f" % init_h)

        besth = float("inf")
        counter = 0
        generated = 1
        expansions = 0
        evaluated = 1
        dead_ends = 0
        expansions_until_last_jump = 0
        highest_f = -1

        while open:
            (f, h, _tie, pop_node) = heapq.heappop(open)
            if h < besth:
                besth = h
                logging.debug("Found new best h: %d after %d expansions" % (besth, counter))
            if f > highest_f:
                highest_f = f
                expansions_until_last_jump = expansions
                logging.debug("f: %d (%d expansions, %d generated)" % (highest_f, counter, generated))

            pop_state = pop_node.state
            # Only expand the node if its associated cost (g value) is the lowest
            # cost known for this state. Otherwise we already found a cheaper
            # path after creating this node and hence can disregard it.
            if state_cost[pop_state] == pop_node.g:
                expansions += 1

                if self.task.goal_reached(pop_state):
                    logging.info("Goal reached. Start extraction of solution.")
                    logging.info("%d Nodes expanded" % expansions)
                    logging.info(f"Evaluated {evaluated} state(s).")
                    logging.info(f"Expanded {expansions} state(s).")
                    logging.info(f"Expanded until last jump: {expansions_until_last_jump} state(s).")
                    logging.info(f"Dead ends: {dead_ends} state(s).")
                    logging.info(f"Generated {generated} state(s).")
                    self.search_space_drawer.set_goal(pop_node)
                    self.search_space_drawer.draw()
                    return pop_node.extract_solution()
                rplan = None
                if self.use_relaxed_plan:
                    (rh, rplan) = self.heuristic.calc_h_with_plan(searchspace.make_root_node(pop_state))
                    logging.debug("relaxed plan %s " % rplan)

                for op, succ_state in self.task.get_successor_states(pop_state):
                    if self.use_relaxed_plan:
                        if rplan and not op.name in rplan:
                            # ignore this operator if we use the relaxed plan
                            # criterion
                            logging.debug("removing operator %s << not a " "preferred operator" % op.name)
                            continue
                        else:
                            logging.debug("keeping operator %s" % op.name)

                    succ_node = searchspace.make_child_node(pop_node, op, succ_state, self.task)

                    self.search_space_drawer.set_successors(pop_node, op, succ_node)
                    self.search_space_drawer.set_g_value(succ_node, succ_node.g)

                    if self.pruning.prune(succ_node):
                        continue

                    h = self.heuristic(succ_node)
                    evaluated += 1
                    self.search_space_drawer.set_heuristic(succ_node, h)
                    # logging.debug(f'h({succ_state}) = {h}')

                    if h == float("inf"):
                        # don't bother with states that can't reach the goal anyway
                        dead_ends += 1
                        continue
                    old_succ_g = state_cost.get(succ_state, float("inf"))
                    if succ_node.g < old_succ_g:
                        # We either never saw succ_state before, or we found a
                        # cheaper path to succ_state than previously.
                        node_tiebreaker += 1
                        heapq.heappush(open, self._make_open_entry(succ_node, h, node_tiebreaker))
                        state_cost[succ_state] = succ_node.g
                        generated += 1

            counter += 1
        logging.info("No operators left. Task unsolvable.")
        logging.info("%d Nodes expanded" % expansions)
        logging.info(f"Evaluated {evaluated} state(s).")
        logging.info(f"Expanded {expansions} state(s).")
        logging.info(f"Expanded until last jump: {expansions_until_last_jump} state(s).")
        logging.info(f"Dead ends: {dead_ends} state(s).")
        logging.info(f"Generated {generated} state(s).")

        self.search_space_drawer.draw()
        return None


@cli_register("astar")
class AStarSearch(BestFirstSearch):
    def _make_open_entry(self, node: SearchNode, h: int, node_tiebreaker: int) -> Tuple[int, int, int, SearchNode]:
        """
        Creates an ordered search node (basically, a tuple containing the node
        itself and an ordering) for A* search.

        @param node The node itself.
        @param heuristic A heuristic function to be applied.
        @param node_tiebreaker An increasing value to prefer the value first
                               inserted if the ordering is the same.
        @returns A tuple to be inserted into priority queues.
        """
        f = node.g + h
        return (f, h, node_tiebreaker, node)


@cli_register("wastar")
class WeightedAStarSearch(BestFirstSearch):
    def __init__(self, task: Task, weight: int = 5, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        self.weight = weight

    def _make_open_entry(self, node: SearchNode, h: int, node_tiebreaker: int) -> Tuple[int, int, int, SearchNode]:
        """
        Creates an ordered search node (basically, a tuple containing the node
        itself and an ordering) for weighted A* search (order: g+weight*h).

        @param weight The weight to be used for h
        @param node The node itself
        @param h The heuristic value
        @param node_tiebreaker An increasing value to prefer the value first
                               inserted if the ordering is the same
        @returns A tuple to be inserted into priority queues
        """
        return (
            node.g + self.weight * h,
            h,
            node_tiebreaker,
            node,
        )


@cli_register("gbfs")
class GreedyBestFirstSearch(BestFirstSearch):
    def _make_open_entry(self, node: SearchNode, h: int, node_tiebreaker: int) -> Tuple[int, int, int, SearchNode]:
        """
        Creates an ordered search node (basically, a tuple containing the node
        itself and an ordering) for greedy best first search (the value with lowest
        heuristic value is used).

        @param node The node itself.
        @param h The heuristic value.
        @param node_tiebreaker An increasing value to prefer the value first
                               inserted if the ordering is the same.
        @returns A tuple to be inserted into priority queues.
        """
        f = h
        return (f, h, node_tiebreaker, node)
