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
Implements the breadth first search algorithm.
"""

from collections import deque
import logging
from typing import List, Optional

from pyperplan.task import Operator, Task

from . import searchspace
from ..cli import cli_register
from .search import Search


@cli_register("bfs")
class BreadthFirstSearch(Search):
    def __init__(self, task: Task):
        super().__init__(task)
        self.task = task

    def search(self) -> Optional[List[str]]:
        """
        Searches for a plan on the given task using breadth first search and
        duplicate detection.

        @return: The solution as a list of operators or None if the task is
        unsolvable.
        """
        # counts the number of loops (only for printing)
        iteration = 0
        # fifo-queue storing the nodes which are next to explore
        queue = deque()
        queue.append(searchspace.make_root_node(self.task.initial_state))
        # set storing the explored nodes, used for duplicate detection
        closed = {self.task.initial_state}
        while queue:
            iteration += 1
            logging.debug("breadth_first_search: Iteration %d, #unexplored=%d" % (iteration, len(queue)))
            # get the next node to explore
            node = queue.popleft()
            # exploring the node or if it is a goal node extracting the plan
            if self.task.goal_reached(node.state):
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % iteration)
                return node.extract_solution()
            for operator, successor_state in self.task.get_successor_states(node.state):
                # duplicate detection
                if successor_state not in closed:
                    queue.append(searchspace.make_child_node(node, operator, successor_state, self.task))
                    # remember the successor state
                    closed.add(successor_state)
        logging.info("No operators left. Task unsolvable.")
        logging.info("%d Nodes expanded" % iteration)
        return None
