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

from pyperplan.search.searchspace import SearchNode
from pyperplan.task import Task

from ..cli import cli_register
from .heuristic_base import Heuristic


@cli_register("hblind")
class BlindHeuristic(Heuristic):
    """
    Implements a simple blind heuristic for convenience.
    It returns 0 if the goal was reached and 1 otherwise.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.task = task

    def __call__(self, node: SearchNode):
        return 0 if self.task.goal_reached(node.state) else 1
