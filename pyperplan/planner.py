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

import importlib
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, List, Optional, Type, Union

from pyperplan.pddl.pddl import Problem
from pyperplan.task import FactoredTask, Operator, Task

from . import grounding, heuristics, search, tools
from .pddl.parser import Parser
from .search.search import Search
from .task_transformation.task_transformation import TaskTransformation
from .translate.pddl_parser import open as open_pddl
from .translate.sas_tasks import open_sas_task, SASTask
from .translate.translate import pddl_to_sas


NUMBER = re.compile(r"\d+")


def get_heuristics() -> List[Any]:
    """
    Scan all python modules in the "heuristics" directory for classes ending
    with "Heuristic".
    """
    heuristics = []
    src_dir = os.path.dirname(os.path.abspath(__file__))
    heuristics_dir = os.path.abspath(os.path.join(src_dir, "heuristics"))
    for filename in os.listdir(heuristics_dir):
        if not filename.endswith(".py"):
            continue
        name = "." + os.path.splitext(os.path.basename(filename))[0]
        module = importlib.import_module(name, package="pyperplan.heuristics")
        heuristics.extend([getattr(module, cls) for cls in dir(module) if cls.endswith("Heuristic") and cls != "Heuristic" and not cls.startswith("_")])
    return heuristics


def _get_heuristic_name(cls: Any) -> str:
    name = cls.__name__
    assert name.endswith("Heuristic")
    return name[:-9].lower()


HEURISTICS = {_get_heuristic_name(heur): heur for heur in get_heuristics()}


def get_pruning_methods() -> List[Any]:
    """
    Scan all python modules in the "pruning" directory for classes ending
    with "Pruning".
    """
    pruning_methods = []
    src_dir = os.path.dirname(os.path.abspath(__file__))
    pruning_dir = os.path.abspath(os.path.join(src_dir, "pruning"))
    for filename in os.listdir(pruning_dir):
        if not filename.endswith(".py"):
            continue
        name = "." + os.path.splitext(os.path.basename(filename))[0]
        module = importlib.import_module(name, package="pyperplan.pruning")
        pruning_methods.extend([getattr(module, cls) for cls in dir(module) if cls.endswith("Pruning") and cls != "Pruning" and not cls.startswith("_")])
    return pruning_methods


def _get_pruning_name(cls: Any) -> str:
    name = cls.__name__
    assert name.endswith("Pruning")
    return name[:-7].lower()


PRUNING = {_get_pruning_name(p): p for p in get_pruning_methods()}


def validator_available() -> bool:
    return tools.command_available(["validate", "-h"])


def find_domain(problem: str) -> str:
    """
    This function tries to guess a domain file from a given problem file.
    It first uses a file called "domain.pddl" in the same directory as
    the problem file. If the problem file's name contains digits, the first
    group of digits is interpreted as a number and the directory is searched
    for a file that contains both, the word "domain" and the number.
    This is conforming to some domains where there is a special domain file
    for each problem, e.g. the airport domain.

    @param problem    The pathname to a problem file
    @return A valid name of a domain
    """
    dir, name = os.path.split(problem)
    number_match = NUMBER.search(name)
    number = number_match.group(0)
    domain = os.path.join(dir, "domain.pddl")
    for file in os.listdir(dir):
        if "domain" in file and number in file:
            domain = os.path.join(dir, file)
            break
    if not os.path.isfile(domain):
        logging.error(f'Domain file "{domain}" can not be found')
        sys.exit(1)
    logging.info(f"Found domain {domain}")
    return domain


def _parse(domain_file: str, problem_file: str) -> Problem:
    # Parsing
    parser = Parser(domain_file, problem_file)
    logging.info(f"Parsing Domain {domain_file}")
    domain = parser.parse_domain()
    logging.info(f"Parsing Problem {problem_file}")
    problem = parser.parse_problem(domain)
    logging.debug(domain)
    logging.info("{} Predicates parsed".format(len(domain.predicates)))
    logging.info("{} Actions parsed".format(len(domain.actions)))
    logging.info("{} Objects parsed".format(len(problem.objects)))
    logging.info("{} Constants parsed".format(len(domain.constants)))
    return problem


def _ground(problem: Problem, remove_statics_from_initial_state: bool = True, remove_irrelevant_operators: bool = True) -> Task:
    logging.info(f"Grounding start: {problem.name}")
    task = grounding.ground(problem, remove_statics_from_initial_state, remove_irrelevant_operators)
    logging.info(f"Grounding end: {problem.name}")
    logging.info("{} Variables created".format(len(task.facts)))
    logging.info("{} Operators created".format(len(task.operators)))
    return task


def write_solution(solution: List[Operator], filename: str):
    assert solution is not None
    with open(filename, "w") as file:
        for op in solution:
            print(op if isinstance(op, str) else op.name, file=file)


def search_plan(
    domain_file: str, problem_file: str, search: type[Search], task_representation: str = "strips", task_transformation: Optional[type[TaskTransformation]] = None, **kwargs
) -> List[Operator]:
    """
    Parses the given input files to a specific planner task and then tries to
    find a solution using the specified  search algorithm and heuristics.

    @param domain_file      The path to a domain file
    @param problem_file     The path to a problem file in the domain given by
                            domain_file
    @param search           A callable that performs a search on the task's
                            search space
    @param heuristic_class  A class implementing the heuristic_base.Heuristic
                            interface
    @return A list of actions that solve the problem
    """
    if task_representation == "strips":
        if not domain_file or not problem_file:
            raise ValueError("Domain and problem PDDL files must be specified for STRIPS representation")
        problem = _parse(domain_file, problem_file)
        task = _ground(problem)
    elif task_representation == "factored":
        if problem_file.endswith(".sas"):
            sas_task: SASTask = open_sas_task(problem_file)
            task_name = os.path.splitext(os.path.basename(problem_file))[0]
        else:
            pddl_task = open_pddl(domain_file, problem_file)
            sas_task: SASTask = pddl_to_sas(pddl_task)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                sas_task.output(Path("output.sas").open("w"))
            task_name = pddl_task.task_name
        task = FactoredTask.from_sas_task(task_name, sas_task)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            Path("task.dot").write_text(task.to_dot())
    else:
        raise ValueError(f"Unknown task representation: {task_representation}")
    logging.info("done reading input!")
    total_start_time = time.process_time()

    if task_transformation is not None:
        task = task_transformation().transform(task)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            Path("transformed_task.dot").write_text(task.to_dot())

    # if not heuristic_class is None:
    #     if qdom == 'none':
    #         heuristic = heuristic_class(task)
    #     elif qdom == 'qdom':
    #         heuristic = QualifiedDominanceHeuristic(task, heuristic_class,
    #                         intersect_original_factor=kwargs.get('intersect_original_factor', False),
    #                         approximate_to_deterministic=kwargs.get('qdom_approx', False),
    #                         comparison_strategy=kwargs.get('qdom_compare')
    #         )
    #     elif qdom == 'iqdom':
    #         heuristic = InterdimensionalQualifiedDominance(task, heuristic_class,
    #                         approximate_to_deterministic=kwargs.get('qdom_approx', False),
    #                        comparison_strategy=kwargs.get('qdom_compare')
    #        )
    #     else:
    #         assert False, f"Unknown qdom option: {qdom}"
    # pruning = pruning_class(task)
    search_start_time = time.process_time()
    # if use_preferred_ops and isinstance(heuristic, heuristics.hFFHeuristic):
    #     solution = _search(task, search, heuristic, pruning, search_space_drawer, use_preferred_ops=True)
    # else:
    #     solution = _search(task, search, heuristic, pruning, search_space_drawer)
    solution = search(task).search()
    cost = sum(task.get_action_cost(a) for a in solution) if solution else -1
    logging.info(f"Plan cost: {cost if cost >= 0 else float('inf')}")
    logging.info("Search time: {:.03}s".format(time.process_time() - search_start_time))
    logging.info("Total time: {:.03}s".format(time.process_time() - total_start_time))
    return solution


def validate_solution(domain_file: str, problem_file: str, solution_file: str):
    if not validator_available():
        logging.info("validate could not be found on the PATH so the plan can " "not be validated.")
        return

    cmd = ["validate", domain_file, problem_file, solution_file]
    exitcode = subprocess.call(cmd)

    if exitcode == 0:
        logging.info("Plan correct")
    else:
        logging.warning("Plan NOT correct")
    return exitcode == 0
