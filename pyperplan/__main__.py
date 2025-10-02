#! /usr/bin/env python3
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

# TODO: Give searches and heuristics commandline options and reenable preferred
# operators.

import argparse
import logging
import os
from pathlib import Path
import sys

# These imports look unused, but are not. Module discovery explores the modules, and translate adds a path to sys.path.
import pyperplan.cli # isort: skip
import pyperplan.module_discovery # isort: skip
import pyperplan.translate.translate # isort: skip

from pyperplan.cli import cli_constructor
from pyperplan.planner import (
    find_domain,
    search_plan,
    validate_solution,
    write_solution,
)
from pyperplan.search.search import Search
from pyperplan.task_transformation.task_transformation import TaskTransformation

def no_traceback_memoryerror(exc_type, exc_value, exc_tb):
    if exc_type is MemoryError:
        print("Memory limit reached.")
    else:
        sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = no_traceback_memoryerror


def main():
    # Commandline parsing
    log_levels = ["debug", "info", "warning", "error"]

    # get pretty print names for the search algorithms:
    # use the function/class name and strip off '_search'
    def get_callable_names(callables, omit_string):
        names = [c.__name__ for c in callables]
        names = [n.replace(omit_string, "").replace("_", " ") for n in names]
        return ", ".join(names)

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(dest="domain", nargs="?")
    argparser.add_argument(dest="problem")
    argparser.add_argument(
        "--plan-file",
        type=Path,
        help="File path for the plan",
        default=None,
    )
    argparser.add_argument("-l", "--loglevel", choices=log_levels, default="info")
    argparser.add_argument(
        "-s",
        "--search",
        type=cli_constructor(Search),
        help=f"",
        default="bfs",
    )
    argparser.add_argument(
        "-t",
        "--task-representation",
        choices=["strips", "factored"],
        help="Select the task representation to use",
        default="strips",
    )
    argparser.add_argument("-T", "--task-transformation", type=cli_constructor(TaskTransformation), help=f"Task transformation to apply before searching.", default=None)
    argparser.add_argument(
        "--draw-search-space",
        choices=["none", "graph"],
        default="none",
        help="Draw the search space to a file (search_space.dot)",
    )
    argparser.add_argument("--intersect-original-factor", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(relativeCreated)dms %(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    # hffpo_searches = ["gbf", "wastar", "ehs"]
    # if args.heuristic == "hffpo" and args.search not in hffpo_searches:
    #     print(
    #         "ERROR: hffpo can currently only be used with %s\n" % hffpo_searches,
    #         file=sys.stderr,
    #     )
    #     argparser.print_help()
    #     sys.exit(2)

    args.problem = os.path.abspath(args.problem)
    if args.problem.endswith(".sas"):
        pass
    elif args.domain is None:
        args.domain = find_domain(args.problem)
    else:
        args.domain = os.path.abspath(args.domain)

    if args.search in ["bfs", "ids", "sat"]:
        heuristic = None

    # use_preferred_ops = args.heuristic == "hffpo"
    search = args.search
    delattr(args, "search")
    solution = search_plan(
        args.domain,
        args.problem,
        search,
        # use_preferred_ops=use_preferred_ops,
        **args.__dict__,
    )

    if solution is None:
        logging.warning("No solution could be found")
    else:
        logging.info("Plan length: %s" % len(solution))
        if args.plan_file:
            solution_file = args.plan_file
            write_solution(solution, solution_file)
            if ".sas" not in args.problem:
                validate_solution(args.domain, args.problem, solution_file)
            else:
                logging.info("Cannot validate plans for SAS files.")


if __name__ == "__main__":
    main()
