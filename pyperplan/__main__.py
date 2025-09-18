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
import sys
from pathlib import Path

from pyperplan.planner import (
    find_domain,
    HEURISTICS,
    search_plan,
    SEARCHES,
    validate_solution,
    write_solution, PRUNING,
)
from pyperplan.search.search_space_drawer import SearchSpaceDrawer, NoneSearchSpaceDrawer

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

    search_names = get_callable_names(SEARCHES.values(), "_search")

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(dest="domain", nargs="?")
    argparser.add_argument(dest="problem")
    argparser.add_argument('--plan-file',
        type=Path,
        help="File path for the plan",
        default=None,
    )
    argparser.add_argument("-l", "--loglevel", choices=log_levels, default="info")
    argparser.add_argument(
        "-H",
        "--heuristic",
        choices=HEURISTICS.keys(),
        help="Select a heuristic",
        default="hff",
    )
    argparser.add_argument(
        "-p",
        "--pruning",
        choices=PRUNING.keys(),
        help="Select a pruning method",
        default="none",
    )
    argparser.add_argument(
        "-s",
        "--search",
        choices=SEARCHES.keys(),
        help=f"Select a search algorithm from {search_names}",
        default="bfs",
    )
    argparser.add_argument(
        "-t",
        "--task-representation",
        choices=["strips", "factored", "qdom"],
        help="Select the task representation to use",
        default="strips",
    )
    argparser.add_argument(
        "--qdom",
        "--qualified-dominance",
        choices=["none", "qdom", "iqdom"],
        default="none",
        help="Use the qualified dominance heuristic (use -H <heuristic> for the base heuristic)",
    )
    argparser.add_argument("--qdom-approx", action="store_true",
        help="Approximate qualified dominance so that it is deterministic",
    )
    argparser.add_argument("--qdom-compare",
        choices=["all", "parent"],
        default="all",
        help="Which strategy to use for comparing nodes in qualified dominance",
    )
    argparser.add_argument(
        "--draw-search-space",
        choices=['none', 'graph'],
        default='none',
        help="Draw the search space to a file (search_space.dot)",
    )
    argparser.add_argument("--intersect-original-factor", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(relativeCreated)dms %(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    hffpo_searches = ["gbf", "wastar", "ehs"]
    if args.heuristic == "hffpo" and args.search not in hffpo_searches:
        print(
            "ERROR: hffpo can currently only be used with %s\n" % hffpo_searches,
            file=sys.stderr,
        )
        argparser.print_help()
        sys.exit(2)

    args.problem = os.path.abspath(args.problem)
    if args.problem.endswith('.sas'):
        pass
    elif args.domain is None:
        args.domain = find_domain(args.problem)
    else:
        args.domain = os.path.abspath(args.domain)

    search = SEARCHES[args.search]
    heuristic = HEURISTICS[args.heuristic]
    pruning = PRUNING[args.pruning]

    if args.qdom_compare == "all":
        from pyperplan.heuristics.qualified_dominance_heuristic import AllComparisonStrategy
        args.qdom_compare = AllComparisonStrategy
    elif args.qdom_compare == "parent":
        from pyperplan.heuristics.qualified_dominance_heuristic import ParentComparisonStrategy
        args.qdom_compare = ParentComparisonStrategy
    else:
        raise ValueError(f"Unknown qdom comparison strategy: {args.qdom_compare}")

    if args.search in ["bfs", "ids", "sat"]:
        heuristic = None

    if args.draw_search_space == 'none':
        search_space_drawer = NoneSearchSpaceDrawer()
    elif args.draw_search_space == 'graph':
        from pyperplan.search.search_space_drawer import GraphSearchSpaceDrawer
        search_space_drawer = GraphSearchSpaceDrawer()
    else:
        raise ValueError(f"Unknown search space drawer: {args.draw_search_space}")


    logging.info("using search: %s" % search.__name__)
    logging.info("using heuristic: %s" % (heuristic.__name__ if heuristic else None))
    use_preferred_ops = args.heuristic == "hffpo"
    delattr(args, 'search')
    delattr(args, 'heuristic')
    delattr(args, 'pruning')
    solution = search_plan(
        args.domain,
        args.problem,
        search,
        heuristic,
        pruning,
        use_preferred_ops=use_preferred_ops,
        search_space_drawer=search_space_drawer,
        **args.__dict__
    )

    if solution is None:
        logging.warning("No solution could be found")
    else:
        logging.info("Plan length: %s" % len(solution))
        if args.plan_file:
            solution_file = args.plan_file
            write_solution(solution, solution_file)
            validate_solution(args.domain, args.problem, solution_file)


if __name__ == "__main__":
    main()
