from abc import ABC
from pathlib import Path
from typing import Any, Callable

from pyperplan.cli import cli_register
from pyperplan.search.searchspace import SearchNode


class SearchSpaceDrawer(ABC):
    """
    A class to visualize the search space of a planning problem. This class should be called when evaluating nodes
    and when generating successors.
    """

    def __init__(self):
        pass

    def set_successors(self, node: SearchNode, label: str, successor: SearchNode):
        raise NotImplementedError("Base class does not implement this method.")

    def set_attribute(self, attribute: str, node: SearchNode, value: Any, aggregator: Callable = min):
        raise NotImplementedError("Base class does not implement this method.")

    def set_heuristic(self, node: SearchNode, h: int):
        return self.set_attribute("h", node, h)

    def set_g_value(self, node: SearchNode, g: int):
        return self.set_attribute("g", node, g)

    def set_goal(self, node: SearchNode):
        raise NotImplementedError("Base class does not implement this method.")

    def draw(self):
        raise NotImplementedError("Base class does not implement this method.")


@cli_register()
class NoneSearchSpaceDrawer(SearchSpaceDrawer):
    """
    A class that does not draw anything.
    """

    def __init__(self):
        super().__init__()

    def set_successors(self, node: SearchNode, label: str, successor: SearchNode):
        pass

    def set_attribute(self, attribute: str, node: SearchNode, value: Any, aggregator: Callable = min):
        pass

    def set_goal(self, node: SearchNode):
        pass

    def draw(self):
        pass


@cli_register()
class GraphSearchSpaceDrawer(SearchSpaceDrawer):
    """
    A class to visualize the search space of a planning problem using graphviz. This class draws the search space in
    terms of states, and thus it is not a tree.
    """

    def __init__(self):
        super().__init__()
        self.state_map: dict[Any, int] = {}
        self.successors_map: dict[Any, list[tuple[str, Any]]] = {}
        self.attributes: dict[Any, dict[str, Any]] = {}
        self.goals = set()

    def _ensure_known_state(self, state: Any):
        if state not in self.state_map:
            self.state_map[state] = len(self.state_map)
        if state not in self.successors_map:
            self.successors_map[state] = []
        if state not in self.attributes:
            self.attributes[state] = {}

    def set_successors(self, node: SearchNode, label: str, successor: SearchNode):
        state = node.state
        self._ensure_known_state(state)

        self.successors_map[state].append((label, successor.state))
        self._ensure_known_state(successor.state)

    def set_attribute(self, attribute: str, node: SearchNode, value: Any, aggregator: Callable = min):
        state = node.state
        self._ensure_known_state(state)

        if attribute in self.attributes[state]:
            self.attributes[state][attribute] = aggregator(self.attributes[state][attribute], value)
        else:
            self.attributes[state][attribute] = value

    def set_goal(self, node: SearchNode):
        state = node.state
        self._ensure_known_state(state)

        self.goals.add(state)

    def draw(self, filename: str = "search_space"):
        lines = ["digraph G {", "  rankdir=TB;"]
        for state, state_id in self.state_map.items():
            label = ", ".join(f"{a}={v}" for a, v in self.attributes[state].items())
            lines.append(f'  {state_id} [label="{state}", xlabel="{label}", peripheries={"2" if state in self.goals else "1"}, shape="box"];')

            aggregated_edges = {}
            for label, succ in self.successors_map.get(state, []):
                if (state, succ) not in aggregated_edges:
                    aggregated_edges[(state, succ)] = []
                aggregated_edges[(state, succ)].append(label)

            for (src, tgt), labels in aggregated_edges.items():
                lines.append(f'  {self.state_map[src]} -> {self.state_map[tgt]} [label="{chr(10).join(labels)}"];')

        lines.append("}")

        Path(f"{filename}.dot").write_text("\n".join(lines))
