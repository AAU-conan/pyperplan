from pyperplan.pruning.dominance_pruning import DominancePruning


class TwoNonDominatedLabelDominancePruning(DominancePruning):
    """
    Like dominance pruning, but the label relation keeps track of up to two non-dominating factors.
    """

    def __init__(self, task):
        super().__init__(task)
        self.label_relation: dict[str, set[tuple[int, int] | int]]

    def _set_label_not_dominates_in(self, i: int, l1: str, l2: str) -> bool:
        """
        Sets the label relation for l1 and l2 in factor i to not dominate.
        """
        if (l1, l2) in self.label_relation:
            f = self.label_relation[(l1, l2)]
            if f == DOMINATES_IN_ALL:
                self.label_relation[(l1, l2)] = (i, -1)
                return True
            elif f[0] == i or f[1] == i:
                # If it was already set to not dominate in this factor, do nothing
                return False
            elif f[1] == -1:
                self.label_relation[(l1, l2)] = (f[0], i)
                return True
            else:
                self.label_relation.pop((l1, l2))
                return True
        return False

    def _dominates_in_all_other_factors(self, i: int, l1: str, l2: str) -> bool:
        """
        Returns if label l2 dominates l1 in all factors except factor i.
        """
        f = self.label_relation.get((l1, l2), DOMINATES_IN_NONE)
        return f != DOMINATES_IN_NONE and (f == DOMINATES_IN_ALL or (f[0] == i and f[1] == -1))

    def label_dominates_label_in_factor(self, i: int, l1: str, l2: str):
        """
        Lookup if label l2 dominates label l1 in factor i.
        """
        r = self.label_relation.get((l1, l2), DOMINATES_IN_NONE)
        return r != DOMINATES_IN_NONE and (r == DOMINATES_IN_ALL or r[0] != i)
