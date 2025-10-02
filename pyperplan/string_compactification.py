class StringCompactifier:
    def __init__(self, universe: set[str], token_delimeter: str = " "):
        """
        This class compactifies a subset of strings from a given universe to a subset of strings with symbols indicating
        wildcard or disjunction. For example, if the universe is {'a a', 'a b', 'a c', 'b a', 'b b', 'b c'}, then the set
        {'a a', 'a b', 'a c'} can be compactified to {'a *'} and the set {'b a', 'b c'} can be compactified to {'b a|c'}.
        """
        self.token_delimeter = token_delimeter
        self.universe = {tuple(s.split(token_delimeter)) for s in universe}

    def compactify(self, strings: set[str]) -> set[str]:
        """
        Compactifies the given set of strings. Only merge strings of the same length.

        First we tokenize all of the strings. Then, we iteratively merge strings that differ in exactly one token by
        replacing the differing token with the options, i.e. we merge 'a b' and 'a c' to 'a b|c'. We can also merge e.g.
        'a a|c' and 'b a|c' to '* a|c'. We continue this until no more merges are possible. Finally, we check that if
        all the options are listed in | we replace them with *.
        """
        token_strings = [tuple(s.split(self.token_delimeter)) for s in strings]
        assert set(token_strings) <= self.universe, "The given strings must be a subset of the universe."

        len_token_strings: dict[int, list[tuple[str, ...]]] = {}
        for ts in token_strings:
            if len(ts) not in len_token_strings:
                len_token_strings[len(ts)] = []
            len_token_strings[len(ts)].append(ts)

        result = set()
        for l, token_strings_l in sorted(len_token_strings.items()):
            changed = True
            while changed:
                changed = False
                for ts in token_strings_l:
                    merge_index = -1
                    # Try to merge ts with another string
                    for other in token_strings_l:
                        if ts == other:
                            continue

                        # Check if they differ in exactly one token
                        diff_indices = [i for i in range(l) if ts[i] != other[i]]
                        if len(diff_indices) == 1:
                            merge_index = diff_indices[0]
                            break
                    if merge_index != -1:
                        # Merge all string that differ from ts in index merge_index
                        options = []
                        for ots in token_strings_l.copy():
                            if all(ots[i] == ts[i] for i in range(l) if i != merge_index):
                                token_strings_l.remove(ots)
                                options.append(ots[merge_index])
                        token_strings_l.append(ts[:merge_index] + ("|".join(sorted(options)),) + ts[merge_index + 1 :])
                        changed = True
                        break

            result.update(self.token_delimeter.join(ts) for ts in token_strings_l)
        return result


def test_simple():
    sc = StringCompactifier({"a a", "a b", "a c", "b a", "b b", "b c"})
    # Tests that shouldn't merge anything
    assert sc.compactify({"a a"}) == {"a a"}
    assert sc.compactify({"a a", "b b"}) == {"a a", "b b"}

    # Tests that should only merge to | cases
    assert sc.compactify({"a a", "a b"}) == {"a a|b"}
    assert sc.compactify({"a a", "a b", "a c"}) == {"a a|b|c"}
    assert sc.compactify({"b a", "b c"}) == {"b a|c"}
    assert sc.compactify({"a a", "a c", "b b"}) == {"a a|c", "b b"}
