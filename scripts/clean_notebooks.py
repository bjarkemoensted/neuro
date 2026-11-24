import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from neuro import helpers

type Content = dict[str, Any]


@dataclass
class Result:
    d_original: Content = field(repr=False)
    d_final: Content = field(repr=False)
    ids_changed: int
    outputs_removed: int
    execution_counts_reset: int
    n_changes: int


def _fix_cell_ids(d: Content) -> int:
    cells: list[Content] = d["cells"]
    res = 0
    for i in range(len(cells)):
        expected = str(i)
        if cells[i]["id"] != expected:
            d["cells"][i]["id"] = expected
            res += 1
    return res


def _remove_outputs(d: Content) -> int:
    res = 0
    k = "outputs"
    for i, cell in enumerate(d["cells"]):
        if k in cell and cell[k] != []:
            d["cells"][i][k] = []
            res += 1
    return res


def _reset_execution_counts(d: Content) -> int:
    res = 0
    k = "execution_count"
    expected = None
    for i, cell in enumerate(d["cells"]):
        if k not in cell:
            continue
        if cell.get(k, expected) is not expected:
            d["cells"][i][k] = expected
            res += 1
    return res


# The functions which fix various issues, keyed by Result attribute
FIXERS: dict[str, Callable[[Content], int]] = {
    "ids_changed": _fix_cell_ids,
    "outputs_removed": _remove_outputs,
    "execution_counts_reset": _reset_execution_counts
}


def fix(d: Content) -> Result:
    d_original = deepcopy(d)
    d_final = deepcopy(d)

    outcomes = {k: fixer(d_final) for k, fixer in FIXERS.items()}

    res = Result(
        d_original=d_original,
        d_final=d_final,
        **outcomes,
        n_changes = sum(outcomes.values())
    )

    return res


def prompt_yes_no(prompt: str="", default: bool|None=None) -> bool:
    response: bool|None=None
    options = {"y": True, "n": False}
    while response is None:
        raw = input(prompt)
        processed = raw.strip().lower()
        if not processed and default is not None:
            response = default
        elif processed in options:
            response = options[processed]
        else:
            print(f"Invalid input: {raw!r}, try again")
    return response


def main() -> None:
    nb_to_results = {
        nb: fix(content)
        for nb, content in helpers.iter_notebooks_json(newest_first=False)
    }

    total_changes = sum(r.n_changes for r in nb_to_results.values())
    n_notebooks = len(nb_to_results)
    print(f"Did {total_changes} changes in {len(nb_to_results)} notebook{'s'*int(n_notebooks != 1)}.")
    
    if total_changes == 0:
        print("Nothing to do...")
        return

    for nb, r in nb_to_results.items():
        print(f"* {nb.name} - {r.n_changes} change{'s'*int(r.n_changes != 1)} *")
        if not r.n_changes:
            continue

        for a in FIXERS:
            print(f" - {a}: {getattr(r, a)}")
        print()

    change = {k: v for k, v in nb_to_results.items() if v.n_changes > 0}
    
    doit = prompt_yes_no(
        f"Update {len(change)} notebook{'s'*int(len(change) != 1)}? (Y/n)?",
        default=True
    )

    if not doit:
        return

    for nb, r in change.items():
        print(f"Writing to {nb}...")
        out = json.dumps(r.d_final)
        nb.write_text(out)

    print("Done!")


if __name__ == '__main__':
    main()
