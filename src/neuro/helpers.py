import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_here = Path(__file__).parent

PROJECT_ROOT = _here.parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def get_notebook_paths(newest_first: bool=True) -> tuple[Path, ...]:
    """Returns a tuple of paths for all notebooks, """
    res = tuple(sorted(NOTEBOOK_DIR.rglob("*.ipynb"), reverse=newest_first))
    return res


def iter_notebooks_json(newest_first: bool=True) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Iterate over pairs of notebook paths and their content as dicts"""
    for p in get_notebook_paths(newest_first=newest_first):
        d: dict[str, object] = json.loads(p.read_text())
        yield p, d
