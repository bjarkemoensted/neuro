import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .cli_utils import (
    run_mypy,
    run_pyright,
)


@dataclass
class Issue:
    """This is just to summarize information about some type checking/linting issue
    in a generalized format"""

    path: Path
    line: int
    kind: str  # the kind of issue (error, warning)
    description: str|None=None

    @property
    def line_zero_indexed(self) -> int:
        res = self.line - 1
        if res < 0:
            raise ValueError
        return res


def iterate_pyright_issues(path: Path) -> Iterator[Issue]:
    # Matches stuff like '/path/to/file.py:209:1 - warning: description'
    PATTERN = r"\s*?(\S+?):(\d+):\d+ - (\w+): (.+)"
    
    result = run_pyright(path)
    output = "\n\n".join((result.stdout, result.stderr))
    hits = re.findall(PATTERN, output)
    for hit in hits:
        path, line, kind, description = hit
        issue = Issue(
            path=Path(path),
            line = int(line),
            kind=kind,
            description=description
        )

        yield issue


def iterate_mypy_issues(path: Path) -> Iterator[Issue]:
    # Matches stuff like '/path/to/file.py:209:1 - warning: description'
    PATTERN = r"\s*?(\S+?):(\d+): (\w+): (.+)"
    
    result = run_mypy(path)
    output = "\n\n".join((result.stdout, result.stderr))
    hits = re.findall(PATTERN, output)
    for hit in hits:
        path, line, kind, description = hit
        issue = Issue(
            path=Path(path),
            line = int(line),
            kind=kind,
            description=description
        )

        yield issue
