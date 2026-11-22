from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from .type_checking import Issue


class ActiveNotebookContext:
    """This represents a temporary context where a jupyter notebook and a corresponding
    temporary location for storing a converted python script are both available.
    After initialization, this class exposes
    1) The path of the jupyter notebook
    2) The path of a temporary python script with the same code
    3) Helper functionality for mapping line numbers in the script back to cell IDs
    """

    # For pointing out where an issue is located
    POINTER = "*HERE* --> "

    def __init__(self, wrapper: NotebookWrapper) -> None:
        self._stack = contextlib.ExitStack()
        self.wrapper = wrapper
        self.notebook_path = wrapper.path
        self.json: dict[str, object] = json.loads(self.notebook_path.read_text())
        self.tmp = self._stack.enter_context(tempfile.NamedTemporaryFile(suffix=".py"))  # noqa: SIM115
        self.script_path = Path(self.tmp.file.name)

        # Store the lines of python code, and the corresponding cell IDs
        cell_ids_and_lines = [(id_, line) for id_, _, line in self.iter_cell_row_line()]

        # Quickfix for if cell list is empty (zero cells), which retains the type hinting as tuples of strings for each
        _empty: list[tuple[str, ...]] = [(), ()]
        a, b = zip(*cell_ids_and_lines, strict=True) if cell_ids_and_lines else _empty
        self.cell_ids = a
        self.lines = b

        self.pure_python_lines = tuple(self.preprocess_python_code(line) for line in self.lines)
        self.pure_python = "".join(self.pure_python_lines)
        self.script_path.write_text(self.pure_python)

    def preprocess_python_code(self, line: str) -> str:
        """Preprocess a line to prepare it for writing as a pure python file, where we can run
        type checkers etc"""

        # Comment out jupyter 'magic comments'
        if line.strip().startswith(r"%"):
            line = f"#  {line}"

        # Ensure the line ends with a newline character
        if not line.endswith("\n"):
            line += "\n"

        return line

    def iter_cell_row_line(self) -> Iterator[tuple[str, int, str]]:
        """Iterates over tuples of cell id, line number, line
        for all code cells"""

        line_number = 0
        cells = self.json["cells"]
        assert isinstance(cells, list)
        for cell in cells:
            # Skip non-code cells
            if cell["cell_type"] != "code":
                continue
            id_ = cell["id"]
            for line in cell["source"]:
                assert isinstance(line, str)
                yield id_, line_number, line
                line_number += 1

    def teardown(self) -> None:
        self._stack.close()

    def _scan_to_cell_end(self, line: int, n_max: int, step: int) -> int:
        """Keep taking step up to n_max times or until the cell changes"""
        running = line
        cell_id = self.cell_ids[line]
        for _ in range(n_max):
            next_ = running + step
            try:
                cell_changed = self.cell_ids[next_] != cell_id
            except IndexError:
                cell_changed = True

            if cell_changed:
                break
            else:
                running = next_

        return running

    def _determine_vicinity(self, target_line: int) -> tuple[int, int]:
        """Takes a line number. Returns the earliest and latest line to include when displaying
        the vicinity of the target line (up to n lines before and after, stopping when the cell ends)"""
        
        n = self.wrapper.n_lines_adjacent_display
        low = self._scan_to_cell_end(target_line, n, step=-1)
        high = self._scan_to_cell_end(target_line, n, step=+1)
        return low, high

    def _format_issue_header(self, issue: Issue) -> str:
        cell_id = self.cell_ids[issue.line_zero_indexed]
        s = f"* {issue.kind} @ cell {cell_id} - line {issue.line_zero_indexed} - {issue.description}*"
        return s

    def _format_issue_details(self, issue: Issue) -> str:
        pad = len(self.POINTER)*" "
        low, high = self._determine_vicinity(issue.line_zero_indexed)
        formatted = ((self.POINTER if i == issue.line_zero_indexed else pad)+self.lines[i] for i in range(low, high+1))
        res = "".join(formatted)
        return res

    def format_issue(self, issue: Issue) -> str:
        if issue.path.resolve() != self.script_path.resolve():
            raise RuntimeError("Wrong file")

        parts = (
            self._format_issue_header(issue),
            self._format_issue_details(issue),
        )

        res = "\n".join(parts)
        return res

    def format_multiple_issues(self, *issues: Issue) -> str:
        issue_str = "\n\n".join(map(self.format_issue, issues))
        summary = f"Total: {len(issues)} issues."
        res = "\n\n".join((issue_str, summary))
        return res


class NotebookWrapper:
    """Wrapper for a jupyter notebook, to simplify the process of temporarily converting into
    a pure python file to run type checkers etc"""

    def __init__(self, path: Path, n_lines_adjacent_display: int=10) -> None:
        """Initialize a notebook wrapper.
        path: the pathlib.Path to the jupyter notebook
        n_lines_adjacent_display (int) - when displaying an issue, the number of lines
            above and below the issue to display"""

        self.path = path
        self.n_lines_adjacent_display = n_lines_adjacent_display
        self._active_context: ActiveNotebookContext|None = None

    def __enter__(self) -> ActiveNotebookContext:
        if self._active_context is not None:
            raise RuntimeError("Context is already active!")
        self._active_context = ActiveNotebookContext(self)
        return self._active_context

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._active_context is not None:
            self._active_context.teardown()
        else:
            raise RuntimeError(f"{self.__class__.__name__} attempted teardown with no active context")

        if self._active_context.script_path.exists():
            raise RuntimeError("Temporary python file survived cleanup")
        self._active_context = None
