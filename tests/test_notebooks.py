from __future__ import annotations

import unittest
from collections.abc import Callable, Iterable
from pathlib import Path

from neuro import helpers
from neuro.qa_utils import (
    Issue,
    NotebookWrapper,
    type_checking,
)


class TestNotebooks(unittest.TestCase):
    def setUp(self) -> None:
        self.notebooks = tuple(helpers.NOTEBOOK_DIR.rglob("*.ipynb"))
        return super().setUp()

    def _check_types(self, notebook: Path, issue_extractor: Callable[[Path], Iterable[Issue]]) -> None:
        with NotebookWrapper(notebook) as ctx:
            issues = list(issue_extractor(ctx.script_path))
            msg = ctx.format_multiple_issues(*issues)
            self.assertEqual(len(issues), 0, msg)

    def test_pyright_checks(self) -> None:
        for notebook in self.notebooks:
            self._check_types(
                notebook,
                issue_extractor=type_checking.iterate_pyright_issues
            )

    def test_mypy_checks(self) -> None:
        for notebook in self.notebooks:
            self._check_types(
                notebook,
                issue_extractor=type_checking.iterate_mypy_issues
            )
