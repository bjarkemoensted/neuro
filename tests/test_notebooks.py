from __future__ import annotations

import json
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
        self.notebooks = tuple(sorted(helpers.NOTEBOOK_DIR.rglob("*.ipynb"), reverse=True))
        return super().setUp()

    def _check_types(self, notebook: Path, issue_extractor: Callable[[Path], Iterable[Issue]]) -> None:
        with NotebookWrapper(notebook) as ctx:
            issues = list(issue_extractor(ctx.script_path))
            msg = ctx.format_multiple_issues(*issues)
            self.assertEqual(len(issues), 0, msg)

    def test_pyright_checks(self) -> None:
        for notebook in self.notebooks:
            with self.subTest(notebook.name):
                self._check_types(
                    notebook,
                    issue_extractor=type_checking.iterate_pyright_issues
                )

    def test_mypy_checks(self) -> None:
        for notebook in self.notebooks:
            with self.subTest(notebook.name):
                self._check_types(
                    notebook,
                    issue_extractor=type_checking.iterate_mypy_issues
                )

    def test_cell_ids(self) -> None:
        """Check that cell IDs are consecutive, starting at zero (using an LLM to initialize notebooks from PDF,
        and they sometimes enter random strings as IDs for some reason)."""

        for notebook in self.notebooks:
            with self.subTest(msg=notebook.name):
                contents = json.loads(notebook.read_text())
                cells = contents["cells"]
                ids = [cell["id"] for cell in cells]
                expected = list(map(str, range(len(cells))))
                self.assertListEqual(ids, expected)

    def test_consistent_metadata(self) -> None:
            """Check that the notebooks have identical metadata."""

            target: dict[str, object]|None = None

            # Start with week1 (ordered) to make that the target value
            for notebook in sorted(self.notebooks):
                meta = json.loads(notebook.read_text())["metadata"]
                if target is None:
                    target = meta

                assert target is not None
                with self.subTest(notebook.name):
                    self.assertDictEqual(meta, target)
