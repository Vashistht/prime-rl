"""Build and execute the notebook from its readable percent-cell source."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "opd_toy_study.py"
NOTEBOOK = HERE / "opd_toy_study.ipynb"


def parse_percent_cells(source: str) -> list[dict[str, str]]:
    """Parse the small subset of Jupytext's percent format used here."""

    header = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")
    cells: list[dict[str, str]] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if kind is None:
            return
        body = "".join(lines)
        if kind == "markdown":
            body = "".join(
                line[2:] if line.startswith("# ") else ("\n" if line == "#\n" else line)
                for line in body.splitlines(keepends=True)
            )
        cells.append({"kind": kind, "source": body.rstrip()})
        lines = []

    for line in source.splitlines(keepends=True):
        match = header.match(line.rstrip("\n"))
        if match:
            flush()
            kind = "markdown" if match.group(1) else "code"
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def build_notebook() -> nbformat.NotebookNode:
    parsed = parse_percent_cells(SOURCE.read_text())
    cells = []
    for cell in parsed:
        constructor = nbformat.v4.new_markdown_cell if cell["kind"] == "markdown" else nbformat.v4.new_code_cell
        cells.append(constructor(cell["source"]))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    return notebook


def report_cell_start(*, cell: nbformat.NotebookNode, cell_index: int) -> None:
    """Emit enough progress to locate an environmental kernel failure."""

    first_line = cell.source.splitlines()[0] if cell.source else "<empty>"
    print(f"Executing cell {cell_index + 1:02d}: {first_line[:80]}", flush=True)


def main() -> None:
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
        allow_errors=False,
        on_cell_start=report_cell_start,
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK)
    print(f"Built and executed {NOTEBOOK} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
