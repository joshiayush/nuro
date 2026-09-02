#!/usr/bin/env python3

# Copyright 2025 The Nuro Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Notebook injection tool for `tools/manage.py`.

Provides programmatic creation and cell-level mutation of `.ipynb`
files so `manage.py notebook` can replace the ad-hoc `build_nb.py`
scripts previously used under `/tmp/opencode/nuro-build/`.

Relative image URLs inside markdown cells are expected to follow the
existing convention: notebooks store images as
`../../../static/images/<name>.png` (relative from notebook path);
`hooks/notebooks.py` symlinks `static/` into `docs/static/` at build time.
This module does not rewrite those URLs itself — see
`image_fetch._relative_static_url()` for the conversion used by
`fetch-image --inject-into`.
"""

from __future__ import annotations

import json
import pathlib
import uuid

_YAPF_STYLE = {
    "based_on_style": "google",
    "indent_width": 2,
    "column_limit": 80,
    "dedent_closing_brackets": False,
    "split_before_first_argument": False,
}


def _format_code(source: str) -> str:
  """Format Python source with yapf Google style if available.

  Args:
    source: Python source string.

  Returns:
    Formatted source, or original if yapf is unavailable or fails.
  """
  try:
    from yapf.yapflib.yapf_api import FormatCode

    formatted, _ = FormatCode(source, style_config=_YAPF_STYLE)
    return formatted
  except Exception:
    return source


def _sort_imports(source: str) -> str:
  """Sort imports with isort Google profile if available."""
  try:
    import isort  # type: ignore

    return isort.code(source, profile="google")
  except Exception:
    return source


def _new_notebook() -> dict:
  """Create a minimal nbformat 4.5 notebook dict."""
  return {
      "cells": [],
      "metadata": {
          "kernelspec": {
              "display_name": "Python 3",
              "language": "python",
              "name": "python3",
          },
          "language_info": {"name": "python"},
      },
      "nbformat": 4,
      "nbformat_minor": 5,
  }


def _load_or_create(path: pathlib.Path, create: bool) -> dict:
  """Load existing notebook or create new one.

  Args:
    path: Notebook path.
    create: If True, create new notebook when missing.

  Returns:
    Notebook dict.

  Raises:
    FileNotFoundError: If missing and create is False.
  """
  if path.exists():
    return json.loads(path.read_text(encoding="utf-8"))
  if not create:
    raise FileNotFoundError(f"notebook not found: {path} (use --create)")
  nb = _new_notebook()
  path.parent.mkdir(parents=True, exist_ok=True)
  return nb


def _make_markdown_cell(source: str) -> dict:
  """Create a markdown cell dict."""
  if not source.endswith("\n"):
    source += "\n"
  return {
      "cell_type": "markdown",
      "id": str(uuid.uuid4()),
      "metadata": {},
      "source": source.splitlines(True),
  }


def _make_code_cell(source: str, do_format: bool = True) -> dict:
  """Create a code cell dict, optionally formatting.

  Args:
    source: Cell source.
    do_format: Whether to run yapf/isort.

  Returns:
    Cell dict.
  """
  if do_format:
    # isort before yapf to avoid yapf undoing import order
    source = _sort_imports(source)
    source = _format_code(source)
  if not source.endswith("\n"):
    source += "\n"
  return {
      "cell_type": "code",
      "id": str(uuid.uuid4()),
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": source.splitlines(True),
  }


def write_cell(
    nb_path: str | pathlib.Path,
    cell_type: str,
    source: str,
    *,
    position: int | str = "append",
    create: bool = False,
    do_format: bool = True,
) -> pathlib.Path:
  """Inject a cell into a notebook, creating the notebook if needed.

  This is the user-facing primitive used by `manage.py notebook`.
  It replaces the pattern of writing a bespoke `build_nb.py` that
  manually constructs `cells` with `md()`/`code()` helpers and handles
  triple-quote delimiter issues.

  Args:
    nb_path: Path to `.ipynb` file (relative or absolute).
    cell_type: "markdown" or "code".
    source: Cell source text (raw string, not split).
    position: "append", "prepend" or integer index. When int, inserts
      before that index; negative indexes count from end like list insert.
    create: Create notebook if it does not exist.
    do_format: Run yapf/isort on code cells (Google style per
      pyproject.toml).

  Returns:
    Resolved notebook path.
  """
  path = pathlib.Path(nb_path)
  nb = _load_or_create(path, create=create)

  if cell_type not in ("markdown", "code"):
    raise ValueError("cell_type must be 'markdown' or 'code'")

  cell = (
      _make_markdown_cell(source)
      if cell_type == "markdown"
      else _make_code_cell(source, do_format=do_format)
  )

  cells = nb["cells"]
  if position == "append":
    cells.append(cell)
  elif position == "prepend":
    cells.insert(0, cell)
  else:
    idx = int(position)  # type: ignore[arg-type]
    if idx < 0:
      idx = max(0, len(cells) + idx + 1)
    idx = min(max(0, idx), len(cells))
    cells.insert(idx, cell)

  path.parent.mkdir(parents=True, exist_ok=True)
  # Atomic write
  tmp = path.with_suffix(".tmp.ipynb")
  tmp.write_text(json.dumps(nb, indent=1), encoding="utf-8")
  tmp.replace(path)
  return path


def replace_cell(
    nb_path: str | pathlib.Path,
    index: int,
    source: str,
    *,
    cell_type: str | None = None,
    do_format: bool = True,
) -> pathlib.Path:
  """Replace cell at index.

  Args:
    nb_path: Notebook path.
    index: Cell index to replace.
    source: New source text.
    cell_type: If given, also change cell type; else keep existing.
    do_format: Format if resulting cell is code.

  Returns:
    Notebook path.
  """
  path = pathlib.Path(nb_path)
  if not path.exists():
    raise FileNotFoundError(f"notebook not found: {path}")
  nb = json.loads(path.read_text(encoding="utf-8"))
  cells = nb["cells"]
  if not 0 <= index < len(cells):
    raise IndexError(f"cell index {index} out of range (0..{len(cells)-1})")
  orig_type = cells[index]["cell_type"]
  new_type = cell_type or orig_type
  if new_type not in ("markdown", "code"):
    raise ValueError("cell_type must be 'markdown' or 'code'")
  new_cell = (
      _make_markdown_cell(source)
      if new_type == "markdown"
      else _make_code_cell(source, do_format=do_format)
  )
  # Preserve id? Generate new one to match nbformat 4.5 uuid style
  # but keep execution_count/outputs cleared for replaced cell
  cells[index] = new_cell
  tmp = path.with_suffix(".tmp.ipynb")
  tmp.write_text(json.dumps(nb, indent=1), encoding="utf-8")
  tmp.replace(path)
  return path


def create_notebook(nb_path: str | pathlib.Path, title: str | None = None) -> pathlib.Path:
  """Create a new empty notebook with optional title markdown cell.

  Args:
    nb_path: Destination path.
    title: If given, first cell will be `# <title>`.

  Returns:
    Created path.

  Raises:
    FileExistsError: If file already exists.
  """
  path = pathlib.Path(nb_path)
  if path.exists():
    raise FileExistsError(f"notebook already exists: {path}")
  nb = _new_notebook()
  if title:
    nb["cells"].append(_make_markdown_cell(f"# {title}\n"))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
  return path
