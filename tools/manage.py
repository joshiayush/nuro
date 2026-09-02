#!/usr/bin/python3

# Copyright 2018 The AI Authors. All Rights Reserved.
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
"""Repository management tool for AI."""

from __future__ import annotations

import os
import sys
import pathlib
import argparse

_BASE_AI_DIR = 'notebooks'
_BASE_DOCS_DIR = 'docs'
_BASE_AI_REPO = pathlib.Path(__file__).parent.parent


def GenerateAIDocs() -> int:
  """Completes the action of command "--generate-docs" by generating markdown
  files from the IPython Notebooks.

  Returns:
    Error code of `PermissionError`.
  """
  import ai_docs
  import ai_utils

  error_code = 0
  base_docs_dir = pathlib.Path(os.getcwd()) / _BASE_DOCS_DIR
  for file in ai_utils.GetFileByExtensionUnderDirectory(
    'ipynb', os.fspath(pathlib.Path(os.getcwd()) / _BASE_AI_DIR)
  ):
    try:
      ai_docs.GenerateDocs(
        base_docs_dir, file, ai_docs.ReadIPythonNotebookToMarkdown(file)
      )
    except PermissionError as exc:
      error_code = exc.errno
  return error_code


_BASE_AI_MODULE_PATH = {
  'ai': (
    'ai/',
    'examples',
  )
}
_BASE_AI_DOCS_PATH = {
  'docs': (
    'docs/',
    'notebooks/',
    'templates/',
  )
}


def GenerateAILogs() -> int:
  """Completes the action of command "--generate-logs" by generating changelog
  file for the changes made in the "ai" or "docs" submodules.

  Returns:
    Error code of `PermissionError`.
  """
  import ai_logs

  error_code = 0
  changelog = ai_logs.Changelog(_BASE_AI_REPO)
  try:
    changelog.write_changelog_md(
      'CHANGELOG.md', follow=(
        _BASE_AI_MODULE_PATH,
        _BASE_AI_DOCS_PATH,
      )
    )
  except PermissionError as exc:
    error_code = exc.errno
  return error_code


def _handle_notebook(ns: argparse.Namespace) -> int:
  """Handle `notebook` subcommand.

  Supports creating and mutating `.ipynb` files without hand-writing
  `build_nb.py` scripts. Source can be inline, a file, or stdin.
  """
  import nb_tool  # type: ignore

  nb_path = ns.path
  # Allow ` --create --title` together with --inject/--replace: create empty notebook first if missing
  if ns.create and ns.path and not pathlib.Path(ns.path).exists() and ns.title:
    try:
      nb_tool.create_notebook(nb_path, title=ns.title)
      print(f"Created {nb_path}")
      # If only --create was intended, we're done
      if not ns.inject and ns.replace is None:
        return 0
      # Otherwise fall through to inject/replace on the newly created notebook
      # and avoid re-creating via write_cell(create=True) title already handled
      ns.create = False
    except FileExistsError as e:
      print(f"error: {e}", file=sys.stderr)
      return 1
  elif ns.create and not ns.inject and ns.replace is None:
    # Just create empty notebook with optional title (no inject/replace)
    try:
      nb_tool.create_notebook(nb_path, title=ns.title)
      print(f"Created {nb_path}")
      return 0
    except FileExistsError as e:
      print(f"error: {e}", file=sys.stderr)
      return 1

  # Determine source text for inject/replace
  source: str | None = None
  if ns.source_file:
    if ns.source_file == "-":
      source = sys.stdin.read()
    else:
      source = pathlib.Path(ns.source_file).read_text(encoding="utf-8")
  elif ns.source is not None:
    source = ns.source

  if ns.inject:
    if source is None:
      print("error: --source or --source-file required for --inject", file=sys.stderr)
      return 2
    ctype = ns.cell_type or "markdown"
    pos = ns.at if ns.at is not None else "append"
    # --at can be "append"/"prepend" or int
    if isinstance(pos, str) and pos not in ("append", "prepend"):
      try:
        pos = int(pos)
      except ValueError:
        pass
    try:
      out = nb_tool.write_cell(
          nb_path,
          ctype,
          source,
          position=pos,  # type: ignore[arg-type]
          create=ns.create,
          do_format=not ns.no_format,
      )
      print(f"Injected {ctype} cell into {out} at {pos}")
      return 0
    except Exception as e:
      print(f"error: {e}", file=sys.stderr)
      return 1

  if ns.replace is not None:
    if source is None:
      print("error: --source or --source-file required for --replace", file=sys.stderr)
      return 2
    try:
      out = nb_tool.replace_cell(
          nb_path,
          int(ns.replace),
          source,
          cell_type=ns.cell_type,
          do_format=not ns.no_format,
      )
      print(f"Replaced cell {ns.replace} in {out}")
      return 0
    except Exception as e:
      print(f"error: {e}", file=sys.stderr)
      return 1

  print("error: specify --inject or --replace, or --create", file=sys.stderr)
  return 2


def _handle_fetch_image(ns: argparse.Namespace) -> int:
  """Handle `fetch-image` subcommand.

  Workflow (per saved preference): first webfetch the page to discover
  the selector from a human description (title or section+nth), then
  download. Never expects a CSS selector from the user.
  """
  import image_fetch  # type: ignore

  # Validate identifier
  if not ns.title and not ns.section:
    print(
        "error: provide --title or --section (nth is optional) "
        "to identify the image without using CSS",
        file=sys.stderr,
    )
    return 2

  try:
    saved, img_url, selector = image_fetch.fetch_image(
        ns.url,
        ns.out,
        title=ns.title,
        section=ns.section,
        nth=ns.nth,
        timeout=ns.timeout,
    )
    # Prefer relative URL style for notebooks (per preference #2)
    rel = None
    if ns.inject_into:
      rel = image_fetch.relative_static_url_for_notebook(ns.inject_into, saved)
      # Optionally inject markdown image into notebook
      if ns.alt is not None:
        alt = ns.alt
      elif ns.title:
        alt = ns.title
      else:
        alt = saved.stem
        # For section-based fetch, use section as alt if title not given
        if ns.section:
          alt = ns.section
      md = f"![{alt}]({rel})"
      import nb_tool  # type: ignore

      # Inject at end or given position
      pos = ns.inject_at if ns.inject_at is not None else "append"
      if isinstance(pos, str) and pos not in ("append", "prepend"):
        try:
          pos = int(pos)
        except ValueError:
          pass
      nb_tool.write_cell(
          ns.inject_into,
          "markdown",
          md,
          position=pos,  # type: ignore[arg-type]
          create=False,
          do_format=False,
      )
      print(f"Injected image markdown into {ns.inject_into}: {md}")

    print(f"Discovered selector: {selector}")
    print(f"Resolved image URL: {img_url}")
    print(f"Saved to: {saved}")
    if rel:
      print(f"Relative URL for notebook: {rel}")
    elif ns.inject_into is None:
      # Still show relative from repo root for convenience
      try:
        print(
            f"Relative URL (from repo root): {image_fetch.relative_static_url_for_notebook(None, saved)}"
        )
      except Exception:
        pass
    return 0
  except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    return 1


def Main(namespace: argparse.Namespace) -> int:
  """Takes actions according to the commands that are given over the
  command-line.

  Args:
    namespace: Stores the commands given over the command-line.
  """
  # Subcommands take precedence
  if getattr(namespace, "subcommand", None) == "notebook":
    return _handle_notebook(namespace)
  if getattr(namespace, "subcommand", None) == "fetch-image":
    return _handle_fetch_image(namespace)
  # Legacy flags
  if getattr(namespace, "generate_docs", False):
    return GenerateAIDocs()
  if getattr(namespace, "generate_logs", False):
    return GenerateAILogs()
  return 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog=(
      'Post an issue at https://github.com/joshiayush/ai/issues '
      'for any modification in this program.'
    )
  )
  # Legacy flags (kept for backward compat)
  parser.add_argument(
    '--generate-docs',
    action='store_true',
    help='Triggers the action of generating documents from IPython Notebooks.'
  )
  parser.add_argument(
    '--generate-logs',
    action='store_true',
    help='Generates changelogs for the project.'
  )

  sub = parser.add_subparsers(dest="subcommand", help="subcommands")

  # ---- notebook ----
  p_nb = sub.add_parser("notebook", help="Create/inject cells into .ipynb files")
  p_nb.add_argument("path", nargs="?", help="Notebook path (e.g. notebooks/ml/Foo.ipynb)")
  # For `notebook --create` without sub-subparser, keep simple flags
  p_nb.add_argument("--create", action="store_true", help="Create notebook if missing (also allows inject to create)")
  p_nb.add_argument("--title", help="Title for --create (first markdown cell '# <title>')")
  p_nb.add_argument("--inject", action="store_true", help="Inject a new cell")
  p_nb.add_argument("--replace", metavar="INDEX", help="Replace cell at INDEX")
  p_nb.add_argument("--cell-type", choices=["markdown", "code"], help="Cell type (default markdown for inject, keep existing for replace)")
  p_nb.add_argument("--source", help="Cell source as inline string")
  p_nb.add_argument("--source-file", help="Read cell source from file (use - for stdin)")
  p_nb.add_argument("--at", help="Position for inject: append, prepend, or integer index (default append)")
  p_nb.add_argument("--inject-at", help="Deprecated alias for --at when used with fetch-image; use --at for notebook")
  p_nb.add_argument("--no-format", action="store_true", help="Skip yapf/isort formatting for code cells")

  # ---- fetch-image ----
  p_img = sub.add_parser(
      "fetch-image",
      help="Download an image from a webpage by title or section-relative position (webfetch first)",
  )
  p_img.add_argument("--url", required=True, help="Webpage URL to fetch")
  p_img.add_argument("--out", required=True, help="Destination file or directory under static/images/")
  p_img.add_argument("--title", help="Substring to match against img alt/title/figcaption/src (case-insensitive)")
  p_img.add_argument("--section", help="Heading text to scope search (first h1..h6 containing this text)")
  p_img.add_argument("--nth", type=int, default=0, help="Nth image among matches (0 = first, default 0)")
  p_img.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds (default 15)")
  p_img.add_argument("--inject-into", help="Notebook path to inject markdown image link into after download")
  p_img.add_argument("--inject-at", help="Position in notebook for injected image markdown (default append)")
  p_img.add_argument("--alt", help="Alt text for injected markdown (default: title or section or filename)")

  sys.exit(Main(parser.parse_args()))
