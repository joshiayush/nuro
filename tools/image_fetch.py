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
"""Image fetch tool for `tools/manage.py`.

Implements `fetch-image` which mirrors the assistant's `WebFetch`
workflow: first fetch the webpage (discover selectors), then resolve
the image by human-provided identifier (title/alt or section-relative
position), then download to `static/images/` and report the relative
URL as stored in notebooks (e.g. `../../../static/images/foo.png`).

Preference: identifier is *never* a CSS selector — it is a title/alt
substring or a heading + nth. The tool derives the selector itself via
an initial webfetch+parse.
"""

from __future__ import annotations

import pathlib
import re
import urllib.parse
import urllib.request

# Optional deps — fall back to stdlib if missing
try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup, Tag  # type: ignore
except ImportError:
    BeautifulSoup = None  # type: ignore
    Tag = None  # type: ignore


def _fetch_html(url: str, timeout: int = 15) -> str:
  """Fetch HTML, preferring requests if available."""
  if requests is not None:
    resp = requests.get(
        url, timeout=timeout, headers={"User-Agent": "nuro-image-fetch/1.0"}
    )
    resp.raise_for_status()
    return resp.text
  req = urllib.request.Request(url, headers={"User-Agent": "nuro-image-fetch/1.0"})
  with urllib.request.urlopen(req, timeout=timeout) as fh:  # type: ignore
    charset = fh.headers.get_content_charset() or "utf-8"
    return fh.read().decode(charset, errors="replace")


def _parse(html: str, url: str):
  """Parse HTML into BeautifulSoup or fallback."""
  if BeautifulSoup is not None:
    return BeautifulSoup(html, "html.parser")
  # Minimal fallback: regex-based extraction (only title/alt exact)
  # Caller will handle fallback path if soup is None-ish.
  return html  # type: ignore


def _relative_static_url(
    notebook_path: str | pathlib.Path | None, image_path: pathlib.Path
) -> str:
  """Return relative URL from notebook to static image.

  Mirrors existing notebooks: e.g. `notebooks/ml/models/Foo.ipynb`
  references `../../../static/images/bar.png`.

  Args:
    notebook_path: Notebook that will embed the image, or None.
    image_path: Absolute or repo-relative static image path.

  Returns:
    Relative POSIX string to use inside `![alt](url)`.
  """
  if notebook_path is None:
    # Repo-root relative
    return image_path.as_posix()
  nb = pathlib.Path(notebook_path)
  # Image is always under static/images, we compute relative from notebook's dir
  try:
    rel = pathlib.Path(
        pathlib.Path.cwd() / image_path
    ).relative_to(pathlib.Path.cwd())
  except Exception:
    rel = image_path
  # Compute relative from notebook dir to image
  # Both are absolute-ish under repo root
  nb_dir = (pathlib.Path.cwd() / nb).parent if not nb.is_absolute() else nb.parent
  img_abs = pathlib.Path.cwd() / image_path if not image_path.is_absolute() else image_path
  try:
    return pathlib.Path(
        pathlib.Path.cwd() / pathlib.Path.relative_to(img_abs, pathlib.Path.cwd())
    ).as_posix()
  except Exception:
    pass
  # Fallback: use os.path.relpath
  import os

  return os.path.relpath(os.fspath(img_abs), os.fspath(nb_dir))


def _find_image_src(
    soup,
    html: str,
    base_url: str,
    *,
    title: str | None,
    section: str | None,
    nth: int = 0,
) -> tuple[str, str]:
  """Resolve image src and a human-readable selector description.

  Preference order (per saved user preference):
  1. If `title` given: first `<img>` where alt/title/figcaption contains title (case-insensitive).
  2. If `section` given: heading (`h1..h6`) whose text contains section, then nth `img` after it until next heading.
  3. Otherwise: nth img on page.

  Returns:
    (resolved_abs_url, selector_description)

  Raises:
    ValueError: If not found.
  """
  # BeautifulSoup path
  if BeautifulSoup is not None and hasattr(soup, "find_all"):
    imgs = soup.find_all("img")
    if title:
      q = title.lower()
      for img in imgs:
        hay = " ".join(
            filter(
                None,
                [
                    img.get("alt", ""),
                    img.get("title", ""),
                    # figcaption sibling
                    (
                        img.find_parent("figure").find("figcaption").get_text()
                        if img.find_parent("figure") and img.find_parent("figure").find("figcaption")
                        else ""
                    ),
                ],
            )
        ).lower()
        # also check src itself contains title (e.g. filename)
        if q in hay or q in (img.get("src") or "").lower():
          src = img.get("src")
          if not src:
            continue
          abs_url = urllib.parse.urljoin(base_url, src)
          sel = f'img[alt~="{title}" i] -> {src}'
          return abs_url, sel
      raise ValueError(f'no image matching title="{title}" (checked {len(imgs)} imgs)')

    if section:
      q = section.lower()
      # Find heading
      heading = None
      for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        if q in tag.get_text(strip=True).lower():
          heading = tag
          break
      if heading is None:
        raise ValueError(f'section heading containing "{section}" not found')
      # Collect imgs after heading until next heading of same or higher level
      h_level = int(heading.name[1])
      section_imgs = []
      for el in heading.find_all_next():
        if el.name and re.match(r"^h[1-6]$", el.name):
          if int(el.name[1]) <= h_level:
            break
          # for lower-level headings, continue but still collect imgs between
        if el.name == "img":
          section_imgs.append(el)
        # also imgs inside figures/divs will be visited as img tags themselves
      if not section_imgs:
        raise ValueError(f'no images under section "{section}"')
      if nth >= len(section_imgs):
        raise ValueError(
            f'section "{section}" has {len(section_imgs)} images, nth={nth} out of range'
        )
      img = section_imgs[nth]
      src = img.get("src")
      abs_url = urllib.parse.urljoin(base_url, src)
      sel = f'section "{section}" -> img[{nth}] ({src})'
      return abs_url, sel

    # Fallback: nth globally
    if not imgs:
      raise ValueError("no images on page")
    if nth >= len(imgs):
      raise ValueError(f"page has {len(imgs)} images, nth={nth} out of range")
    img = imgs[nth]
    src = img.get("src")
    abs_url = urllib.parse.urljoin(base_url, src)
    return abs_url, f"img[{nth}] ({src})"

  # Stdlib fallback (html string, regex)
  if title:
    # Find img tags via regex
    img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', html, re.I)
    alts = re.findall(r'<img[^>]+alt=["\']([^"\']*)["\']', html, re.I)
    # naive contains check
    q = title.lower()
    for i, src in enumerate(img_srcs):
      alt = alts[i] if i < len(alts) else ""
      if q in alt.lower() or q in src.lower():
        return urllib.parse.urljoin(base_url, src), f'img alt~="{title}"'
    raise ValueError(f'no image matching title="{title}" (fallback parser)')
  # nth fallback
  img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.I)
  if not img_srcs:
    raise ValueError("no images found (fallback)")
  if nth >= len(img_srcs):
    raise ValueError(f"page has {len(img_srcs)} images, nth={nth} out of range")
  src = img_srcs[nth]
  return urllib.parse.urljoin(base_url, src), f"img[{nth}]"


def fetch_image(
    page_url: str,
    dest: str | pathlib.Path,
    *,
    title: str | None = None,
    section: str | None = None,
    nth: int = 0,
    timeout: int = 15,
) -> tuple[pathlib.Path, str, str]:
  """Fetch webpage, resolve image, download to dest.

  Implements the preference: first webfetch to discover selector,
  identifier is never a CSS selector.

  Args:
    page_url: Webpage URL.
    dest: Destination file path under `static/images/` (will be created).
      If a directory, filename is derived from src URL.
    title: Substring to match against img alt/title/figcaption/src.
    section: Heading text to scope search (first heading containing
      this text); combined with nth for relative positioning.
    nth: Index among matched images (0 = first).
    timeout: HTTP timeout seconds.

  Returns:
    (saved_path, resolved_image_url, selector_description)
  """
  html = _fetch_html(page_url, timeout=timeout)
  soup = _parse(html, page_url)
  img_url, selector = _find_image_src(
      soup, html, page_url, title=title, section=section, nth=nth
  )

  dest_path = pathlib.Path(dest)
  if dest_path.is_dir() or str(dest).endswith("/"):
    dest_path.mkdir(parents=True, exist_ok=True)
    fname = pathlib.Path(urllib.parse.urlparse(img_url).path).name or "image.png"
    dest_path = dest_path / fname
  else:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

  # Download image
  if requests is not None:
    resp = requests.get(img_url, timeout=timeout, stream=True, headers={"User-Agent": "nuro-image-fetch/1.0"})
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
      for chunk in resp.iter_content(8192):
        f.write(chunk)
  else:
    urllib.request.urlretrieve(img_url, dest_path)

  return dest_path, img_url, selector


def relative_static_url_for_notebook(
    notebook_path: str | pathlib.Path | None,
    image_path: str | pathlib.Path,
) -> str:
  """Public wrapper for relative URL, used by manage.py --inject-into."""
  import os

  nb = pathlib.Path(notebook_path) if notebook_path else None
  img = pathlib.Path(image_path)

  # Resolve both to repo-root absolute for relpath
  # Assume cwd is repo root (as manage.py does)
  repo_root = pathlib.Path.cwd()
  if nb is not None and not nb.is_absolute():
    nb_abs = repo_root / nb
  elif nb is not None:
    nb_abs = nb
  else:
    return img.as_posix()

  if not img.is_absolute():
    img_abs = repo_root / img
  else:
    img_abs = img

  try:
    rel = os.path.relpath(os.fspath(img_abs), os.fspath(nb_abs.parent))
    return rel.replace(os.sep, "/")
  except Exception:
    return img.as_posix()
