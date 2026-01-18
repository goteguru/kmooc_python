#!/usr/bin/env python3
"""
Insert an "Open in Colab" badge markdown cell at the top of a .ipynb notebook.

Usage:
  python insert_node.py path/to/notebook.ipynb

Behavior:
- Prepends one markdown cell to notebook["cells"].
- The cell is inserted exactly as provided, except the filename inside the href.
- Modifies the file in place (writes back to the same path).
- Idempotent: if the first cell already matches the badge (same metadata + href for this filename),
  it will not add another.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List


COLAB_HREF_PREFIX = "https://colab.research.google.com/github/goteguru/kmooc_python/blob/main/notebooks/en/"
COLAB_BADGE_HTML = (
    '<a href="{href}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" '
    'alt="Open In Colab"/></a>'
)


def make_badge_cell(filename: str) -> Dict[str, Any]:
    href = f"{COLAB_HREF_PREFIX}{filename}"
    return {
        "cell_type": "markdown",
        "metadata": {
            "id": "view-in-github",
            "colab_type": "text",
        },
        "source": [COLAB_BADGE_HTML.format(href=href)],
    }


def is_same_badge_cell(cell: Dict[str, Any], filename: str) -> bool:
    """Check whether `cell` is the exact badge cell for `filename`."""
    if not isinstance(cell, dict):
        return False

    if cell.get("cell_type") != "markdown":
        return False

    meta = cell.get("metadata")
    if meta != {"id": "view-in-github", "colab_type": "text"}:
        return False

    expected_source = make_badge_cell(filename)["source"]
    return cell.get("source") == expected_source


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python insert_node.py path/to/notebook.ipynb", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not path.lower().endswith(".ipynb"):
        print(f"Error: not an .ipynb file: {path}", file=sys.stderr)
        return 2

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    filename = os.path.basename(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        return 1

    cells = nb.get("cells")
    if not isinstance(cells, list):
        print("Error: notebook JSON has no 'cells' list.", file=sys.stderr)
        return 1

    badge_cell = make_badge_cell(filename)

    # Idempotency: if first cell is already the same badge, do nothing.
    if cells and is_same_badge_cell(cells[0], filename):
        return 0

    cells.insert(0, badge_cell)

    # Write back, preserving other notebook keys exactly as loaded.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
