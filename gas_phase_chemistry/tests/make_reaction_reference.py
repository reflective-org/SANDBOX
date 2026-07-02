"""Regenerate REACTIONS.md -- the shareable reaction reference table.

Usage:
    python3 tests/make_reaction_reference.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reactions import reference_table  # noqa: E402

HEADER = """# Reaction reference

Stable reference numbers for the box-model mechanism (`src-python/reactions.py`).

- **R#** — stable position in the mechanism table.
- **k-label** — the original MATLAB rate-constant name (`concs_het.m`), incl. a/b branches.

In code: `reactions.BY_RNUMBER["R26"]` or `reactions.BY_KLABEL["k22"]`; list everything with
`reactions.MECHANISM.describe()`.

"""

# Path to the committed REACTIONS.md (single source of truth for both the writer and the sync test).
REACTIONS_MD = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "REACTIONS.md"))


def render() -> str:
    """The full REACTIONS.md content (header + generated table). Used by main() and the sync test."""
    return HEADER + reference_table() + "\n"


def main():
    with open(REACTIONS_MD, "w") as f:
        f.write(render())
    print(f"Wrote {os.path.relpath(REACTIONS_MD)}")


if __name__ == "__main__":
    main()
