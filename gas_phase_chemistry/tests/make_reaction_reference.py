"""Regenerate REACTIONS.md -- the shareable reaction reference table.

Usage:
    python3 tests/make_reaction_reference.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from reactions import reference_table  # noqa: E402

HEADER = """# Reaction reference

Stable reference numbers for the box-model mechanism (`src-python/reactions.py`).

- **R#** — stable position in the mechanism table.
- **k-label** — the original MATLAB rate-constant name (`concs_het.m`), incl. a/b branches.

In code: `reactions.BY_RNUMBER["R26"]` or `reactions.BY_KLABEL["k22"]`; list everything with
`reactions.MECHANISM.describe()`.

"""


def main():
    out = os.path.join(os.path.dirname(__file__), "..", "REACTIONS.md")
    with open(out, "w") as f:
        f.write(HEADER + reference_table() + "\n")
    print(f"Wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
