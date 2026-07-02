"""Guard: the generated REACTIONS.md must stay in sync with the reaction table.

Without this, a change to a reaction note (e.g. the '298 K ref' labels) can leave the committed
REACTIONS.md stale and contradicting its source -- exactly what re-seeded the 298/300 K confusion.
Regenerate with:  python tests/make_reaction_reference.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_reaction_reference import REACTIONS_MD, render  # noqa: E402


def test_reactions_md_is_in_sync():
    with open(REACTIONS_MD) as f:
        committed = f.read()
    assert committed == render(), (
        "REACTIONS.md is stale vs reactions.py. Regenerate: python tests/make_reaction_reference.py")
