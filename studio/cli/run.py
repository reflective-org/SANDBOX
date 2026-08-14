# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``python -m studio.cli.run <input.json> <outdir>`` -- the subprocess entry point.

Deliberately thin. ``run_coupled`` is a library call with no ``__main__`` of its own (BLOCKING-3),
so something has to be the process that ``LocalSubprocessRunner`` launches, and this is it. All the
work is in ``studio.modelio.execute``; everything here is argument handling and exit codes.

Exit codes are the runner's only structured signal, so they mean something specific:

* ``0`` -- the model ran and the outputs were written.
* ``2`` -- the input could not be read or validated. Nothing was run.
* ``1`` -- the model raised. The traceback is on stderr, which the runner captures in full so the
  failure can be diagnosed **without re-running it**.

Nothing is caught and turned into a plausible result. A run that fails must look failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from studio.resolve import ResolvedConfig

#: Exit code for a bad or unreadable input, distinguished from a model failure so the runner can
#: tell "we never started" from "it broke".
EXIT_BAD_INPUT = 2
EXIT_MODEL_FAILURE = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m studio.cli.run",
        description="Run one resolved Plume Studio configuration and write its outputs.",
    )
    parser.add_argument(
        "input", type=Path, help="resolved config JSON (a serialised ResolvedConfig)"
    )
    parser.add_argument(
        "outdir", type=Path, help="directory to write state.npz and summary.json into"
    )
    args = parser.parse_args(argv)

    try:
        config = ResolvedConfig.model_validate_json(args.input.read_text(encoding="utf-8"))
        config.require_consistent()
    except Exception as exc:
        print(f"[studio] cannot run {args.input}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    # Imported here, not at module scope: it pulls in the model and therefore JAX, and a bad input
    # should fail in milliseconds rather than after a second of imports.
    from studio.modelio.execute import run_and_write

    try:
        written = run_and_write(config, args.outdir)
    except Exception:
        traceback.print_exc()
        return EXIT_MODEL_FAILURE

    for name, path in written.items():
        print(f"[studio] wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
