# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for ``studio.cli.run`` with the same argv shape and no model.

``LocalSubprocessRunner`` takes ``entry_module`` as a parameter so its lifecycle -- queueing,
timeouts, cancellation, log capture, exit codes -- can be exercised in milliseconds instead of the
three to five minutes a real 10-day case costs. Nothing in the runner branches on the value; the
default is the real entry point, and the real one is exercised separately by
``test_a_bad_input_file_exits_distinctly_from_a_model_failure``.

Behaviour comes from the ``STUDIO_FAKE_RUN`` environment variable, as JSON. An environment variable
rather than a file in the work directory because the subprocess can start before a file written
after ``submit()`` lands -- a race that would make these tests flaky in exactly the way process
tests usually are.

Modes: ``ok`` (print and exit 0), ``fail`` (print to stderr and exit 1), ``sleep`` (sleep, to be
killed by the wall-clock limit or by cancellation), ``dump_env`` (print the thread-pinning variables
this process actually received).
"""

from __future__ import annotations

import json
import os
import sys
import time

#: The pinning the runner is supposed to apply. Imported rather than restated so that the test
#: asserting it cannot pass against a stale copy of the list.
from studio.runner.local import THREAD_PINNING


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(f"usage: fake_run <input.json> <outdir>; got {argv}", file=sys.stderr)
        return 2

    directive = json.loads(os.environ.get("STUDIO_FAKE_RUN", '{"mode": "ok"}'))
    mode = directive.get("mode", "ok")

    if mode == "dump_env":
        print(json.dumps({key: os.environ.get(key) for key in THREAD_PINNING}))
        return 0
    if mode == "sleep":
        time.sleep(float(directive.get("seconds", 30)))
        return 0
    if mode == "fail":
        print(directive.get("message", "failed"), file=sys.stderr)
        return 1
    print(directive.get("message", "[studio] fake run complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
