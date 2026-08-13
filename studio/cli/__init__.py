# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface.

SCRIPTED ENSEMBLES MUST NOT REQUIRE THE BROWSER. This is a hard requirement (ADR-002): the existing
workflow is entirely scripted and has to stay viable.

Planned surface:

* ``plume-studio run <config.yaml> [--out DIR]``   -- one run
* ``plume-studio sweep <sweep.yaml> [--plan]``     -- expand axes; ``--plan`` prints N and submits
                                                      nothing (mirrors the existing runners'
                                                      ``plan`` verb)
* ``python -m studio.cli.run <config.yaml> <outdir>`` -- the subprocess entry point that
  ``LocalSubprocessRunner`` launches. Deliberately thin: ``run_coupled`` is a library call with no
  ``__main__`` of its own, so something has to be the process.

The existing runner scripts share a CLI shape -- ``plan | one <i> | run <lo> <hi>``
(``run_ensemble.py:174``) -- and Studio's verbs stay recognisable to anyone who has used them.
"""

from __future__ import annotations

__all__: list[str] = []
