#!/usr/bin/env python
"""Pipeline runner — the port of ``run.R`` (``targets::tar_make()``).

IMPORTANT: unlike ``tar_make()``, this does NOT execute the pipeline. Execution
requires the real, access-restricted NCDS data (or the synthetic set of Task 2.4).
What it does instead is BUILD and VALIDATE the dependency graph and print an honest
plan: the topological order length, the stub (unimplemented) roots, how many targets
are blocked by them, and the frontier that would run once real data are
in place.

Run:  python run.py
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.path.insert(0, "src")
    from llm_cong_predict.pipeline.build import build_pipeline

    pipe = build_pipeline()  # raises if the wiring is unsound (dangling deps / cycle)

    print("Pipeline wiring validated: dependencies resolve and the graph is acyclic.\n")
    print(pipe.status_report())

    blocked = pipe.blocked()
    roots = pipe.stub_roots()
    print("\nWhat blocks the pipeline (stub roots and their downstream impact):")
    for r in roots:
        downstream = sum(1 for _n, causes in blocked.items() if r in causes and _n != r)
        print(f"  - {r}: blocks {downstream} downstream target(s)")
        print(f"      {pipe.get(r).note}")

    print(
        "\nBottom line: the graph is structurally sound, but it cannot run. The "
        "remaining stubs are the readability boundary (koRpus, Task 2.5) and the "
        "restricted gene data; execution itself is Task 2.4."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
