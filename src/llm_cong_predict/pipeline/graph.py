"""Generic dependency-graph machinery for the pipeline.

This is the STRUCTURE: it resolves target dependencies, detects cycles, produces a
topological order, and computes which targets are *blocked* because some dependency
is not implemented (a "stub"). Execution is in pipeline/execute.py (Task 2.4).

A ``Target`` is a name, its dependency names, and a status flag:
  * BUILT    — the underlying function is implemented and runs given its inputs;
  * EXTERNAL — produced outside this pipeline by another program, whose output a
               downstream target reads (tokenisation happens inside the koRpus
               script r/readability.R); it has no Python value and blocks nothing;
  * STUB     — not implemented / raises.

"Blocked" is transitive: a target is blocked if it is a STUB or if any of its
dependencies is blocked. So one STUB deep in the graph blocks everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    BUILT = "built"
    EXTERNAL = "external"
    STUB = "stub"


@dataclass(frozen=True)
class Target:
    name: str
    deps: tuple[str, ...] = ()
    status: Status = Status.BUILT
    note: str = ""


class Pipeline:
    """A validated DAG of targets."""

    def __init__(self, targets: list[Target]):
        self._t: dict[str, Target] = {}
        for t in targets:
            if t.name in self._t:
                raise ValueError(f"duplicate target: {t.name!r}")
            self._t[t.name] = t

    def __len__(self) -> int:
        return len(self._t)

    def __contains__(self, name: str) -> bool:
        return name in self._t

    def get(self, name: str) -> Target:
        return self._t[name]

    def names(self) -> list[str]:
        return list(self._t)

    # --- validation ---------------------------------------------------------

    def check_dependencies(self) -> None:
        """Raise if any target depends on a name that is not a registered target."""
        for t in self._t.values():
            for d in t.deps:
                if d not in self._t:
                    raise ValueError(f"target {t.name!r} depends on unknown target {d!r}")

    def topo_order(self) -> list[str]:
        """Return a topological order (dependencies before dependents).

        Uses Kahn's algorithm; raises ``ValueError`` if the graph has a cycle.
        """
        self.check_dependencies()
        indeg = {n: len(t.deps) for n, t in self._t.items()}
        dependents: dict[str, list[str]] = {n: [] for n in self._t}
        for t in self._t.values():
            for d in t.deps:
                dependents[d].append(t.name)

        queue = sorted(n for n, dg in indeg.items() if dg == 0)
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            newly_ready = []
            for m in dependents[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    newly_ready.append(m)
            queue.extend(sorted(newly_ready))

        if len(order) != len(self._t):
            cyclic = sorted(n for n, dg in indeg.items() if dg > 0)
            raise ValueError(f"cycle detected involving: {cyclic[:12]}")
        return order

    def validate(self) -> None:
        """Full structural validation: dependencies resolve and the graph is acyclic."""
        self.topo_order()

    # --- blocked-node analysis ---------------------------------------------

    def blocked(self) -> dict[str, set[str]]:
        """Map each blocked target -> the set of STUB targets that block it.

        A target is blocked if it is a STUB itself or transitively depends on one.
        Computed in topological order so each node sees its dependencies' results.
        """
        order = self.topo_order()
        blocked: dict[str, set[str]] = {}
        for n in order:
            t = self._t[n]
            roots: set[str] = set()
            if t.status is Status.STUB:
                roots.add(n)
            for d in t.deps:
                roots |= blocked.get(d, set())
            if roots:
                blocked[n] = roots
        return blocked

    def stub_roots(self) -> list[str]:
        """The STUB targets — the root causes of everything that is blocked."""
        return sorted(n for n, t in self._t.items() if t.status is Status.STUB)

    def runnable_frontier(self) -> list[str]:
        """Targets that are BUILT and NOT blocked, i.e. that run given their inputs
        (pipeline/execute.py; synthetic inputs in tests, real ones in Phase 4)."""
        b = self.blocked()
        return sorted(n for n, t in self._t.items() if t.status is Status.BUILT and n not in b)

    def status_report(self) -> str:
        """A plain-text summary of the pipeline's structural state."""
        order = self.topo_order()
        b = self.blocked()
        roots = self.stub_roots()
        frontier = self.runnable_frontier()
        lines = [
            f"targets: {len(self._t)}   (topological order resolves, graph is acyclic)",
            f"stub roots (not implemented): {len(roots)} -> {roots}",
            f"blocked (stub or downstream of one): {len(b)} / {len(self._t)}",
            f"external (produced by another program): {sorted(n for n, t in self._t.items() if t.status is Status.EXTERNAL)}",
            f"runnable given their inputs: {len(frontier)}",
            "",
            "The numbers above describe the WIRING. pipeline/execute.py runs it.",
        ]
        return "\n".join(lines)
