#!/usr/bin/env python
"""Extract the model-target inventory from the original ``_targets.R``.

Reads ``reference/llm_paper/_targets.R`` and ``reference/llm_paper/R/create_data.R``
(public code at commit b0cfe4c, see docs/REFERENCE_SOURCES.md) and writes
``docs/reference/r_targets_inventory.json``. For every model target it records:

  * the R name and line;
  * the method: ``get_general_superlearner_cv_model`` -> "superlearner",
    ``get_lm_cv_model`` -> "lm";
  * the outcome (an outcome-list target, or a literal column) and ``pattern``;
  * the predictor sets (variable-list targets or literal columns, in order);
  * the sample (the data-frame target) and the data preparation applied to it
    (``as.numeric`` conversions, joins, column drops);
  * the scorer, from ``_targets.R`` metric targets and from ``create_data.R``;
    "none" when neither file scores the target.

It also resolves the constant variable lists (literal vectors and their
compositions) so that the number of fits can be counted. The parser is a small
scanner for balanced parentheses and quotes, written for this one pinned file; any
construct it does not recognise raises instead of being guessed.

Usage:  python scripts/extract_r_targets.py [--check]
        --check  compare with the committed JSON instead of writing it (exit 1 if different)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LLM_PAPER = REPO / "reference" / "llm_paper"
TARGETS_R = LLM_PAPER / "_targets.R"
CREATE_DATA_R = LLM_PAPER / "R" / "create_data.R"
OUT = REPO / "docs" / "reference" / "r_targets_inventory.json"
COMMIT = "b0cfe4ceba0c29fcc6121aa7ff49c761b8127285"

MODEL_FUNCTIONS = {"get_general_superlearner_cv_model": "superlearner", "get_lm_cv_model": "lm"}
SCORERS = {"get_cv_superlearner_metrics": "superlearner", "get_cv_lm_metrics": "lm"}
IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"


# ------------------------------------------------------------------ scanning --

def _mask(text: str) -> str:
    """Copy of ``text`` with comments blanked and string contents kept, so that
    positions line up; used to find calls outside comments."""
    out, i, quote = list(text), 0, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _matching_paren(text: str, open_idx: int) -> int:
    depth, i, quote = 0, open_idx, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced parenthesis at {open_idx}")


def find_calls(text: str, fname: str):
    """Yield ``(line, args_text)`` for every call ``fname(...)`` outside comments."""
    masked = _mask(text)
    for m in re.finditer(rf"(?<![A-Za-z0-9_.:]){re.escape(fname)}\s*\(", masked):
        open_idx = m.end() - 1
        close = _matching_paren(masked, open_idx)
        yield masked.count("\n", 0, m.start()) + 1, masked[open_idx + 1:close]


def split_args(args: str) -> list[str]:
    """Split an argument list at top-level commas; drop empty trailing arguments."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(args):
        c = args[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(args[start:i].strip())
            start = i + 1
        i += 1
    parts.append(args[start:].strip())
    return [p for p in parts if p]


def _named(arg: str) -> tuple[str | None, str]:
    m = re.fullmatch(rf"({IDENT})\s*=\s*(.+)", arg, flags=re.S)
    if m and not arg.startswith('"'):
        return m.group(1), m.group(2).strip()
    return None, arg


# ------------------------------------------------------------- R expressions --

def parse_ref(expr: str) -> list[dict]:
    """A predictor/outcome expression: identifier, "literal", or c(...) of those."""
    expr = " ".join(expr.split())
    if re.fullmatch(IDENT, expr):
        return [{"kind": "list", "value": expr}]
    m = re.fullmatch(r'"([^"]*)"', expr)
    if m:
        return [{"kind": "literal", "value": m.group(1)}]
    m = re.fullmatch(r"c\((.*)\)", expr)
    if m:
        return [item for a in split_args(m.group(1)) for item in parse_ref(a)]
    raise ValueError(f"unrecognised predictor/outcome expression: {expr!r}")


_PREP_PATTERNS = [
    (rf"dplyr::mutate\(({IDENT}) = as\.numeric\(\1\)\)",
     lambda m: {"op": "as_numeric", "columns": [m.group(1)]}),
    (rf"dplyr::mutate_at\(dplyr::vars\(({IDENT})\), as\.numeric\)",
     lambda m: {"op": "as_numeric", "columns_from": m.group(1)}),
    (rf'dplyr::inner_join\(({IDENT}), by = c\("ncdsid" = "id"\)\)',
     lambda m: {"op": "inner_join", "table": m.group(1), "by": {"ncdsid": "id"}}),
    (r'dplyr::select\(-dplyr::starts_with\("([^"]+)"\)\)',
     lambda m: {"op": "drop_columns_starting_with", "prefix": m.group(1)}),
]


def parse_data(expr: str) -> tuple[str, list[dict]]:
    """``sample %>% step %>% step`` -> (sample, data preparation steps)."""
    steps = [" ".join(s.split()) for s in expr.split("%>%")]
    sample, prep = steps[0], []
    if not re.fullmatch(IDENT, sample):
        raise ValueError(f"unrecognised sample expression: {sample!r}")
    for step in steps[1:]:
        for pattern, build in _PREP_PATTERNS:
            m = re.fullmatch(pattern, step)
            if m:
                prep.append(build(m))
                break
        else:
            raise ValueError(f"unrecognised data-preparation step: {step!r}")
    return sample, prep


# ------------------------------------------------------------------ targets --

def parse_targets(text: str) -> tuple[dict, list[dict], list[dict]]:
    variable_lists: dict[str, dict] = {}
    models: list[dict] = []
    metrics: list[dict] = []
    for line, args in find_calls(text, "tar_target"):
        parts = split_args(args)
        name, command = parts[0], parts[1]
        options = dict(_named(p) for p in parts[2:])
        call = re.match(rf"({IDENT}(?:::{IDENT})?)\s*\(", command)
        head = call.group(1) if call else None

        if head in MODEL_FUNCTIONS:
            margs = split_args(command[command.index("(") + 1: command.rindex(")")])
            if len(margs) != 3:
                raise ValueError(f"{name}: expected 3 model arguments, got {margs}")
            outcome = parse_ref(margs[0])
            if len(outcome) != 1:
                raise ValueError(f"{name}: outcome must be one reference")
            sample, prep = parse_data(margs[2])
            models.append({
                "r_name": name, "line": line, "method": MODEL_FUNCTIONS[head],
                "outcome": outcome[0], "pattern": options.get("pattern"),
                "predictors": parse_ref(margs[1]), "sample": sample, "data_prep": prep,
            })
        elif head in SCORERS:
            inner = command[command.index("(") + 1: command.rindex(")")].strip()
            metrics.append({"r_name": name, "line": line, "scorer": SCORERS[head],
                            "model": inner, "pattern": options.get("pattern")})
        elif head == "c":
            items = parse_ref(command)
            kind = "literal" if all(i["kind"] == "literal" for i in items) else "composite"
            variable_lists[name] = {"line": line, "kind": kind, "items": items}
        elif head == "paste0" and name == "roberta_embeddings_variables":
            m = re.fullmatch(r'paste0\("([^"]+)", 1:(\d+)\)', " ".join(command.split()))
            variable_lists[name] = {"line": line, "kind": "literal", "items": [
                {"kind": "literal", "value": f"{m.group(1)}{i}"} for i in range(1, int(m.group(2)) + 1)]}
        elif head == "colnames":
            variable_lists[name] = {"line": line, "kind": "data-dependent", "expression": " ".join(command.split())}
    return variable_lists, models, metrics


def resolve(variable_lists: dict) -> dict:
    """Members of each constant list; None when it depends on data."""
    resolved: dict[str, list[str] | None] = {}

    def members(name: str):
        if name in resolved:
            return resolved[name]
        entry = variable_lists[name]
        if entry["kind"] == "data-dependent":
            out = None
        else:
            out = []
            for item in entry["items"]:
                if item["kind"] == "literal":
                    out.append(item["value"])
                else:
                    sub = members(item["value"])
                    if sub is None:
                        out = None
                        break
                    out.extend(sub)
        resolved[name] = out
        return out

    for n in variable_lists:
        members(n)
    return resolved


def parse_create_data(text: str, model_names: set[str]) -> list[dict]:
    """Scorer calls on model targets in create_data.R (masked of comments)."""
    uses = []
    for lineno, line in enumerate(_mask(text).splitlines(), start=1):
        for m in re.finditer(rf"tar_read\(({IDENT})\)\s*%>%\s*purrr::map_dfr\(({IDENT})\)", line):
            if m.group(1) in model_names and m.group(2) in SCORERS:
                uses.append({"model": m.group(1), "scorer": SCORERS[m.group(2)], "line": lineno,
                             "form": "tar_read(model) %>% purrr::map_dfr(scorer)"})
        for m in re.finditer(rf"({IDENT})\(tar_read\(({IDENT})\)\[\[1\]\]\)", line):
            if m.group(1) in SCORERS and m.group(2) in model_names:
                uses.append({"model": m.group(2), "scorer": SCORERS[m.group(1)], "line": lineno,
                             "form": "scorer(tar_read(model)[[1]])"})
        for m in re.finditer(rf"({IDENT})\(({IDENT})\)", line):
            if m.group(1) in SCORERS and m.group(2) in model_names:
                uses.append({"model": m.group(2), "scorer": SCORERS[m.group(1)], "line": lineno,
                             "form": "scorer(model) on a bare symbol, not tar_read(): cannot run as written"})
    return uses


def build_inventory() -> dict:
    targets_text = TARGETS_R.read_text()
    variable_lists, models, metrics = parse_targets(targets_text)
    resolved = resolve(variable_lists)
    model_names = {m["r_name"] for m in models}
    create_uses = parse_create_data(CREATE_DATA_R.read_text(), model_names)

    for m in models:
        sources = [{"file": "_targets.R", "line": t["line"], "scorer": t["scorer"],
                    "metric_target": t["r_name"]} for t in metrics if t["model"] == m["r_name"]]
        sources += [{"file": "R/create_data.R", "line": u["line"], "scorer": u["scorer"],
                     "form": u["form"]} for u in create_uses if u["model"] == m["r_name"]]
        scorers = {s["scorer"] for s in sources}
        if len(scorers) > 1:
            raise ValueError(f"{m['r_name']}: conflicting scorers {scorers}")
        m["scorer"] = scorers.pop() if scorers else "none"
        m["scorer_sources"] = sources
        if m["pattern"] is not None:
            if m["pattern"] != m["outcome"]["value"]:
                raise ValueError(f"{m['r_name']}: pattern differs from outcome list")
            m["n_fits"] = len(resolved[m["pattern"]])
        else:
            if m["outcome"]["kind"] != "literal":
                raise ValueError(f"{m['r_name']}: list outcome without pattern")
            m["n_fits"] = 1

    return {
        "source": {"repository": "https://github.com/tobiaswolfram/llm_paper", "commit": COMMIT,
                   "files": ["_targets.R", "R/create_data.R"]},
        "generated_by": "scripts/extract_r_targets.py",
        "variable_lists": {n: {**e, "members": resolved[n]} for n, e in variable_lists.items()},
        "model_targets": models,
        "metric_targets": metrics,
        "totals": {"model_targets": len(models), "fits": sum(m["n_fits"] for m in models),
                   "metric_targets_in_targets_R": len(metrics),
                   "unscored_model_targets": sorted(m["r_name"] for m in models if m["scorer"] == "none")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not TARGETS_R.exists():
        print(f"{TARGETS_R} not found: clone the reference sources (docs/REFERENCE_SOURCES.md)",
              file=sys.stderr)
        return 2
    text = json.dumps(build_inventory(), indent=2) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("inventory up to date" if same else "inventory differs from the committed JSON")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
