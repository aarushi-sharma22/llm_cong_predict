#!/usr/bin/env python
"""Pipeline runner — the port of ``run.R`` (``targets::tar_make()``).

    python run.py                      print the plan: the wiring, what runs, what is
                                       produced elsewhere. Executes nothing.
    python run.py --run                run the whole graph with the paper configuration
                                       (10 outer folds, 5 inner folds, 6 learners)
    python run.py --run --config smoke run the SMOKE configuration; it is refused
                                       unless $LCP_DATA_ROOT holds synthetic data
    python run.py --run --targets essay_superlearner_metrics [...]   run only these
                                       targets and what they depend on

Running needs the restricted inputs under ``$LCP_DATA_ROOT`` (config.RESTRICTED_INPUTS)
and the steps that happen outside this pipeline: the RoBERTa embeddings
(``python -m llm_cong_predict.features.roberta_step``) and the koRpus readability CSV
(``r/readability.R``, Task 2.5). See docs/ORCHESTRATION.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, "src")
    from llm_cong_predict import config
    from llm_cong_predict.pipeline.build import build_pipeline

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="execute the pipeline (default: print the plan)")
    ap.add_argument("--config", choices=("paper", "smoke"), default="paper")
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="worker processes for the model fits")
    args = ap.parse_args(argv)

    pipe = build_pipeline()  # raises if the wiring is unsound (dangling deps / cycle)

    if not args.run:
        print("Pipeline wiring validated: dependencies resolve and the graph is acyclic.\n")
        print(pipe.status_report())
        print("\nSteps done outside this pipeline, whose output it reads:")
        print("  - RoBERTa embeddings: python -m llm_cong_predict.features.roberta_step")
        print("  - readability: r/readability.R (TreeTagger + koRpus), Task 2.5")
        print("  - SALAT metrics and LanguageTool spelling errors: the tools' CSVs")
        print("\nNothing was executed. Use --run (see --help).")
        return 0

    from llm_cong_predict.pipeline.execute import run_pipeline

    run_config = config.SMOKE_RUN if args.config == "smoke" else config.PAPER_RUN
    run_config = dataclasses.replace(run_config, n_jobs=args.n_jobs)
    result = run_pipeline(run_config, targets=args.targets)
    print(f"{result.run_config.name} run on {result.data_source} data: "
          f"{len(result.metrics)} metric row(s) in {result.elapsed_seconds:.0f}s"
          + (f"  [{result.label}]" if result.label else ""))
    for entry in result.not_run:
        outcome = f" [{entry['outcome']}]" if entry["outcome"] else ""
        print(f"  not run: {entry['target']}{outcome} — {entry['reason']}")
    print(f"  metrics:     {result.metrics_path}")
    print(f"  run log:     {result.log_path}")
    print(f"  predictions: {len(result.prediction_paths)} file(s) under $LCP_DATA_ROOT/fits/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
