"""Reproduce the two final head cohorts without historical screening results.

Calls the original design, geometric-refinement, evaluation and inference code.
Only orchestration and the final-cohort audit are new. No historical screening
run is claimed to have been reproduced by this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "experiments/nonlinear_allocation_2026_10_08"
COHORTS = [("confirmation", .5, list(range(96001, 96011))),
           ("budget_extension", .25, list(range(106001, 106011)))]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(program: str, *args) -> None:
    subprocess.run([sys.executable, str(STUDY / program), *map(str, args)],
                   cwd=ROOT, check=True)


def evaluate(directory, seed_ids, rho):
    # Importing the fast driver installs exactly the original extended settings
    # and original source-hash chain before any case is loaded.
    import run_fast_head as fast
    import independent_evaluation as final
    r, s = fast.r, fast.r.s
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert list(map(int, manifest["seeds"].split(","))) == seed_ids
    assert manifest["settings"] == "train_fast384" and float(manifest["rhos"]) == rho
    assert [manifest[k] for k in ["opt_paths", "val_paths", "eval_paths", "iterations"]] == [128, 256, 512, 20]
    assert manifest["source_sha256"] == r.source_hashes()
    final.DIRECTORY, final.OUT = directory, directory / "independent_evaluation"
    protocol = {
        "scope": "Fresh final-cohort reproduction; original evaluator, no historical screening prerequisite",
        "orchestrator_sha256": digest(Path(__file__)),
        "evaluator_sha256": digest(Path(final.__file__)),
        "public_seed_offset": 8_000_000, "private_seed_offset": 9_000_000,
        "paths": 512, "all_schedules_unchanged": True,
        "cohort_seeds": seed_ids, "rho": rho,
    }
    protocol_path = final.OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol, "Evaluation protocol mismatch"
    else:
        s.save(protocol_path, protocol)
    paths = [directory / f"train_fast384_seed{seed}_rho{rho:g}.json" for seed in seed_ids]
    assert all(p.is_file() for p in paths), "Design must finish before final evaluation"
    for path in paths:
        final.evaluate_case(path, manifest)
    # Enumerate the actual namespaces used in both reproduced cohorts.
    all_seeds = [seed for _, _, seeds in COHORTS for seed in seeds]
    design_rng = {seed + offset for seed in all_seeds
                  for offset in [0, 5000, 180000, 181000, 82000,
                                 184000, 185000, 186000, 187000, 188000]}
    fresh_rng = [seed + offset for seed in all_seeds for offset in [8_000_000, 9_000_000]]
    assert len(fresh_rng) == len(set(fresh_rng)) and not (set(fresh_rng) & design_rng)
    rows = [json.loads((final.OUT / p.name).read_text(encoding="utf-8")) for p in paths]
    s.save(final.OUT / "verification.json", {
        "status": "passed", "cases": len(rows),
        "new_rng_namespaces_disjoint": True,
        "all_private_schedules_fixed_before_evaluation": True,
        "max_replay_error": max(x["max_independent_replay_error"] for x in rows),
        "case_sha256": {p.name: digest(final.OUT / p.name) for p in paths},
        "scope": "Both final cohorts; historical public screens were not rerun",
    })
    final.summarize(directory, independent=True)


def combine():
    import run_fast_head as fast
    from experiment_utils import holm_adjusted_pvalues
    np, s, r = fast.r.np, fast.r.s, fast.r
    comparisons, cases = [], []
    for dirname, rho, seeds in COHORTS:
        directory = STUDY / dirname
        summary = json.loads((directory / "independent_summary.json").read_text(encoding="utf-8"))
        assert summary["cases"] == 10 and summary["independent_evaluation"]
        assert summary["source_sha256"] == r.source_hashes()
        for name, sha in summary["evaluation_sha256"].items():
            path = directory / "independent_evaluation" / name
            assert digest(path) == sha
            case = json.loads(path.read_text(encoding="utf-8"))
            assert digest(directory / name) == case["case_sha256"]
            assert digest(directory / "refined_geometric" / name) == case["refined_sha256"]
            cases.append(case)
        for row in summary["comparisons"]:
            values = row["seed_level_gains"]
            comparisons.append(dict(row, study=dirname,
                median_error_reduction_percent=float(np.median(values)),
                minimum_error_reduction_percent=min(values),
                maximum_error_reduction_percent=max(values)))
        assert json.loads((directory / "verification.json").read_text(encoding="utf-8"))["status"] == "passed"
        assert json.loads((directory / "independent_evaluation/verification.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert len(cases) == 20 and len(comparisons) == 8
    for row, adjusted in zip(comparisons, holm_adjusted_pvalues([x["pvalue"] for x in comparisons]), strict=True):
        row["joint_holm_pvalue_8_tests"] = adjusted
    s.save(STUDY / "combined_summary.json", {
        "status": "verified", "cases": 20, "comparisons": comparisons,
        "orchestrator_sha256": digest(Path(__file__)),
        "scope": "Only the two final cohorts; no historical exploratory conclusions are regenerated",
    })
    s.save(STUDY / "final_audit.json", {
        "status": "passed", "confirmatory_cases": 20,
        "historical_public_screens_rerun": False,
        "max_final_replay_error": max(x["max_independent_replay_error"] for x in cases),
        "combined_summary_sha256": digest(STUDY / "combined_summary.json"),
        "scope": "Input hashes, final-cohort replay and independent evaluation; not the original 37-case historical audit",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    sys.path.insert(0, str(STUDY))
    for dirname, rho, seeds in COHORTS:
        directory = STUDY / dirname
        call("run_fast_head.py", "--settings", "train_fast384",
             "--seeds", ",".join(map(str, seeds)), "--rhos", rho,
             "--opt-paths", 128, "--val-paths", 256, "--eval-paths", 512,
             "--iterations", 20, "--workers", args.workers,
             "--phase", "confirmation", "--out", directory)
        call("refine_fast.py", directory, "--workers", args.workers)
        evaluate(directory, seeds, rho)
        call("test_and_verify.py", directory, "--replay", "--workers", args.workers)
        call("mechanism_diagnostics.py", directory, "--independent")
    call("conditional_diagnostics.py")
    combine()


if __name__ == "__main__":
    main()
