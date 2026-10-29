"""Verify source integrity; optionally run small offline checks in a temporary copy."""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def tracked_files() -> list[Path]:
    paths = []
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Source integrity check failed: {relative}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run data-free scientific checks in a temporary source copy")
    args = parser.parse_args()
    paths = tracked_files()
    scripts = [p for p in paths if p.suffix == ".py"]
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Source checksums verified for {len(paths)} files; {len(scripts)} Python files parsed.", flush=True)
    if not args.smoke:
        return
    os.environ["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory(prefix="dp_sgd_code_check_") as directory:
        work = Path(directory).resolve() / "code with spaces 测试"
        work.mkdir()
        for path in paths:
            target = work / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        solver = work / "experiments/solver_comparison_2026_10_08"
        lr = work / "experiments/lr_baseline_2026_10_08"
        jobs = [
            ("Portable entry points and README command contracts",
             ["-m", "unittest", "discover", "-s", str(work / "tests"), "-p", "test_portability.py", "-v"]),
            ("Simplex gap, PB-AMD/PGD updates and LQ bookkeeping",
             ["-m", "unittest", "discover", "-s", str(solver), "-p", "test_benchmark.py", "-v"]),
            ("Armijo toy problem and line-search bookkeeping",
             ["-c", f"import sys; sys.path.insert(0, {str(solver)!r}); import test_armijo; test_armijo.toy_check()"]),
            ("Full-network prediction metrics", [str(work / "experiments/network_performance_2026_10_09/test_study.py")]),
            ("Seed-level intervals and Holm adjustment", [str(work / "experiments/network_performance_2026_10_09/test_statistics.py")]),
            ("Head forward metrics and Gaussian chunk indexing", [str(work / "experiments/nonlinear_allocation_2026_10_08/test_metrics.py")]),
            ("Head VJP, adjoint, linearization and refinement checks", [str(work / "experiments/nonlinear_allocation_2026_10_08/test_and_verify.py")]),
            ("LR allocation and active-floor checks",
             ["-c", f"import sys; sys.path.insert(0, {str(lr)!r}); import run_comparison; print(run_comparison.self_test())"]),
        ]
        for label, command in jobs:
            print(f"\nChecking: {label}", flush=True)
            subprocess.run([sys.executable, "-B", *command], cwd=work, check=True, timeout=60)
        print("\nOffline checks passed. No dataset was downloaded and no full experiment was rerun.")


if __name__ == "__main__":
    main()
