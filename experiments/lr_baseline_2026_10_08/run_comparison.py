"""Evaluate a learning-rate baseline against version-checked archived PB-AMD."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set these before importing NumPy, including when the program is resumed.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Control_Theoretic_DP_Numerics"
sys.path.insert(0, str(ROOT / "code"))
import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion
from experiment_utils import student_t_quantile, student_t_two_sided_pvalue, holm_adjusted_pvalues


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def lr_schedule(eta: np.ndarray, sensitivity: float, rho: float, floor: float) -> np.ndarray:
    weights = np.asarray(eta, dtype=float) * sensitivity
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)) or not 0 < len(weights) * floor < rho:
        raise ValueError("invalid allocation inputs")
    if np.all(weights == weights[0]):
        return np.full(len(weights), rho / len(weights))
    # Solve max(floor, c * eta * Delta), not an additive-floor approximation.
    lower, upper = 0.0, rho / float(weights.min())
    for _ in range(100):
        middle = (lower + upper) / 2
        if np.maximum(floor, middle * weights).sum() < rho:
            lower = middle
        else:
            upper = middle
    q = np.maximum(floor, (lower + upper) / 2 * weights)
    assert abs(q.sum() - rho) < 1e-12 and q.min() >= floor
    return q


def self_test() -> dict:
    eta = np.r_[np.full(25, .02), np.full(20, .075)]
    q = lr_schedule(eta, 3.5 / 192, .5, 1e-5)
    np.testing.assert_allclose(q[:25], .005, rtol=0, atol=1e-15)
    np.testing.assert_allclose(q[25:], .01875, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(lr_schedule(np.full(60, .18), .03125, .5, 1e-5), np.full(60, .5 / 60))
    active = lr_schedule(np.array([.001, 1., 2.]), 1., 1., .1)
    np.testing.assert_allclose(active, [.1, .3, .6], rtol=0, atol=1e-14)
    assert np.sum(eta**2 / q) < np.sum(eta**2 / (.5 / 45))
    return {"status": "passed", "tests": ["restart closed form", "constant equals uniform exactly", "active floor water filling", "LQ objective decrease"],
            "identity_propagation_reduction_percent": float(100 * (1 - eta.sum()**2 / (len(eta) * np.sum(eta**2))))}


def interval(values: list[float]) -> dict:
    x = np.asarray(values)
    mean = float(x.mean())
    if len(x) < 2:
        return {"mean": mean, "low": None, "high": None, "pvalue": None}
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = student_t_quantile(.975, len(x) - 1) * se
    p = student_t_two_sided_pvalue(mean / se, len(x) - 1) if se else float(mean == 0)
    return {"mean": mean, "low": mean - half, "high": mean + half, "pvalue": p}


def summarize(cases: list[dict], out: Path) -> None:
    raw = [case["comparison"] for case in cases]
    write_csv(out / "paired_seed_results.csv", raw)
    summary = []
    for task, rho in sorted({(r["task"], r["rho"]) for r in raw}):
        group = [r for r in raw if (r["task"], r["rho"]) == (task, rho)]
        item = {"task": task, "rho": rho, "n_seeds": len(group)}
        for metric in ("uniform_error", "lr_error", "pb_error", "lr_vs_uniform_percent", "pb_vs_uniform_percent",
                       "pb_vs_lr_percent", "pb_minus_lr_error", "pb_minus_lr_accuracy_pp", "lr_vs_uniform_accuracy_pp"):
            stat = interval([r[metric] for r in group])
            item.update({f"{metric}_{key}": value for key, value in stat.items()})
        item["max_uniform_reproduction_error"] = max(r["uniform_reproduction_error"] for r in group)
        summary.append(item)
    if all(r["n_seeds"] == 10 for r in summary) and len(summary) == 9:
        adjusted = holm_adjusted_pvalues([r["pb_minus_lr_error_pvalue"] for r in summary])
        for r, value in zip(summary, adjusted):
            r["pb_minus_lr_error_holm_pvalue"] = value
    write_csv(out / "summary.csv", summary)


def run_case(task: str, seed: int, rho: float, config: dict, archived: dict) -> dict:
    start = time.perf_counter()
    floor = config["q_min"]
    if task == "tabular":
        problem = tabular.make_problem(seed, rho_max=rho, q_min=floor, clip=config["tabular_clip"])
    else:
        problem = fashion.make_problem(ROOT / "data/fashion_mnist", seed=seed,
            n_train=config["fashion_n_train"], n_test=config["fashion_n_test"], horizon=45,
            batch_size=192, hidden=config["fashion_hidden"], scenario=task.removeprefix("fashion-"),
            rho_max=rho, q_min=floor, clip=config["fashion_clip"])
    q = lr_schedule(problem.eta, problem.sensitivity, rho, floor)
    uniform = np.full(problem.T, rho / problem.T)
    identical = np.array_equal(q, uniform)
    schedules = [uniform] if identical else [q, uniform]
    if task == "tabular":
        xis = np.random.default_rng(seed + 202).normal(size=(problem.T, config["tabular_M_eval"], problem.d))
        metrics = tabular.simulate_many(problem, schedules, xis)
    else:
        metrics = fashion.simulate_eval_many(problem, schedules, problem.theta_design_ref[0],
            seed=seed + 1000, paths=config["fashion_M_eval"], chunk=16)
    lr, unif = (metrics[0], metrics[0]) if identical else metrics
    old_uniform = archived[(task, seed, rho, "Uniform")]
    pb = archived[(task, seed, rho, "PB-AMD")]
    checks = {}
    for name in ("terminal_mean", "terminal_std", "test_loss", "test_acc"):
        expected = float(old_uniform[name])
        checks[name] = abs(unif[name] - expected) / max(1., abs(expected))
        if checks[name] > 1e-10:
            raise AssertionError(f"archive mismatch {task} {seed} {rho}: {name} {unif[name]} != {expected}")
    assert float(pb["saa_kkt_residual"]) <= config["pb_kkt_tol"] * (1 + 1e-10)
    assert abs(float(pb["rho_spent"]) - rho) < 1e-10
    ue, le, pe = unif["terminal_mean"], lr["terminal_mean"], float(pb["terminal_mean"])
    comparison = {"task": task, "seed": seed, "rho": rho,
        "uniform_error": ue, "lr_error": le, "pb_error": pe,
        "lr_vs_uniform_percent": 100 * (ue - le) / ue,
        "pb_vs_uniform_percent": 100 * (ue - pe) / ue,
        "pb_vs_lr_percent": 100 * (le - pe) / le,
        "pb_minus_lr_error": pe - le,
        "uniform_accuracy": unif["test_acc"], "lr_accuracy": lr["test_acc"], "pb_accuracy": float(pb["test_acc"]),
        "pb_minus_lr_accuracy_pp": 100 * (float(pb["test_acc"]) - lr["test_acc"]),
        "lr_vs_uniform_accuracy_pp": 100 * (lr["test_acc"] - unif["test_acc"]),
        "lr_late_early_ratio": float(q[-(problem.T // 3):].sum() / q[:problem.T // 3].sum()),
        "pb_late_early_ratio": float(pb["late_over_early"]),
        "uniform_reproduction_error": max(checks.values()), "elapsed_seconds": time.perf_counter() - start}
    return {"comparison": comparison, "lr_schedule": q.tolist(), "eta": problem.eta.tolist(),
        "sensitivity": problem.sensitivity, "floor": floor, "budget_error": abs(float(q.sum()) - rho),
        "lr_equals_uniform": identical, "lr_metrics": lr, "uniform_metrics": unif,
        "archived_pb_row": pb, "uniform_reproduction_checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Pilot limit; all cases are run by default")
    args = parser.parse_args()
    config_path = ROOT / "results/comprehensive/config.json"
    raw_path = ROOT / "results/comprehensive/raw_results.csv"
    config = json.loads(config_path.read_text())
    for name, expected in config["source_sha256"].items():
        assert digest(ROOT / "code" / name) == expected, f"source mismatch: {name}"
    archived = {(r["task"], int(r["seed"]), float(r["rho"]), r["method_family"]): r for r in read_csv(raw_path)}
    manifest = {"baseline": "max(q_min,c*eta_t*Delta_hat_t), exact normalization",
        "archive_config_sha256": digest(config_path), "archive_raw_sha256": digest(raw_path),
        "runner_sha256": digest(Path(__file__)), "source_sha256": config["source_sha256"],
        "python": sys.version, "numpy": np.__version__, "sampling": "original final-evaluation seeds, path counts and chunk order",
        "pb_reuse": "archived seed-level PB-AMD metrics; verify source hashes, feasibility, KKT and independently reproduced uniform metrics",
        "inference": "paired t intervals across ten independent experiment seeds, not treating paths as independent seeds; positive PB-vs-LR percent favors PB",
        "manuscript": "local archived numerical protocol, NOT verified against latest Overleaf manuscript"}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("existing results: use --resume")
        assert json.loads(manifest_path.read_text()) == manifest, "resume manifest mismatch"
    else:
        write_json(manifest_path, manifest)
    write_json(out / "self_tests.json", self_test())
    cases = []
    jobs = []
    # Start with the pivotal restart comparison, followed by the two controls.
    for task in ("fashion-lr_restart", "tabular", "fashion-standard"):
        prefix = "tabular" if task == "tabular" else "fashion"
        for rho in map(float, config[prefix + "_rhos"].split(",")):
            for seed in map(int, config[prefix + "_seeds"].split(",")):
                jobs.append((task, seed, rho))
    if args.limit:
        jobs = jobs[:args.limit]
    for index, (task, seed, rho) in enumerate(jobs, 1):
        path = out / f"{task}_seed{seed}_rho{rho:g}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            print(f"START {index}/{len(jobs)} {task} seed={seed} rho={rho}", flush=True)
            result = run_case(task, seed, rho, config, archived)
            write_json(path, result)
        cases.append(result)
        summarize(cases, out)
        row = result["comparison"]
        print(f"DONE {index}/{len(jobs)} LR gain={row['lr_vs_uniform_percent']:.7f}% PB gain={row['pb_vs_uniform_percent']:.7f}% PB-vs-LR={row['pb_vs_lr_percent']:.7f}% time={row['elapsed_seconds']:.1f}s", flush=True)
    write_json(out / "completion.json", {"completed_cases": len(cases), "expected_cases": 90,
        "status": "complete" if len(cases) == 90 else "pilot", "all_uniform_reproductions_passed": True,
        "max_uniform_scaled_error": max(c["comparison"]["uniform_reproduction_error"] for c in cases),
        "max_budget_error": max(c["budget_error"] for c in cases)})


if __name__ == "__main__":
    main()
