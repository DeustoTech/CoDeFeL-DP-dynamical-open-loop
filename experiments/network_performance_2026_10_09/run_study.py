"""Recover frozen schedules and evaluate individual-network predictive utility."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / "Control_Theoretic_DP_Numerics"
sys.path.insert(0, str(BUNDLE / "code"))
import fashion_mnist_mlp_schedule_compare as f
from experiment_utils import active_kkt_residual

sys.path.insert(0, str(ROOT / "experiments/lr_baseline_2026_10_08"))
from run_comparison import lr_schedule

RESULTS = HERE / "results"
CONFIG = BUNDLE / "results/comprehensive/config.json"
ARCHIVE = BUNDLE / "results/comprehensive/raw_results.csv"
METHODS = ["PB-AMD", "Uniform", "LR"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def cases():
    c = json.loads(CONFIG.read_text())
    return [(scenario, int(seed), float(rho))
            for scenario in c["fashion_scenarios"].split(",")
            for rho in c["fashion_rhos"].split(",")
            for seed in c["fashion_seeds"].split(",")]


def stem(case):
    scenario, seed, rho = case
    return f"{scenario}_seed{seed}_rho{rho:g}"


def protocol():
    c = json.loads(CONFIG.read_text())
    for name, expected in c["source_sha256"].items():
        assert sha(BUNDLE / "code" / name) == expected, name
    paths = [CONFIG, ARCHIVE, Path(__file__),
             ROOT / "experiments/lr_baseline_2026_10_08/run_comparison.py"]
    paths += sorted((BUNDLE / "data/fashion_mnist").glob("*.gz"))
    paths += [BUNDLE / "code" / name for name in c["source_sha256"]]
    return {
        "purpose": "Predictive performance of all-parameter network training at unchanged settings",
        "cases": cases(), "methods": METHODS + ["No noise"],
        "paths": 256, "chunk": 16, "evaluation_seed": "problem seed + 1000",
        "schedule_recovery": "Original PB-AMD source, stepsize, fixed design paths and KKT return rule; no retuning",
        "reference": "Same clipped, regularized SGD, data, batches, initialization and learning rates; remove Gaussian noise only",
        "metrics": "Accuracy and unregularized test cross-entropy of each individual trained network; no ensembling",
        "inference": "Ten seed-level means; paired t intervals for differences; paths are not independent problem seeds",
        "all_results": "Both original full-network scenarios and all three budgets; no outcome-based exclusions",
        "replay_indices": [0, 127, 255],
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "python": sys.version, "numpy": np.__version__,
    }


def make_problem(case):
    scenario, seed, rho = case
    c = json.loads(CONFIG.read_text())
    return f.make_problem(BUNDLE / "data/fashion_mnist", seed=seed,
        n_train=c["fashion_n_train"], n_test=c["fashion_n_test"], horizon=45,
        batch_size=192, hidden=c["fashion_hidden"], scenario=scenario,
        rho_max=rho, q_min=c["q_min"], clip=c["fashion_clip"])


def archive_rows(case):
    scenario, seed, rho = case
    with ARCHIVE.open(newline="") as handle:
        return {r["method_family"]: r for r in csv.DictReader(handle)
                if r["task"] == f"fashion-{scenario}" and int(r["seed"]) == seed
                and float(r["rho"]) == rho}


def close(actual, expected, name, tol=1e-10):
    error = abs(float(actual) - float(expected)) / max(1., abs(float(expected)))
    if error > tol:
        raise AssertionError(f"{name}: {actual} != {expected}, scaled error {error}")
    return error


def recover_schedule(problem, case, locked):
    path = RESULTS / "schedules" / (stem(case) + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved["protocol_sha256"] == locked
        return saved
    scenario, seed, rho = case
    c = json.loads(CONFIG.read_text())
    xis = np.random.default_rng(seed + 77).normal(size=(45, 8, problem.shape.p))
    alpha = c["fashion_standard_alpha" if scenario == "standard" else "fashion_restart_alpha"]
    pb = f.run_pb(problem, problem.theta_design_ref[0], xis, K=c["fashion_K"],
                  alpha=alpha, record_trace=False, kkt_tolerance=c["pb_kkt_tol"],
                  max_iterations=c["fashion_pb_max_K"])
    assert pb["converged"]
    archived = archive_rows(case)["PB-AMD"]
    close(pb["design_objective"], archived["design_objective"], "design objective")
    assert pb["iterations"] == int(float(archived["iterations"]))
    valpaths = np.random.default_rng(seed + 177).normal(size=(45, 48, problem.shape.p))
    val = f.objective_only(problem, pb["q"], valpaths, problem.theta_design_ref[0])
    close(val, archived["validation_objective"], "validation objective")
    q_lr = lr_schedule(problem.eta, problem.sensitivity, rho, problem.q_min)
    schedules = {"PB-AMD": pb["q"].tolist(), "Uniform": np.full(45, rho / 45).tolist(),
                 "LR": q_lr.tolist()}
    for q in schedules.values():
        assert min(q) >= problem.q_min and abs(sum(q) - rho) < 1e-12
    saved = {"protocol_sha256": locked, "case": case, "schedules": schedules,
             "design_objective": pb["design_objective"], "validation_objective": val,
             "iterations": pb["iterations"], "kkt_residual": pb["certified_kkt_residual"]}
    save(path, saved)
    return saved


def evaluate(problem, schedules, seed, paths=256, chunk=16):
    names = list(schedules)
    q = np.array(list(schedules.values()))
    rng = np.random.default_rng(seed)
    values = {k: np.empty((len(names), paths)) for k in ("accuracy", "cross_entropy", "regularized_loss", "tracking")}
    for start in range(0, paths, chunk):
        m = min(chunk, paths - start)
        theta = np.broadcast_to(problem.theta_design_ref[0], (len(names), m, problem.shape.p)).copy()
        for t, idx in enumerate(problem.batches):
            grad = f.mlp_grad(theta.reshape(-1, problem.shape.p), problem.x_train[idx],
                problem.y_train[idx], problem.shape, problem.l2, problem.clip,
                problem.active_masks[t]).reshape(theta.shape)
            innovation = problem.active_masks[t][None, :] * rng.normal(size=(m, problem.shape.p))
            amp = problem.sensitivity * problem.eta[t] / np.sqrt(2 * q[:, t])
            theta -= problem.eta[t] * grad + amp[:, None, None] * innovation[None, :, :]
        flat = theta.reshape(-1, problem.shape.p)
        ce, acc = f.mlp_loss_acc(flat, problem.x_test, problem.y_test, problem.shape, 0.)
        values["accuracy"][:, start:start+m] = acc.reshape(len(names), m)
        values["cross_entropy"][:, start:start+m] = ce.reshape(len(names), m)
        values["regularized_loss"][:, start:start+m] = (ce + .5 * problem.l2 * (flat**2).sum(axis=1)).reshape(len(names), m)
        values["tracking"][:, start:start+m] = ((theta - problem.theta_ref[-1])**2).sum(axis=2)
    return {name: {key: arr[j].tolist() for key, arr in values.items()} for j, name in enumerate(names)}


def run_case(case):
    begun = time.perf_counter()
    locked = sha(RESULTS / "protocol.json")
    out = RESULTS / "cases" / (stem(case) + ".json")
    if out.exists():
        saved = json.loads(out.read_text())
        assert saved["protocol_sha256"] == locked
        return stem(case), "cached"
    print("START", stem(case), flush=True)
    problem = make_problem(case)
    sched = recover_schedule(problem, case, locked)
    evaluated = evaluate(problem, sched["schedules"], case[1] + 1000)
    archived = archive_rows(case)
    checks = {}
    for method in ("PB-AMD", "Uniform"):
        v = evaluated[method]
        for new, old in (("accuracy", "test_acc"), ("regularized_loss", "test_loss"), ("tracking", "terminal_mean")):
            checks[f"{method}:{old}"] = close(np.mean(v[new]), archived[method][old], method + old)
        checks[f"{method}:terminal_std"] = close(np.std(v["tracking"]), archived[method]["terminal_std"], method + " std")
    lrpath = ROOT / "experiments/lr_baseline_2026_10_08/results" / ("fashion-" + stem(case) + ".json")
    old_lr = json.loads(lrpath.read_text())["lr_metrics"]
    for new, old in (("accuracy", "test_acc"), ("regularized_loss", "test_loss"), ("tracking", "terminal_mean")):
        checks["LR:" + old] = close(np.mean(evaluated["LR"][new]), old_lr[old], "LR " + old)
    ref_ce, ref_acc = f.mlp_loss_acc(problem.theta_ref[-1][None], problem.x_test, problem.y_test, problem.shape, 0.)
    reference = {"accuracy": float(ref_acc[0]), "cross_entropy": float(ref_ce[0]),
                 "regularized_loss": float(ref_ce[0] + .5 * problem.l2 * np.sum(problem.theta_ref[-1]**2))}
    checks["reference:accuracy"] = close(reference["accuracy"], archived["PB-AMD"]["reference_test_acc"], "reference accuracy")
    checks["reference:loss"] = close(reference["regularized_loss"], archived["PB-AMD"]["reference_test_loss"], "reference loss")
    result = {"protocol_sha256": locked, "case": case, "metrics": evaluated, "no_noise": reference,
              "archive_checks": checks, "schedule_sha256": sha(RESULTS / "schedules" / (stem(case) + ".json")),
              "lr_archive_sha256": sha(lrpath), "seconds": time.perf_counter() - begun}
    save(out, result)
    print("DONE", stem(case), f"{result['seconds']:.1f}s", "max mismatch", max(checks.values()), flush=True)
    return stem(case), "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    desired = protocol()
    lock = RESULTS / "protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == json.loads(json.dumps(desired)), "protocol mismatch"
    else:
        save(lock, desired)
    jobs = cases()[:args.limit or None]
    if args.workers == 1:
        for case in jobs:
            run_case(case)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case) for case in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)


if __name__ == "__main__":
    main()
