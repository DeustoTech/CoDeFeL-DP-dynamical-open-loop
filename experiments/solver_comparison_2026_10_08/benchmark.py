"""Symmetrically calibrated, serial PB-AMD/PGD fixed-oracle comparison."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / "Control_Theoretic_DP_Numerics" / "code"
sys.path.insert(0, str(CORE))
import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion
import experiment_utils as utils
sys.path.insert(0, str(HERE.parent / "nonlinear_allocation_2026_10_08"))
import run_search as head

TASKS = ("tabular", "fashion_constant", "fashion_restart", "nonlinear_head")
METHODS = ("PB-AMD", "PGD")
GAMMAS = (.03, .1, .3, 1.)
PATHS = {"tabular": 256, "fashion_constant": 8, "fashion_restart": 8,
         "nonlinear_head": 128}
UPDATES = {"tabular": 180, "fashion_constant": 100, "fashion_restart": 100,
           "nonlinear_head": 60}
RHO = .5
POOL = ThreadPoolExecutor(max_workers=4)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), CORE / "dp_sgd_schedule_compare.py",
             CORE / "fashion_mnist_mlp_schedule_compare.py", CORE / "experiment_utils.py",
             HERE.parent / "nonlinear_allocation_2026_10_08" / "run_search.py",
             HERE.parent / "neural_mechanism_2026_10_08" / "run_study.py",
             HERE.parent / "neural_mechanism_2026_10_08" / "run_confirmation.py",
             HERE.parent / "neural_mechanism_2026_10_08" / "run_regularized.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def gaussian(task, seed, shape, role=1):
    return np.random.default_rng(np.random.SeedSequence(
        [20261008, TASKS.index(task), seed, role])).normal(size=shape)


@dataclass
class Context:
    task: str
    seed: int
    problem: object
    xi: np.ndarray

    def serial_fg(self, q, xi=None):
        xi = self.xi if xi is None else xi
        if self.task == "tabular":
            return tabular.objective_and_grad(self.problem, q, xi)
        return fashion.objective_and_grad(self.problem, q, xi,
                                          self.problem.theta_design_ref[0])

    def fg(self, q):
        if self.task == "tabular":
            return self.serial_fg(q)
        chunks = np.array_split(self.xi, min(4, self.xi.shape[1]), axis=1)
        futures = [POOL.submit(self.serial_fg, q, xi) for xi in chunks]
        results = [f.result() for f in futures]
        weights = [xi.shape[1] / self.xi.shape[1] for xi in chunks]
        return (sum(w * r[0] for w, r in zip(weights, results)),
                sum(w * r[1] for w, r in zip(weights, results)))

    def objective(self, q):
        if self.task == "tabular":
            return tabular.objective_only(self.problem, q, self.xi)
        return fashion.objective_only(self.problem, q, self.xi,
                                      self.problem.theta_design_ref[0])

    @property
    def uniform(self):
        return np.full(self.problem.T, self.problem.rho_max / self.problem.T)


def make_context(task, seed, paths=None):
    if task == "tabular":
        p = tabular.make_problem(seed, rho_max=RHO)
        dimension = p.d
    elif task == "nonlinear_head":
        setting = head.Setting("train_fast384", eta=1., batch=384)
        p = head.make_problem(setting, seed, RHO)
        dimension = p.shape.p
        head.public_features.cache_clear()
    else:
        p = fashion.make_problem(ROOT / "Control_Theoretic_DP_Numerics/data/fashion_mnist",
                                 seed=seed, rho_max=RHO,
                                 scenario="standard" if task == "fashion_constant" else "lr_restart")
        dimension = p.shape.p
    xi = gaussian(task, seed, (p.T, paths or PATHS[task], dimension))
    return Context(task, seed, p, xi)


def update(q, gradient, method, alpha, rho, floor):
    if method == "PB-AMD":
        return utils.floor_aware_entropy_step(q, gradient, alpha, rho, floor)[0]
    if method == "PGD":
        return utils.project_lower_bounded_simplex(q - alpha * gradient, rho, floor)
    raise ValueError(method)


def solve(ctx, method, gamma, updates, tolerance=None):
    """One initial adjoint plus one adjoint at each updated point; no line search."""
    p = ctx.problem
    q = ctx.uniform
    trajectory = []
    gc.collect()
    start = time.perf_counter()
    alpha = initial = None
    for k in range(updates + 1):
        value, gradient = ctx.fg(q)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(f"nonfinite oracle in {ctx.task} {method} k={k}")
        if k == 0:
            initial = float(value)
            assert initial > 0
            alpha = gamma * p.rho_max / initial
            if method == "PGD":
                alpha *= p.rho_max / p.T
        gap = utils.variational_gap(q, gradient, p.rho_max, p.q_min)
        kkt = utils.active_kkt_residual(q, gradient, p.q_min)
        error = abs(float(q.sum()) - p.rho_max)
        assert error < 1e-12 and q.min() >= p.q_min - 1e-14
        trajectory.append({"iteration": k, "gradient_evaluations": k + 1,
            "forward_only_evaluations": 0, "objective": float(value),
            "relative_objective": float(value / initial), "gap": float(gap),
            "relative_gap": float(gap / initial), "kkt": float(kkt),
            "scaled_kkt": float(p.rho_max * kkt / initial),
            "budget_error": error, "min_q": float(q.min()),
            "q": q.tolist(), "gradient": gradient.tolist(),
            "seconds": time.perf_counter() - start})
        if tolerance is not None and gap / initial <= tolerance:
            break
        if k != updates:
            q = update(q, gradient, method, alpha, p.rho_max, p.q_min)
    return {"task": ctx.task, "seed": ctx.seed, "method": method,
            "gamma": gamma, "alpha": alpha, "rho": p.rho_max,
            "q_min": p.q_min, "T": p.T, "paths": ctx.xi.shape[1],
            "max_updates": updates, "updates": len(trajectory) - 1,
            "stopping_tolerance": tolerance,
            "source_sha256": sources(), "trace": trajectory}


def seeds(task, calibration):
    base = 210000 + 1000 * TASKS.index(task)
    return list(range(base + (1 if calibration else 101), base + (4 if calibration else 111)))


def protocol():
    return {"version": 1, "rho": RHO, "tasks": list(TASKS), "paths": PATHS,
            "updates": UPDATES, "calibration_updates": 30,
            "gamma_grid": list(GAMMAS), "calibration_seeds": {t: seeds(t, True) for t in TASKS},
            "test_seeds": {t: seeds(t, False) for t in TASKS},
            "step_scaling": "PB alpha=gamma*rho/F(q0); PGD alpha=gamma*rho^2/(T*F(q0))",
            "scaling_reason": "Equal infinitesimal initial displacement at the uniform point.",
            "selection": "Minimum geometric mean final relative variational gap across three calibration problems; tie by smaller gamma.",
            "oracle": "Same fixed SAA objective/paths/initial point for both methods. No line search. K+1 gradients, K updates.",
            "timing": "Solvers run serially in alternating order. Four fixed Monte Carlo path workers inside each neural gradient, one BLAS thread each; tabular vectorized in one thread. Data/setup/warmup excluded; identical diagnostics included; no private evaluation.",
            "test_stopping": "First relative variational gap <=1e-6, or the common per-task update cap. Calibration always uses exactly 30 updates.",
            "thresholds_relative_gap": [1e-2, 1e-3, 1e-4, 1e-5],
            "seed_scheme": "SeedSequence([20261008, task_index, problem_seed, role=1]); calibration/test IDs disjoint.",
            "source_sha256": sources(), "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform()}


def ensure_protocol(out):
    expected = protocol()
    path = out / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected, "protocol/source mismatch"
    else:
        save(path, expected)


def case_path(out, phase, task, seed, method, gamma):
    return out / phase / f"{task}_{seed}_{method}_gamma{gamma:g}.json"


def calibrate(out, tasks):
    for task in tasks:
        for i, seed in enumerate(seeds(task, True)):
            jobs = [(g, m) for g in GAMMAS for m in METHODS]
            if i % 2:
                jobs.reverse()
            if all(case_path(out, "calibration", task, seed, m, g).exists() for g, m in jobs):
                continue
            print(f"BUILD calibration {task} seed={seed}", flush=True)
            ctx = make_context(task, seed)
            ctx.fg(ctx.uniform)
            for gamma, method in jobs:
                path = case_path(out, "calibration", task, seed, method, gamma)
                if path.exists():
                    continue
                result = solve(ctx, method, gamma, 30)
                save(path, result)
                final = result["trace"][-1]
                print(f"CAL {task} {seed} {method} gamma={gamma:g} gap={final['relative_gap']:.4g} time={final['seconds']:.1f}s", flush=True)
            del ctx
        selection = {}
        for method in METHODS:
            scores = []
            for gamma in GAMMAS:
                values = [json.loads(case_path(out, "calibration", task, seed, method, gamma).read_text())["trace"][-1]["relative_gap"]
                          for seed in seeds(task, True)]
                scores.append({"gamma": gamma, "relative_gaps": values,
                               "log_score": float(np.mean(np.log(np.maximum(values, 1e-16))))})
            selected = min(scores, key=lambda x: (x["log_score"], x["gamma"]))["gamma"]
            selection[method] = {"selected_gamma": selected, "candidates": scores}
        save(out / "selection" / f"{task}.json", selection)
        print(f"SELECT {task}: { {m: selection[m]['selected_gamma'] for m in METHODS} }", flush=True)


def test(out, tasks):
    for task in tasks:
        selection = json.loads((out / "selection" / f"{task}.json").read_text())
        for i, seed in enumerate(seeds(task, False)):
            order = METHODS if i % 2 == 0 else METHODS[::-1]
            jobs = [(selection[m]["selected_gamma"], m) for m in order]
            if all(case_path(out, "test", task, seed, m, g).exists() for g, m in jobs):
                continue
            print(f"BUILD test {task} seed={seed}", flush=True)
            ctx = make_context(task, seed)
            ctx.fg(ctx.uniform)
            for gamma, method in jobs:
                path = case_path(out, "test", task, seed, method, gamma)
                if path.exists():
                    continue
                result = solve(ctx, method, gamma, UPDATES[task], tolerance=1e-6)
                result["selection_sha256"] = digest(out / "selection" / f"{task}.json")
                result["order_position"] = list(order).index(method)
                save(path, result)
                final = result["trace"][-1]
                print(f"TEST {task} {seed} {method} gap={final['relative_gap']:.4g} time={final['seconds']:.1f}s", flush=True)
            del ctx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "calibrate", "test", "all"))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    assert all(task in TASKS for task in tasks)
    if args.phase == "pilot":
        for task in tasks:
            ctx = make_context(task, 219001 + TASKS.index(task) * 1000)
            ctx.fg(ctx.uniform)
            for method in METHODS:
                result = solve(ctx, method, .3, 2)
                print(task, method, result["trace"][-1]["seconds"], flush=True)
        return
    ensure_protocol(args.out)
    if args.phase in ("calibrate", "all"):
        calibrate(args.out, tasks)
    if args.phase in ("test", "all"):
        test(args.out, tasks)


if __name__ == "__main__":
    main()
