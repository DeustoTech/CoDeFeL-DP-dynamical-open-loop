from __future__ import annotations

import os

# Fix the numerical-library thread pool before importing NumPy.  The benchmark
# itself is serial; this prevents a method from changing the amount of implicit
# parallelism available to later methods.
for thread_variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[thread_variable] = "1"

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion


FAMILIES = ["PB-AMD", "PGD grid", "Penalty grid"]
TABULAR_PGD = [3e-5, 1e-4, 3e-4, 1e-3]
FASHION_PGD = [3e-6, 1e-5, 3e-5, 1e-4]
TABULAR_PENALTY = [
    (1.0, 0.04),
    (3.0, 0.04),
    (10.0, 0.035),
    (100.0, 0.025),
    (1000.0, 0.015),
    (10000.0, 0.008),
]
FASHION_PENALTY = [
    (1.0, 0.035),
    (3.0, 0.032),
    (10.0, 0.025),
    (100.0, 0.017),
    (1000.0, 0.01),
    (10000.0, 0.006),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rotated_order(repeat: int) -> list[str]:
    offset = repeat % len(FAMILIES)
    return FAMILIES[offset:] + FAMILIES[:offset]


def retain_complete_runtime_groups(rows: list[dict]) -> list[dict]:
    """Retain only complete task-repeat blocks in their prescribed order."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["repeat"]))
        groups.setdefault(key, []).append(row)
    retained = []
    for (_, repeat), group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["order_position"]))
        if (
            len(ordered) == len(FAMILIES)
            and [row["family"] for row in ordered] == rotated_order(repeat)
            and [int(row["order_position"]) for row in ordered]
            == list(range(len(FAMILIES)))
        ):
            retained.extend(ordered)
    return retained


def repair_exact_budget(q: np.ndarray, rho: float, q_min: float) -> np.ndarray:
    residual = np.maximum(np.asarray(q, dtype=float) - q_min, 0.0)
    if float(residual.sum()) <= 0.0:
        residual = np.ones_like(residual)
    return q_min + (rho - len(q) * q_min) * residual / residual.sum()


def time_tabular_family(
    family: str,
    problem: tabular.Problem,
    xis_opt: np.ndarray,
    xis_val: np.ndarray,
    K: int,
) -> dict:
    gc.collect()
    optimization_start = time.perf_counter()
    if family == "PB-AMD":
        methods = [
            tabular.run_pb_amd(
                problem,
                xis_opt,
                alpha=0.005,
                K=K,
                record_trace=False,
            )
        ]
    elif family == "PGD grid":
        methods = [
            tabular.run_projected_gradient(
                problem,
                xis_opt,
                alpha=alpha,
                K=K,
                record_trace=False,
            )
            for alpha in TABULAR_PGD
        ]
    elif family == "Penalty grid":
        methods = [
            tabular.run_penalty(
                problem,
                xis_opt,
                beta=beta,
                lr=lr,
                K=K,
                compute_diagnostics=False,
                record_trace=False,
            )
            for beta, lr in TABULAR_PENALTY
        ]
    else:
        raise ValueError(f"unknown family: {family}")
    optimization_wall = time.perf_counter() - optimization_start

    validation_start = time.perf_counter()
    validation_schedules = [method["q"] for method in methods]
    if family == "Penalty grid":
        validation_schedules = [
            repair_exact_budget(q, problem.rho_max, problem.q_min)
            for q in validation_schedules
        ]
    validation = [
        tabular.objective_only(problem, q, xis_val)
        for q in validation_schedules
    ]
    validation_wall = time.perf_counter() - validation_start
    selected = int(np.argmin(validation))
    solver_time = sum(float(method["time_sec"]) for method in methods)
    return {
        "candidates": len(methods),
        "solver_time_sec": solver_time,
        "optimization_wall_time_sec": optimization_wall,
        "validation_time_sec": validation_wall,
        "pipeline_time_sec": solver_time + validation_wall,
        "selected_method": methods[selected]["method"],
        "selected_validation_objective": float(validation[selected]),
        "selected_rho_spent": float(np.sum(validation_schedules[selected])),
    }


def time_fashion_family(
    family: str,
    problem: fashion.Problem,
    theta0: np.ndarray,
    xis_opt: np.ndarray,
    xis_val: np.ndarray,
    K: int,
    pb_alpha: float,
) -> dict:
    gc.collect()
    optimization_start = time.perf_counter()
    if family == "PB-AMD":
        methods = [
            fashion.run_pb(
                problem,
                theta0,
                xis_opt,
                K=K,
                alpha=pb_alpha,
                record_trace=False,
            )
        ]
    elif family == "PGD grid":
        methods = [
            fashion.run_projected_gradient(
                problem,
                theta0,
                xis_opt,
                K=K,
                alpha=alpha,
                record_trace=False,
            )
            for alpha in FASHION_PGD
        ]
    elif family == "Penalty grid":
        methods = [
            fashion.run_penalty(
                problem,
                theta0,
                xis_opt,
                K=K,
                beta=beta,
                lr=lr,
                compute_diagnostics=False,
                record_trace=False,
            )
            for beta, lr in FASHION_PENALTY
        ]
    else:
        raise ValueError(f"unknown family: {family}")
    optimization_wall = time.perf_counter() - optimization_start

    validation_start = time.perf_counter()
    validation_schedules = [method["q"] for method in methods]
    if family == "Penalty grid":
        validation_schedules = [
            repair_exact_budget(q, problem.rho_max, problem.q_min)
            for q in validation_schedules
        ]
    validation = [
        fashion.objective_only(problem, q, xis_val, theta0)
        for q in validation_schedules
    ]
    validation_wall = time.perf_counter() - validation_start
    selected = int(np.argmin(validation))
    solver_time = sum(float(method["time_sec"]) for method in methods)
    return {
        "candidates": len(methods),
        "solver_time_sec": solver_time,
        "optimization_wall_time_sec": optimization_wall,
        "validation_time_sec": validation_wall,
        "pipeline_time_sec": solver_time + validation_wall,
        "selected_method": methods[selected]["method"],
        "selected_validation_objective": float(validation[selected]),
        "selected_rho_spent": float(np.sum(validation_schedules[selected])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/submission_v4_controlled_runtime")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", default="tabular,fashion-standard,fashion-lr_restart")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tabular-K", type=int, default=180)
    parser.add_argument("--fashion-K", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "runtime_runs.csv"
    config_path = out_dir / "config.json"
    source_files = [
        Path(__file__).resolve(),
        Path(tabular.__file__).resolve(),
        Path(fashion.__file__).resolve(),
        Path(__file__).resolve().with_name("experiment_utils.py"),
    ]
    config = {
        key: value for key, value in vars(args).items() if key != "resume"
    } | {
        "protocol_version": 4,
        "execution": "single Python process; one method family at a time",
        "thread_environment": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            ]
        },
        "family_order": [rotated_order(repeat) for repeat in range(args.repeats)],
        "checkpoint_policy": "only complete task-repeat blocks are retained on explicit resume",
        "source_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in source_files
        },
        "timed_pipeline": "the K adjoint-gradient updates, one final fixed-path forward objective check, and one independent public-validation forward evaluation per candidate; common problem construction and warm-up, iteration-trace diagnostics, post-run derivative diagnostics, and private evaluation are excluded; penalty candidates are repaired before validation",
        "private_evaluation_included": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    if args.resume and raw_path.exists():
        if not config_path.exists():
            raise ValueError("cannot resume controlled timing without its protocol config")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        protocol_keys = [
            "data_dir",
            "tasks",
            "rho",
            "tabular_K",
            "fashion_K",
            "repeats",
            "protocol_version",
            "thread_environment",
            "family_order",
            "timed_pipeline",
            "python",
            "numpy",
            "platform",
            "source_sha256",
        ]
        mismatched = [key for key in protocol_keys if previous.get(key) != config.get(key)]
        if mismatched:
            raise ValueError(
                "controlled timing resume protocol mismatch: " + ", ".join(mismatched)
            )
        rows = retain_complete_runtime_groups(read_rows(raw_path))
    else:
        rows = []
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    completed = {
        (row["task"], int(row["repeat"]), row["family"])
        for row in rows
    }
    tasks = [task for task in args.tasks.split(",") if task]

    # Construct each public design problem and warm both the adjoint and
    # forward-validation code paths before any timed family is run.
    contexts: dict[str, dict] = {}
    if "tabular" in tasks:
        problem = tabular.make_problem(17, rho_max=args.rho)
        xis_opt = np.random.default_rng(118).normal(size=(problem.T, 256, problem.d))
        xis_val = np.random.default_rng(168).normal(size=(problem.T, 512, problem.d))
        uniform = np.full(problem.T, args.rho / problem.T)
        tabular.objective_and_grad(problem, uniform, xis_opt[:, :1])
        tabular.objective_only(problem, uniform, xis_val[:, :1])
        contexts["tabular"] = {
            "problem": problem,
            "xis_opt": xis_opt,
            "xis_val": xis_val,
            "K": args.tabular_K,
        }

    if any(task.startswith("fashion-") for task in tasks):
        fashion.load_preprocessed_fashion_mnist(str(Path(args.data_dir).resolve()))
    for task, scenario, alpha in [
        ("fashion-standard", "standard", 0.0005),
        ("fashion-lr_restart", "lr_restart", 0.005),
    ]:
        if task not in tasks:
            continue
        problem = fashion.make_problem(
            Path(args.data_dir),
            seed=123,
            scenario=scenario,
            rho_max=args.rho,
        )
        theta0 = problem.theta_design_ref[0]
        xis_opt = np.random.default_rng(200).normal(
            size=(problem.T, 8, problem.shape.p)
        )
        xis_val = np.random.default_rng(300).normal(
            size=(problem.T, 48, problem.shape.p)
        )
        uniform = np.full(problem.T, args.rho / problem.T)
        fashion.objective_and_grad(problem, uniform, xis_opt[:, :1], theta0)
        fashion.objective_only(problem, uniform, xis_val[:, :1], theta0)
        contexts[task] = {
            "problem": problem,
            "theta0": theta0,
            "xis_opt": xis_opt,
            "xis_val": xis_val,
            "K": args.fashion_K,
            "pb_alpha": alpha,
        }

    for task in tasks:
        context = contexts[task]
        for repeat in range(args.repeats):
            for position, family in enumerate(rotated_order(repeat)):
                if (task, repeat, family) in completed:
                    continue
                print(f"[{task}] repeat={repeat} family={family}", flush=True)
                if task == "tabular":
                    result = time_tabular_family(family, **context)
                else:
                    result = time_fashion_family(family, **context)
                row = {
                    "task": task,
                    "repeat": repeat,
                    "order_position": position,
                    "family": family,
                    "K": context["K"],
                    "m_opt": context["xis_opt"].shape[1],
                    "m_val": context["xis_val"].shape[1],
                    "adjoint_iterations": int(result["candidates"]) * context["K"],
                    **result,
                }
                rows.append(row)
                write_csv(raw_path, rows)

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(raw_path)


if __name__ == "__main__":
    main()
