from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion
from experiment_utils import (
    active_kkt_residual,
    floor_diagnostics,
    holm_adjusted_pvalues,
    mirror_residual,
    privacy_epsilon,
    projected_gradient_residual,
    student_t_quantile,
    student_t_two_sided_pvalue,
    variational_gap,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_raw_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    numeric = {
        "seed", "rho", "beta", "design_objective", "validation_objective", "terminal_mean", "terminal_std",
        "test_loss", "test_acc", "reference_test_loss", "reference_test_acc",
        "rho_spent", "rho_violation", "rho_slack", "late_over_early",
        "phase_late_over_early", "max_over_avg", "time_sec", "epsilon_target", "epsilon_actual", "saa_variational_gap",
        "saa_mirror_residual", "saa_kkt_residual", "validation_variational_gap", "floor_count", "min_floor_excess",
        "optimizer_grad_norm", "iterations", "max_prox_feasibility_error", "delta", "q_min", "lr",
        "q_min_over_uniform", "sensitivity", "clip", "alpha",
        "saa_projected_residual", "tuning_time_sec",
    }
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = int(row[key]) if key == "seed" else float(row[key])
    return rows


def retain_complete_cases(
    rows: list[dict],
    methods_per_case: int | dict[str, int] = 24,
) -> list[dict]:
    """Discard interrupted or duplicated cases before a resumed run."""
    groups: dict[tuple[str, int, float], list[dict]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["seed"]), float(row["rho"]))
        groups.setdefault(key, []).append(row)
    complete: list[dict] = []
    for key, group in groups.items():
        expected = (
            methods_per_case[key[0]]
            if isinstance(methods_per_case, dict)
            else methods_per_case
        )
        families = {str(row["method_family"]) for row in group}
        if len(group) == expected and len(families) == expected:
            complete.extend(group)
    return complete


def shaped_schedule(T: int, rho: float, q_min: float, kind: str) -> np.ndarray:
    if T * q_min >= rho:
        raise ValueError("q_min is too large for the requested privacy budget")
    x = np.linspace(0.0, 1.0, T)
    if kind == "uniform":
        raw = np.ones(T)
    elif kind == "front":
        raw = 0.15 + (1.0 - x)
    elif kind == "late":
        raw = 0.15 + x
    elif kind == "cosine_late":
        raw = 0.15 + 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        raise ValueError(f"unknown schedule kind: {kind}")
    raw = raw / raw.sum()
    return q_min + (rho - T * q_min) * raw


def repair_exact_budget(q: np.ndarray, rho: float, q_min: float) -> np.ndarray:
    """Radially repair a positive schedule onto the lower-bounded simplex."""
    residual = np.maximum(np.asarray(q, dtype=float) - q_min, 0.0)
    if float(residual.sum()) <= 0.0:
        residual = np.ones_like(residual)
    return q_min + (rho - len(q) * q_min) * residual / residual.sum()


def summarize_replicates(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item["n_seeds"] = len({g["seed"] for g in group})
        for value_key in value_keys:
            vals = np.array([float(g[value_key]) for g in group], dtype=float)
            item[f"{value_key}_mean"] = float(vals.mean())
            item[f"{value_key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def select_penalty(rows: list[dict], task: str, seed: int, rho: float) -> list[dict]:
    repaired = [
        r for r in rows
        if r["task"] == task and int(r["seed"]) == seed and abs(float(r["rho"]) - rho) < 1e-12
        and str(r["method_family"]).startswith("Penalty repaired beta=")
    ]
    raw = [
        r for r in rows
        if r["task"] == task and int(r["seed"]) == seed and abs(float(r["rho"]) - rho) < 1e-12
        and str(r["method_family"]).startswith("Penalty beta=")
    ]
    if not repaired or not raw:
        return []
    # Hyperparameters are selected on independent public validation paths.
    # Private trajectories and final evaluation paths are never used for tuning.
    best_repaired = min(repaired, key=lambda r: float(r["validation_objective"]))
    best_raw = min(raw, key=lambda r: float(r["validation_objective"]))
    selected = []
    for label, row in [
        ("Penalty repaired-tuned", best_repaired),
        ("Penalty raw-tuned", best_raw),
    ]:
        copied = dict(row)
        copied["method"] = label
        copied["method_family"] = label
        selected.append(copied)
    return selected


def select_pgd(methods: list[dict]) -> dict:
    candidates = [method for method in methods if method["family"] == "PGD"]
    if not candidates:
        raise ValueError("the PGD candidate grid is empty")
    best = min(candidates, key=lambda method: float(method["validation_objective"]))
    selected = dict(best)
    selected["method"] = "PGD public-tuned"
    selected["family"] = "PGD tuned"
    selected["tuning_time_sec"] = sum(float(method["time_sec"]) for method in candidates)
    return selected


def select_penalty_methods(methods: list[dict]) -> list[dict]:
    selected = []
    for family, label in [
        ("Penalty repaired", "Penalty repaired-tuned"),
        ("Penalty", "Penalty raw-tuned"),
    ]:
        candidates = [method for method in methods if method["family"] == family]
        if len(candidates) != 6:
            raise ValueError(f"expected six candidates for {family}")
        best = min(candidates, key=lambda method: float(method["validation_objective"]))
        copied = dict(best)
        copied["method"] = label
        copied["family"] = label
        copied["tuning_time_sec"] = sum(float(method["time_sec"]) for method in candidates)
        selected.append(copied)
    return selected


def run_tabular_case(
    seed: int,
    rho: float,
    K: int,
    pb_max_K: int,
    pb_kkt_tolerance: float,
    m_opt: int,
    m_val: int,
    m_eval: int,
    alpha: float,
    pgd_alphas: list[float],
    q_min: float,
    clip: float,
    delta: float,
) -> list[dict]:
    problem = tabular.make_problem(seed, rho_max=rho, q_min=q_min, clip=clip)
    xis_opt = np.random.default_rng(seed + 101).normal(size=(problem.T, m_opt, problem.d))
    xis_val = np.random.default_rng(seed + 151).normal(size=(problem.T, m_val, problem.d))
    xis_eval = np.random.default_rng(seed + 202).normal(size=(problem.T, m_eval, problem.d))

    methods = []
    pb_method = tabular.run_pb_amd(
        problem,
        xis_opt,
        alpha=alpha,
        K=K,
        kkt_tolerance=pb_kkt_tolerance,
        max_iterations=pb_max_K,
    )
    if not pb_method["converged"]:
        raise RuntimeError(
            f"PB-AMD failed its KKT certificate for tabular seed={seed}, rho={rho}: "
            f"{pb_method['certified_kkt_residual']:.6g}"
        )
    methods.append(pb_method)
    for pgd_alpha in pgd_alphas:
        methods.append(tabular.run_projected_gradient(problem, xis_opt, alpha=pgd_alpha, K=K))
    for beta, lr in [(1.0, 0.04), (3.0, 0.04), (10.0, 0.035), (100.0, 0.025), (1000.0, 0.015), (10000.0, 0.008)]:
        methods.append(tabular.run_penalty(problem, xis_opt, beta=beta, lr=lr, K=K))
    raw_penalties = [m for m in methods if m["family"] == "Penalty"]
    for m in raw_penalties:
        q = repair_exact_budget(m["q"], rho, problem.q_min)
        methods.append({
            "method": f"Penalty repaired beta={m['beta']:g}",
            "family": "Penalty repaired",
            "alpha": "",
            "beta": m["beta"],
            "lr": m["lr"],
            "q": q,
            "design_objective": tabular.objective_only(problem, q, xis_opt),
            "time_sec": m["time_sec"],
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", K),
            "initialization": m.get("initialization", "uniform"),
        })
    for kind, label in [
        ("uniform", "Uniform"),
        ("late", "Linear late"),
        ("front", "Linear front"),
        ("cosine_late", "Cosine late"),
    ]:
        q = shaped_schedule(problem.T, rho, problem.q_min, kind)
        methods.append({
            "method": label,
            "family": label,
            "alpha": "",
            "beta": "",
            "lr": "",
            "q": q,
            "design_objective": tabular.objective_only(problem, q, xis_opt),
            "time_sec": 0.0,
        })

    for method in methods:
        method["validation_objective"] = tabular.objective_only(problem, method["q"], xis_val)
    methods.append(select_pgd(methods))
    methods.extend(select_penalty_methods(methods))

    rows = []
    evaluation_metrics = tabular.simulate_many(problem, [m["q"] for m in methods], xis_eval)
    reference_test_loss, reference_test_acc = tabular.logistic_loss_acc(
        problem.x_test,
        problem.y_test,
        problem.theta_ref[-1][None, :],
        problem.l2,
    )
    for m, eval_metrics in zip(methods, evaluation_metrics):
        q = m["q"]
        sched = tabular.summarize_schedule(q, problem.rho_max)
        exact_feasible = (
            abs(float(np.sum(q)) - rho) <= 1e-10
            and float(np.min(q)) >= problem.q_min - 1e-12
        )
        report_stationarity = exact_feasible and m["family"] in {"PB-AMD", "PGD tuned"}
        if report_stationarity:
            _, saa_gradient = tabular.objective_and_grad(problem, q, xis_opt)
            _, validation_gradient = tabular.objective_and_grad(problem, q, xis_val)
        floor_stats = floor_diagnostics(q, problem.q_min)
        beta = m.get("beta", "")
        if m["family"] == "Penalty":
            method_family = f"Penalty beta={float(beta):g}"
        elif m["family"] == "Penalty repaired":
            method_family = f"Penalty repaired beta={float(beta):g}"
        elif m["family"] == "PGD":
            method_family = f"PGD alpha={float(m['alpha']):g}"
        else:
            method_family = m["family"]
        rows.append({
            "task": "tabular",
            "seed": seed,
            "rho": rho,
            "method": m["method"],
            "method_family": method_family,
            "beta": beta,
            "lr": m.get("lr", ""),
            "alpha": m.get("alpha", ""),
            "initialization": m.get("initialization", ""),
            "design_objective": float(m["design_objective"]),
            "validation_objective": float(m["validation_objective"]),
            "terminal_mean": eval_metrics["terminal_mean"],
            "terminal_std": eval_metrics["terminal_std"],
            "test_loss": eval_metrics["test_loss"],
            "test_acc": eval_metrics["test_acc"],
            "reference_test_loss": float(reference_test_loss[0]),
            "reference_test_acc": float(reference_test_acc[0]),
            "rho_spent": sched["rho_spent"],
            "rho_violation": sched["rho_violation"],
            "rho_slack": sched["rho_slack"],
            "late_over_early": sched["late_over_early"],
            "phase_late_over_early": "",
            "max_over_avg": sched["max_over_avg"],
            "delta": delta,
            "epsilon_target": privacy_epsilon(rho, delta),
            "epsilon_actual": privacy_epsilon(sched["rho_spent"], delta),
            "q_min": problem.q_min,
            "q_min_over_uniform": problem.q_min / (rho / problem.T),
            "sensitivity": problem.sensitivity,
            "clip": problem.clip,
            "saa_variational_gap": variational_gap(q, saa_gradient, rho, problem.q_min) if report_stationarity else "",
            "saa_mirror_residual": mirror_residual(q, saa_gradient, alpha, rho, problem.q_min) if m["family"] == "PB-AMD" else "",
            "saa_projected_residual": projected_gradient_residual(q, saa_gradient, float(m["alpha"]), rho, problem.q_min) if m["family"] == "PGD tuned" else "",
            "saa_kkt_residual": active_kkt_residual(q, saa_gradient, problem.q_min) if report_stationarity else "",
            "validation_variational_gap": variational_gap(q, validation_gradient, rho, problem.q_min) if report_stationarity else "",
            **floor_stats,
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", K if m["family"] == "PB-AMD" else ""),
            "max_prox_feasibility_error": m.get("max_prox_feasibility_error", ""),
            "tuning_time_sec": float(m.get("tuning_time_sec", 0.0)),
            "time_sec": float(m.get("time_sec", 0.0)),
        })
    return rows


def run_fashion_case(
    scenario: str,
    seed: int,
    rho: float,
    K: int,
    pb_max_K: int,
    pb_kkt_tolerance: float,
    m_opt: int,
    m_val: int,
    m_eval: int,
    data_dir: Path,
    n_train: int,
    n_test: int,
    hidden: int,
    alpha: float,
    pgd_alphas: list[float],
    q_min: float,
    clip: float,
    delta: float,
) -> list[dict]:
    problem = fashion.make_problem(
        data_dir,
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        horizon=45,
        batch_size=192,
        hidden=hidden,
        scenario=scenario,
        rho_max=rho,
        q_min=q_min,
        clip=clip,
    )
    theta0 = problem.theta_design_ref[0]
    xis = np.random.default_rng(seed + 77).normal(size=(problem.T, m_opt, problem.shape.p))
    xis_val = np.random.default_rng(seed + 177).normal(size=(problem.T, m_val, problem.shape.p))

    methods = []
    pb_method = fashion.run_pb(
        problem,
        theta0,
        xis,
        K=K,
        alpha=alpha,
        kkt_tolerance=pb_kkt_tolerance,
        max_iterations=pb_max_K,
    )
    if not pb_method["converged"]:
        raise RuntimeError(
            f"PB-AMD failed its KKT certificate for {scenario} seed={seed}, rho={rho}: "
            f"{pb_method['certified_kkt_residual']:.6g}"
        )
    methods.append(pb_method)
    for pgd_alpha in pgd_alphas:
        methods.append(
            fashion.run_projected_gradient(
                problem,
                theta0,
                xis,
                K=K,
                alpha=pgd_alpha,
            )
        )
    for beta, lr in [(1.0, 0.035), (3.0, 0.032), (10.0, 0.025), (100.0, 0.017), (1000.0, 0.01), (10000.0, 0.006)]:
        methods.append(fashion.run_penalty(problem, theta0, xis, K=K, beta=beta, lr=lr))
    raw_penalties = [m for m in methods if m["family"] == "Penalty"]
    for m in raw_penalties:
        q = repair_exact_budget(m["q"], rho, problem.q_min)
        methods.append({
            "method": f"Penalty repaired beta={m['beta']:g}",
            "family": "Penalty repaired",
            "q": q,
            "design_objective": fashion.objective_only(problem, q, xis, theta0),
            "time_sec": m["time_sec"],
            "beta": m["beta"],
            "lr": m.get("lr", ""),
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", K),
            "initialization": m.get("initialization", "uniform"),
        })
    for kind, label in [
        ("uniform", "Uniform"),
        ("late", "Linear late"),
        ("front", "Linear front"),
        ("cosine_late", "Cosine late"),
    ]:
        q = shaped_schedule(problem.T, rho, problem.q_min, kind)
        methods.append({
            "method": label,
            "family": label,
            "q": q,
            "design_objective": fashion.objective_only(problem, q, xis, theta0),
            "time_sec": 0.0,
            "beta": "",
        })

    for method in methods:
        method["validation_objective"] = fashion.objective_only(problem, method["q"], xis_val, theta0)
    methods.append(select_pgd(methods))
    methods.extend(select_penalty_methods(methods))

    rows = []
    task = f"fashion-{scenario}"
    eval_seed = seed + 1000
    evaluation_metrics = fashion.simulate_eval_many(
        problem,
        [m["q"] for m in methods],
        theta0,
        seed=eval_seed,
        paths=m_eval,
    )
    reference_test_loss, reference_test_acc = fashion.mlp_loss_acc(
        problem.theta_ref[-1][None, :],
        problem.x_test,
        problem.y_test,
        problem.shape,
        problem.l2,
    )
    for m, eval_metrics in zip(methods, evaluation_metrics):
        q = m["q"]
        sched = fashion.summarize_schedule(q, problem.rho_max)
        exact_feasible = (
            abs(float(np.sum(q)) - rho) <= 1e-10
            and float(np.min(q)) >= problem.q_min - 1e-12
        )
        report_stationarity = exact_feasible and m["family"] in {"PB-AMD", "PGD tuned"}
        if report_stationarity:
            _, saa_gradient = fashion.objective_and_grad(problem, q, xis, theta0)
            _, validation_gradient = fashion.objective_and_grad(problem, q, xis_val, theta0)
        phase_ratio = ""
        if problem.phase_split is not None:
            phase_ratio = float(q[problem.phase_split :].mean() / max(float(q[: problem.phase_split].mean()), 1e-12))
        floor_stats = floor_diagnostics(q, problem.q_min)
        beta = m.get("beta", "")
        if m["family"] == "Penalty":
            method_family = f"Penalty beta={float(beta):g}"
        elif m["family"] == "Penalty repaired":
            method_family = f"Penalty repaired beta={float(beta):g}"
        elif m["family"] == "PGD":
            method_family = f"PGD alpha={float(m['alpha']):g}"
        else:
            method_family = m["family"]
        rows.append({
            "task": task,
            "seed": seed,
            "rho": rho,
            "method": m["method"],
            "method_family": method_family,
            "beta": beta,
            "lr": m.get("lr", ""),
            "alpha": m.get("alpha", ""),
            "initialization": m.get("initialization", ""),
            "design_objective": float(m["design_objective"]),
            "validation_objective": float(m["validation_objective"]),
            "terminal_mean": eval_metrics["terminal_mean"],
            "terminal_std": eval_metrics["terminal_std"],
            "test_loss": eval_metrics["test_loss"],
            "test_acc": eval_metrics["test_acc"],
            "reference_test_loss": float(reference_test_loss[0]),
            "reference_test_acc": float(reference_test_acc[0]),
            "rho_spent": sched["rho_spent"],
            "rho_violation": sched["rho_violation"],
            "rho_slack": sched["rho_slack"],
            "late_over_early": sched["late_over_early"],
            "phase_late_over_early": phase_ratio,
            "max_over_avg": sched["max_q"] / (rho / problem.T),
            "delta": delta,
            "epsilon_target": privacy_epsilon(rho, delta),
            "epsilon_actual": privacy_epsilon(sched["rho_spent"], delta),
            "q_min": problem.q_min,
            "q_min_over_uniform": problem.q_min / (rho / problem.T),
            "sensitivity": problem.sensitivity,
            "clip": problem.clip,
            "saa_variational_gap": variational_gap(q, saa_gradient, rho, problem.q_min) if report_stationarity else "",
            "saa_mirror_residual": mirror_residual(q, saa_gradient, alpha, rho, problem.q_min) if m["family"] == "PB-AMD" else "",
            "saa_projected_residual": projected_gradient_residual(q, saa_gradient, float(m["alpha"]), rho, problem.q_min) if m["family"] == "PGD tuned" else "",
            "saa_kkt_residual": active_kkt_residual(q, saa_gradient, problem.q_min) if report_stationarity else "",
            "validation_variational_gap": variational_gap(q, validation_gradient, rho, problem.q_min) if report_stationarity else "",
            **floor_stats,
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", K if m["family"] == "PB-AMD" else ""),
            "max_prox_feasibility_error": m.get("max_prox_feasibility_error", ""),
            "tuning_time_sec": float(m.get("tuning_time_sec", 0.0)),
            "time_sec": float(m.get("time_sec", 0.0)),
        })
    return rows


def run_fashion_job(job: tuple) -> list[dict]:
    scenario, seed, rho = job[:3]
    print(f"[fashion:{scenario}] seed={seed} rho={rho}", flush=True)
    return run_fashion_case(*job)


def write_plot_tables(summary: list[dict], out_dir: Path) -> None:
    selected = [
        "PB-AMD",
        "PGD tuned",
        "Uniform",
        "Linear late",
        "Cosine late",
        "Penalty repaired-tuned",
    ]
    name_map = {
        "PB-AMD": "pbamd",
        "PGD tuned": "pgd",
        "Uniform": "uniform",
        "Linear late": "linear_late",
        "Cosine late": "cosine_late",
        "Penalty repaired-tuned": "penalty_repaired",
    }
    for task in sorted({r["task"] for r in summary}):
        rhos = sorted({float(r["rho"]) for r in summary if r["task"] == task})
        rows = []
        for rho in rhos:
            item = {"rho": rho}
            for method in selected:
                match = [
                    r for r in summary
                    if r["task"] == task and abs(float(r["rho"]) - rho) < 1e-12 and r["method_family"] == method
                ]
                if match:
                    r = match[0]
                    prefix = name_map[method]
                    item[prefix] = r["terminal_mean_mean"]
                    item[f"{prefix}_violation"] = r["rho_violation_mean"]
                    item[f"{prefix}_late"] = r["late_over_early_mean"]
            rows.append(item)
        keys = ["rho"]
        for method in selected:
            prefix = name_map[method]
            keys.extend([prefix, f"{prefix}_violation", f"{prefix}_late"])
        with (out_dir / f"plot_{task}.dat").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", delimiter=" ")
            writer.writeheader()
            writer.writerows(rows)


def paired_comparisons(raw_rows: list[dict], summary: list[dict]) -> list[dict]:
    fixed_methods = ["Linear late", "Linear front", "Cosine late"]
    summary_index = {
        (row["task"], float(row["rho"]), row["method_family"]): row
        for row in summary
    }
    out = []
    task_budgets = sorted({(row["task"], float(row["rho"])) for row in raw_rows})
    for task, rho in task_budgets:
        best_fixed = min(
            fixed_methods,
            key=lambda method: float(summary_index[(task, rho, method)]["validation_objective_mean"]),
        )
        comparators = [
            "Uniform",
            best_fixed,
            "PGD tuned",
            "Penalty repaired-tuned",
        ]
        for comparator in dict.fromkeys(comparators):
            pb = {
                int(row["seed"]): row for row in raw_rows
                if row["task"] == task and float(row["rho"]) == rho and row["method_family"] == "PB-AMD"
            }
            baseline = {
                int(row["seed"]): row for row in raw_rows
                if row["task"] == task and float(row["rho"]) == rho and row["method_family"] == comparator
            }
            seeds = sorted(pb.keys() & baseline.keys())
            for metric in ["terminal_mean", "test_acc"]:
                differences = np.array([
                    float(pb[seed][metric]) - float(baseline[seed][metric])
                    for seed in seeds
                ])
                n = len(differences)
                mean = float(differences.mean())
                standard_error = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                critical = student_t_quantile(0.975, n - 1) if n > 1 else 0.0
                half_width = critical * standard_error
                if n <= 1:
                    pvalue = 1.0
                elif standard_error == 0.0:
                    pvalue = 0.0 if mean != 0.0 else 1.0
                else:
                    statistic = mean / standard_error
                    pvalue = student_t_two_sided_pvalue(statistic, n - 1)
                out.append({
                    "task": task,
                    "rho": rho,
                    "comparator": comparator,
                    "metric": metric,
                    "analysis_role": "primary" if comparator == "Uniform" else "descriptive",
                    "n_pairs": n,
                    "mean_difference_pb_minus_comparator": mean,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "pvalue_two_sided": pvalue,
                    "holm_adjusted_pvalue": "",
                    "holm_family_size": "",
                    "holm_reject_0_05": "",
                })
    for metric in ["terminal_mean", "test_acc"]:
        indices = [
            index for index, row in enumerate(out)
            if row["metric"] == metric and row["analysis_role"] == "primary"
        ]
        adjusted = holm_adjusted_pvalues([float(out[index]["pvalue_two_sided"]) for index in indices])
        for index, adjusted_value in zip(indices, adjusted):
            out[index]["holm_adjusted_pvalue"] = adjusted_value
            out[index]["holm_family_size"] = len(indices)
            out[index]["holm_reject_0_05"] = int(adjusted_value <= 0.05)
    return out


def build_protocol_config(
    args: argparse.Namespace,
    tabular_pgd_alphas: list[float],
    fashion_pgd_alphas: list[float],
) -> dict:
    source_files = [
        Path(__file__).resolve(),
        Path(tabular.__file__).resolve(),
        Path(fashion.__file__).resolve(),
        Path(__file__).resolve().with_name("experiment_utils.py"),
    ]
    return {key: value for key, value in vars(args).items() if key != "resume"} | {
        "protocol_version": 5,
        "source_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in source_files
        },
        "schedule_design": "public proxy data only",
        "private_training": "disjoint private split with fixed selected schedule",
        "adjacency": "replace-one",
        "aggregation": "average of smoothly clipped per-example gradients",
        "sensitivity": "public calibration bound 2 * C_clip / batch_size",
        "privacy_charge_interpretation": "q_t is a certified upper bound; exact worst-case charge is (Delta_exact / Delta_hat)^2 q_t",
        "privacy_amplification": "none; accountant conditions on public batches and uses direct adaptive composition",
        "noise_gradient_std": "sensitivity / sqrt(2 * q_t)",
        "pb_amd_output_rule": "minimum fixed-path SAA objective among q^0,...,q^K at the first K at or above the prescribed minimum for which that returned iterate satisfies the KKT tolerance",
        "pb_amd_stopping": {
            "kkt_tolerance": args.pb_kkt_tol,
            "tabular_minimum_iterations": args.tabular_K,
            "tabular_maximum_iterations": args.tabular_pb_max_K,
            "fashion_minimum_iterations": args.fashion_K,
            "fashion_maximum_iterations": args.fashion_pb_max_K,
            "failure_policy": "reject the case if the returned minimum-objective iterate is not certified by the maximum iteration count",
        },
        "selection": "independent public validation Gaussian paths",
        "fixed_nonuniform_selection": "one schedule per task-budget pair, minimizing mean public-validation objective over the ten design seeds",
        "primary_inference": "PB-AMD minus prespecified uniform allocation; endpoint-wise Holm correction over nine task-budget comparisons",
        "other_comparators": "descriptive mechanism and computational-cost comparisons",
        "final_evaluation": "fresh private-training Gaussian paths shared across methods",
        "tabular_settings": {
            "n_public_design": 4500,
            "n_private_train": 4500,
            "n_test": 1500,
            "dimension": 64,
            "horizon": 60,
            "batch_size": 256,
            "learning_rate": 0.18,
            "l2": 0.002,
            "penalty_grid": [[1.0, 0.04], [3.0, 0.04], [10.0, 0.035], [100.0, 0.025], [1000.0, 0.015], [10000.0, 0.008]],
            "pgd_alpha_grid": tabular_pgd_alphas,
        },
        "fashion_settings": {
            "n_public_auxiliary": 12000,
            "n_public_design": args.fashion_n_train,
            "n_private_train": args.fashion_n_train,
            "n_test": args.fashion_n_test,
            "input_dimension": 196,
            "hidden_units": args.fashion_hidden,
            "classes": 10,
            "parameters": 196 * args.fashion_hidden + args.fashion_hidden + 10 * args.fashion_hidden + 10,
            "active_parameters_every_step": 196 * args.fashion_hidden + args.fashion_hidden + 10 * args.fashion_hidden + 10,
            "gaussian_noise_dimension": 196 * args.fashion_hidden + args.fashion_hidden + 10 * args.fashion_hidden + 10,
            "active_mask": "deterministic all-ones mask in both protocols",
            "horizon": 45,
            "batch_size": 192,
            "standard_learning_rate": 0.15,
            "restart_learning_rates": [0.02, 0.075],
            "restart_split": 25,
            "public_pretrain_steps": 300,
            "public_pretrain_learning_rate": 0.05,
            "public_pretrain_records": 12000,
            "pretrain_data_relation": "public auxiliary split; disjoint from private training and overlapping the 8000-record public design subset",
            "l2": 0.0001,
            "penalty_grid": [[1.0, 0.035], [3.0, 0.032], [10.0, 0.025], [100.0, 0.017], [1000.0, 0.01], [10000.0, 0.006]],
            "pgd_alpha_grid": fashion_pgd_alphas,
        },
    }


def protocol_mismatches(previous: dict, current: dict) -> list[str]:
    ignored = {"out", "workers"}
    keys = sorted((set(previous) | set(current)) - ignored)
    return [key for key in keys if previous.get(key) != current.get(key)]


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/comprehensive_schedule_benchmark")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    parser.add_argument("--tabular-seeds", default="17,31,53,71,97,113,137,151,173,197")
    parser.add_argument("--fashion-seeds", default="123,321,777,1001,2027,409,613,887,1301,1601")
    parser.add_argument("--tabular-rhos", default="0.25,0.5,1.0")
    parser.add_argument("--fashion-rhos", default="0.25,0.5,1.0")
    parser.add_argument("--fashion-scenarios", default="standard,lr_restart")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-fashion", action="store_true")
    parser.add_argument("--tabular-K", type=int, default=180)
    parser.add_argument("--fashion-K", type=int, default=100)
    parser.add_argument("--tabular-pb-max-K", type=int, default=2000)
    parser.add_argument("--fashion-pb-max-K", type=int, default=2000)
    parser.add_argument("--pb-kkt-tol", type=float, default=5e-5)
    parser.add_argument("--tabular-M-opt", type=int, default=256)
    parser.add_argument("--fashion-M-opt", type=int, default=8)
    parser.add_argument("--tabular-M-val", type=int, default=512)
    parser.add_argument("--fashion-M-val", type=int, default=48)
    parser.add_argument("--tabular-M-eval", type=int, default=512)
    parser.add_argument("--fashion-M-eval", type=int, default=256)
    parser.add_argument("--tabular-alpha", type=float, default=0.005)
    parser.add_argument("--fashion-standard-alpha", type=float, default=0.0005)
    parser.add_argument("--fashion-restart-alpha", type=float, default=0.005)
    parser.add_argument("--tabular-pgd-alphas", default="3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--fashion-pgd-alphas", default="3e-6,1e-5,3e-5,1e-4")
    parser.add_argument("--tabular-clip", type=float, default=4.0)
    parser.add_argument("--fashion-clip", type=float, default=1.75)
    parser.add_argument("--q-min", type=float, default=1e-5)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--fashion-n-train", type=int, default=8000)
    parser.add_argument("--fashion-n-test", type=int, default=3000)
    parser.add_argument("--fashion-hidden", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"
    tabular_seeds = [int(x) for x in args.tabular_seeds.split(",") if x]
    fashion_seeds = [int(x) for x in args.fashion_seeds.split(",") if x]
    tabular_rhos = [float(x) for x in args.tabular_rhos.split(",") if x]
    fashion_rhos = [float(x) for x in args.fashion_rhos.split(",") if x]
    tabular_pgd_alphas = [float(x) for x in args.tabular_pgd_alphas.split(",") if x]
    fashion_pgd_alphas = [float(x) for x in args.fashion_pgd_alphas.split(",") if x]
    methods_per_case = {
        "tabular": 20 + len(tabular_pgd_alphas),
        "fashion-standard": 20 + len(fashion_pgd_alphas),
        "fashion-lr_restart": 20 + len(fashion_pgd_alphas),
    }
    config = build_protocol_config(args, tabular_pgd_alphas, fashion_pgd_alphas)
    config_path = out_dir / "config.json"
    if raw_path.exists() and not args.resume:
        raise ValueError(
            f"refusing to overwrite existing comprehensive results in {out_dir}; "
            "use an empty output directory or an exactly matching --resume request"
        )
    if args.resume and raw_path.exists():
        if not config_path.exists():
            raise ValueError("cannot resume comprehensive results without config.json")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        mismatched = protocol_mismatches(previous, config)
        if mismatched:
            raise ValueError(
                "comprehensive resume protocol mismatch: " + ", ".join(mismatched)
            )
    write_json_atomic(config_path, config)
    raw_rows: list[dict] = (
        retain_complete_cases(read_raw_results(raw_path), methods_per_case)
        if args.resume
        else []
    )
    completed = {(r["task"], int(r["seed"]), float(r["rho"])) for r in raw_rows}

    if not args.skip_tabular:
        for seed in tabular_seeds:
            for rho in tabular_rhos:
                if ("tabular", seed, rho) in completed:
                    continue
                print(f"[tabular] seed={seed} rho={rho}", flush=True)
                raw_rows.extend(run_tabular_case(
                    seed,
                    rho,
                    args.tabular_K,
                    args.tabular_pb_max_K,
                    args.pb_kkt_tol,
                    args.tabular_M_opt,
                    args.tabular_M_val,
                    args.tabular_M_eval,
                    args.tabular_alpha,
                    tabular_pgd_alphas,
                    args.q_min,
                    args.tabular_clip,
                    args.delta,
                ))
                write_csv(raw_path, raw_rows)

    if not args.skip_fashion:
        # Populate and verify the shared read-only cache before worker threads
        # start, avoiding concurrent first-download attempts on a clean run.
        fashion.load_preprocessed_fashion_mnist(str(Path(args.data_dir).resolve()))
        scenarios = [value for value in args.fashion_scenarios.split(",") if value]
        fashion_jobs = []
        for scenario in scenarios:
            for seed in fashion_seeds:
                for rho in fashion_rhos:
                    task = f"fashion-{scenario}"
                    if (task, seed, rho) in completed:
                        continue
                    fashion_alpha = (
                        args.fashion_standard_alpha
                        if scenario == "standard"
                        else args.fashion_restart_alpha
                    )
                    fashion_jobs.append((
                        scenario,
                        seed,
                        rho,
                        args.fashion_K,
                        args.fashion_pb_max_K,
                        args.pb_kkt_tol,
                        args.fashion_M_opt,
                        args.fashion_M_val,
                        args.fashion_M_eval,
                        Path(args.data_dir),
                        args.fashion_n_train,
                        args.fashion_n_test,
                        args.fashion_hidden,
                        fashion_alpha,
                        fashion_pgd_alphas,
                        args.q_min,
                        args.fashion_clip,
                        args.delta,
                    ))
        if args.workers <= 1:
            fashion_results = map(run_fashion_job, fashion_jobs)
        else:
            executor = ThreadPoolExecutor(max_workers=args.workers)
            fashion_results = executor.map(run_fashion_job, fashion_jobs)
        try:
            for job, result in zip(fashion_jobs, fashion_results):
                scenario, seed, rho = job[:3]
                print(f"[fashion:{scenario}] completed seed={seed} rho={rho}", flush=True)
                raw_rows.extend(result)
                write_csv(raw_path, raw_rows)
        finally:
            if args.workers > 1:
                executor.shutdown()

    write_csv(raw_path, raw_rows)
    summary = summarize_replicates(
        raw_rows,
        ["task", "rho", "method_family"],
        ["design_objective", "validation_objective", "terminal_mean", "test_loss", "test_acc", "reference_test_loss", "reference_test_acc", "rho_violation", "rho_slack", "late_over_early", "max_over_avg", "epsilon_target", "epsilon_actual", "time_sec", "tuning_time_sec"],
    )
    write_csv(out_dir / "summary_by_task_budget.csv", summary)
    write_csv(out_dir / "paired_comparisons.csv", paired_comparisons(raw_rows, summary))
    write_plot_tables(summary, out_dir)
    write_json_atomic(config_path, config)
    print(json.dumps({
        "out_dir": str(out_dir),
        "raw": str(raw_path),
        "summary": str(out_dir / "summary_by_task_budget.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
