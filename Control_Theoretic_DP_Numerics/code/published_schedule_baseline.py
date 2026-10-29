from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion
from experiment_utils import exponential_privacy_schedule, student_t_quantile


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def case_key(row: dict) -> tuple[str, int, float]:
    return str(row["task"]), int(row["seed"]), float(row["rho"])


def retain_complete_cases(
    rows: list[dict],
    candidates: list[dict],
    schedules: list[dict],
    ratios: list[float],
) -> tuple[list[dict], list[dict], list[dict]]:
    raw_groups: dict[tuple[str, int, float], list[dict]] = {}
    candidate_groups: dict[tuple[str, int, float], list[dict]] = {}
    schedule_groups: dict[tuple[str, int, float], list[dict]] = {}
    for row in rows:
        raw_groups.setdefault(case_key(row), []).append(row)
    for row in candidates:
        candidate_groups.setdefault(case_key(row), []).append(row)
    for row in schedules:
        schedule_groups.setdefault(case_key(row), []).append(row)

    expected_ratios = sorted(float(value) for value in ratios)
    complete = set()
    for key, group in raw_groups.items():
        task = key[0]
        horizon = 60 if task == "tabular" else 45
        candidate_group = candidate_groups.get(key, [])
        schedule_group = schedule_groups.get(key, [])
        if (
            len(group) == 1
            and len(candidate_group) == len(expected_ratios)
            and sorted(float(row["ratio"]) for row in candidate_group) == expected_ratios
            and sum(int(row["selected"]) for row in candidate_group) == 1
            and len(schedule_group) == horizon
            and {int(row["t"]) for row in schedule_group} == set(range(horizon))
        ):
            complete.add(key)
    return (
        [row for row in rows if case_key(row) in complete],
        [row for row in candidates if case_key(row) in complete],
        [row for row in schedules if case_key(row) in complete],
    )


def checkpoint(
    out_dir: Path,
    rows: list[dict],
    candidates: list[dict],
    schedules: list[dict],
) -> None:
    ordered_rows = sorted(rows, key=lambda row: (row["task"], int(row["seed"]), float(row["rho"])))
    ordered_candidates = sorted(
        candidates,
        key=lambda row: (row["task"], int(row["seed"]), float(row["rho"]), float(row["ratio"])),
    )
    ordered_schedules = sorted(
        schedules,
        key=lambda row: (row["task"], int(row["seed"]), float(row["rho"]), int(row["t"])),
    )
    write_csv(out_dir / "published_schedule_raw.csv", ordered_rows)
    write_csv(out_dir / "published_schedule_candidates.csv", ordered_candidates)
    write_csv(out_dir / "published_schedule_schedules.csv", ordered_schedules)
    write_csv(out_dir / "published_schedule_summary.csv", summarize(ordered_rows))


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    if len(data) == 1:
        return mean, mean, mean
    standard_error = float(data.std(ddof=1) / np.sqrt(len(data)))
    half_width = student_t_quantile(0.975, len(data) - 1) * standard_error
    return mean, mean - half_width, mean + half_width


def choose_schedule(
    horizon: int,
    rho: float,
    q_min: float,
    ratios: list[float],
    objective,
) -> tuple[np.ndarray, float, list[dict]]:
    candidates = []
    for ratio in ratios:
        q = exponential_privacy_schedule(horizon, rho, q_min, ratio)
        candidates.append({
            "ratio": ratio,
            "q": q,
            "validation_objective": float(objective(q)),
        })
    selected = min(candidates, key=lambda item: (item["validation_objective"], item["ratio"]))
    return selected["q"], float(selected["ratio"]), candidates


def run_tabular(
    seed: int,
    rho: float,
    ratios: list[float],
    q_min: float,
) -> tuple[dict, list[dict], list[dict]]:
    problem = tabular.make_problem(seed, rho_max=rho, q_min=q_min, clip=4.0)
    xis_val = np.random.default_rng(seed + 151).normal(
        size=(problem.T, 512, problem.d)
    )
    q, ratio, candidates = choose_schedule(
        problem.T,
        rho,
        q_min,
        ratios,
        lambda schedule: tabular.objective_only(problem, schedule, xis_val),
    )
    uniform = np.full(problem.T, rho / problem.T)
    xis_eval = np.random.default_rng(seed + 202).normal(
        size=(problem.T, 512, problem.d)
    )
    selected_eval, uniform_eval = tabular.simulate_many(
        problem, [q, uniform], xis_eval
    )
    schedule = tabular.summarize_schedule(q, rho)
    row = {
        "task": "tabular",
        "seed": seed,
        "rho": rho,
        "method": "HWZ exponential public-tuned",
        "selected_ratio": ratio,
        "candidate_count": len(ratios),
        "validation_objective": min(item["validation_objective"] for item in candidates),
        "terminal_mean": selected_eval["terminal_mean"],
        "terminal_std": selected_eval["terminal_std"],
        "test_loss": selected_eval["test_loss"],
        "test_acc": selected_eval["test_acc"],
        "uniform_terminal_mean": uniform_eval["terminal_mean"],
        "uniform_test_acc": uniform_eval["test_acc"],
        "rho_spent": schedule["rho_spent"],
        "rho_violation": schedule["rho_violation"],
        "late_over_early": schedule["late_over_early"],
    }
    candidate_rows = [
        {
            "task": "tabular",
            "seed": seed,
            "rho": rho,
            "ratio": item["ratio"],
            "validation_objective": item["validation_objective"],
            "selected": int(item["ratio"] == ratio),
        }
        for item in candidates
    ]
    schedule_rows = [
        {"task": "tabular", "seed": seed, "rho": rho, "t": t, "q": value}
        for t, value in enumerate(q)
    ]
    return row, candidate_rows, schedule_rows


def run_fashion(
    scenario: str,
    seed: int,
    rho: float,
    ratios: list[float],
    q_min: float,
    data_dir: Path,
) -> tuple[dict, list[dict], list[dict]]:
    problem = fashion.make_problem(
        data_dir,
        seed=seed,
        n_train=8000,
        n_test=3000,
        horizon=45,
        batch_size=192,
        hidden=32,
        scenario=scenario,
        rho_max=rho,
        q_min=q_min,
        clip=1.75,
    )
    theta0 = problem.theta_design_ref[0]
    xis_val = np.random.default_rng(seed + 177).normal(
        size=(problem.T, 48, problem.shape.p)
    )
    q, ratio, candidates = choose_schedule(
        problem.T,
        rho,
        q_min,
        ratios,
        lambda schedule: fashion.objective_only(problem, schedule, xis_val, theta0),
    )
    uniform = np.full(problem.T, rho / problem.T)
    selected_eval, uniform_eval = fashion.simulate_eval_many(
        problem,
        [q, uniform],
        theta0,
        seed=seed + 1000,
        paths=256,
    )
    schedule = fashion.summarize_schedule(q, rho)
    task = f"fashion-{scenario}"
    row = {
        "task": task,
        "seed": seed,
        "rho": rho,
        "method": "HWZ exponential public-tuned",
        "selected_ratio": ratio,
        "candidate_count": len(ratios),
        "validation_objective": min(item["validation_objective"] for item in candidates),
        "terminal_mean": selected_eval["terminal_mean"],
        "terminal_std": selected_eval["terminal_std"],
        "test_loss": selected_eval["test_loss"],
        "test_acc": selected_eval["test_acc"],
        "uniform_terminal_mean": uniform_eval["terminal_mean"],
        "uniform_test_acc": uniform_eval["test_acc"],
        "rho_spent": schedule["rho_spent"],
        "rho_violation": schedule["rho_violation"],
        "late_over_early": schedule["late_over_early"],
    }
    candidate_rows = [
        {
            "task": task,
            "seed": seed,
            "rho": rho,
            "ratio": item["ratio"],
            "validation_objective": item["validation_objective"],
            "selected": int(item["ratio"] == ratio),
        }
        for item in candidates
    ]
    schedule_rows = [
        {"task": task, "seed": seed, "rho": rho, "t": t, "q": value}
        for t, value in enumerate(q)
    ]
    return row, candidate_rows, schedule_rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), float(row["rho"])), []).append(row)
    output = []
    for (task, rho), group in sorted(groups.items()):
        terminal = [
            float(row["terminal_mean"]) - float(row["uniform_terminal_mean"])
            for row in group
        ]
        accuracy = [
            float(row["test_acc"]) - float(row["uniform_test_acc"])
            for row in group
        ]
        term_mean, term_low, term_high = paired_interval(terminal)
        acc_mean, acc_low, acc_high = paired_interval(accuracy)
        output.append({
            "task": task,
            "rho": rho,
            "n_seeds": len(group),
            "selected_ratio_mean": float(np.mean([float(row["selected_ratio"]) for row in group])),
            "terminal_difference_vs_uniform": term_mean,
            "terminal_ci95_low": term_low,
            "terminal_ci95_high": term_high,
            "accuracy_difference_vs_uniform": acc_mean,
            "accuracy_ci95_low": acc_low,
            "accuracy_ci95_high": acc_high,
            "maximum_budget_error": max(abs(float(row["rho_spent"]) - rho) for row in group),
        })
    return output


def run_job(job: tuple) -> tuple[dict, list[dict], list[dict]]:
    kind = job[0]
    if kind == "tabular":
        return run_tabular(*job[1:])
    return run_fashion(*job[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/submission_v5_published_baseline")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    parser.add_argument("--ratios", default="1,2,4,8")
    parser.add_argument("--tabular-seeds", default="17,31,53,71,97,113,137,151,173,197")
    parser.add_argument("--fashion-seeds", default="123,321,777,1001,2027,409,613,887,1301,1601")
    parser.add_argument("--rhos", default="0.25,0.5,1.0")
    parser.add_argument("--q-min", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ratios = [float(value) for value in args.ratios.split(",") if value]
    tabular_seeds = [int(value) for value in args.tabular_seeds.split(",") if value]
    fashion_seeds = [int(value) for value in args.fashion_seeds.split(",") if value]
    rhos = [float(value) for value in args.rhos.split(",") if value]
    data_dir = Path(args.data_dir)
    fashion.load_preprocessed_fashion_mnist(str(data_dir.resolve()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != "resume"} | {
        "protocol_version": 5,
        "source_method": "Hong, Wang, and Zhou (FAccT 2022), DOI 10.1145/3531146.3533070",
        "adaptation": "geometrically decreasing noise variance on the paper's exact certified zCDP simplex; fixed clipping and no momentum",
        "selection": "end-to-start charge ratio selected only by independent public validation paths",
        "private_evaluation": "fresh private-training paths shared with the uniform comparator",
        "adjacency": "replace-one",
        "sensitivity": "2 * C_clip / batch_size",
        "privacy_charge": "q_t is a certified upper bound; the exact charge uses delta_exact and is no larger",
        "privacy_amplification": "none",
    }
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        protocol_fields = [
            "data_dir",
            "ratios",
            "tabular_seeds",
            "fashion_seeds",
            "rhos",
            "q_min",
            "protocol_version",
            "source_method",
            "adaptation",
            "selection",
            "private_evaluation",
        ]
        if any(previous.get(field) != config.get(field) for field in protocol_fields):
            raise ValueError("the saved published-baseline protocol does not match this resume request")
    temporary_config = config_path.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary_config.replace(config_path)

    jobs = []
    for seed in tabular_seeds:
        for rho in rhos:
            jobs.append(("tabular", seed, rho, ratios, args.q_min))
    for scenario in ["standard", "lr_restart"]:
        for seed in fashion_seeds:
            for rho in rhos:
                jobs.append(("fashion", scenario, seed, rho, ratios, args.q_min, data_dir))

    rows: list[dict] = []
    candidates: list[dict] = []
    schedules: list[dict] = []
    raw_path = out_dir / "published_schedule_raw.csv"
    candidate_path = out_dir / "published_schedule_candidates.csv"
    schedule_path = out_dir / "published_schedule_schedules.csv"
    if args.resume and raw_path.exists() and candidate_path.exists() and schedule_path.exists():
        rows, candidates, schedules = retain_complete_cases(
            read_csv(raw_path),
            read_csv(candidate_path),
            read_csv(schedule_path),
            ratios,
        )
        if rows:
            checkpoint(out_dir, rows, candidates, schedules)
    completed = {case_key(row) for row in rows}
    pending_jobs = []
    for job in jobs:
        if job[0] == "tabular":
            key = ("tabular", int(job[1]), float(job[2]))
        else:
            key = (f"fashion-{job[1]}", int(job[2]), float(job[3]))
        if key not in completed:
            pending_jobs.append(job)

    if args.workers == 1:
        iterator = map(run_job, pending_jobs)
        for result in iterator:
            rows.append(result[0])
            candidates.extend(result[1])
            schedules.extend(result[2])
            checkpoint(out_dir, rows, candidates, schedules)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_job, pending_jobs):
                rows.append(result[0])
                candidates.extend(result[1])
                schedules.extend(result[2])
                checkpoint(out_dir, rows, candidates, schedules)
    if not pending_jobs and rows:
        checkpoint(out_dir, rows, candidates, schedules)
    print(json.dumps({"rows": len(rows), "resumed_cases": len(completed), "out": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
