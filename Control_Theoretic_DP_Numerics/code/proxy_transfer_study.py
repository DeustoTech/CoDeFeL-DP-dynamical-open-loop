from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import fashion_mnist_mlp_schedule_compare as fashion
from experiment_utils import student_t_quantile


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


def retain_complete_cases(
    rows: list[dict],
    schedules: list[dict],
    horizon: int = 45,
) -> tuple[list[dict], list[dict]]:
    raw_groups: dict[tuple[int, float], list[dict]] = {}
    schedule_groups: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        raw_groups.setdefault((int(row["seed"]), float(row["rho"])), []).append(row)
    for row in schedules:
        schedule_groups.setdefault((int(row["seed"]), float(row["rho"])), []).append(row)
    complete = {
        key
        for key, group in raw_groups.items()
        if len(group) == 1
        and len(schedule_groups.get(key, [])) == horizon
        and {int(row["t"]) for row in schedule_groups.get(key, [])} == set(range(horizon))
    }
    return (
        [row for row in rows if (int(row["seed"]), float(row["rho"])) in complete],
        [row for row in schedules if (int(row["seed"]), float(row["rho"])) in complete],
    )


def contaminate_labels(
    labels: np.ndarray,
    fraction: float,
    seed: int,
    classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if not 0.0 <= fraction < 1.0 or classes < 2:
        raise ValueError("invalid contamination parameters")
    count = int(round(fraction * len(labels)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(labels), size=count, replace=False))
    contaminated = labels.copy()
    contaminated[indices] = (
        contaminated[indices] + rng.integers(1, classes, size=count)
    ) % classes
    return contaminated, indices


def shift_features(
    features: np.ndarray,
    contrast: float,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    if contrast <= 0.0 or noise_std < 0.0:
        raise ValueError("invalid feature-shift parameters")
    rng = np.random.default_rng(seed)
    return contrast * np.asarray(features, dtype=float) + noise_std * rng.normal(
        size=features.shape
    )


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        raise ValueError("paired_interval requires at least one value")
    mean = float(data.mean())
    if len(data) == 1:
        return mean, mean, mean
    standard_error = float(data.std(ddof=1) / np.sqrt(len(data)))
    half_width = student_t_quantile(0.975, len(data) - 1) * standard_error
    return mean, mean - half_width, mean + half_width


def summarize(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("cannot summarize an empty proxy-transfer experiment")
    settings = {
        (
            float(row["rho"]),
            float(row["label_contamination"]),
            float(row["feature_contrast"]),
            float(row["feature_noise_std"]),
        )
        for row in rows
    }
    if len(settings) != 1:
        raise ValueError("proxy-transfer rows do not share one protocol")
    rho, label_fraction, feature_contrast, feature_noise_std = settings.pop()
    reduction = paired_interval([float(row["terminal_reduction_percent"]) for row in rows])
    accuracy = paired_interval([float(row["accuracy_difference_pp"]) for row in rows])
    return [{
        "n_seeds": len(rows),
        "rho": rho,
        "label_contamination": label_fraction,
        "feature_contrast": feature_contrast,
        "feature_noise_std": feature_noise_std,
        "late_over_early_mean": float(np.mean([float(row["late_over_early"]) for row in rows])),
        "terminal_reduction_percent": reduction[0],
        "terminal_reduction_ci95_low": reduction[1],
        "terminal_reduction_ci95_high": reduction[2],
        "accuracy_difference_pp": accuracy[0],
        "accuracy_difference_ci95_low": accuracy[1],
        "accuracy_difference_ci95_high": accuracy[2],
        "maximum_budget_error": max(abs(float(row["rho_spent"]) - rho) for row in rows),
    }]


def checkpoint(out_dir: Path, rows: list[dict], schedules: list[dict]) -> None:
    ordered_rows = sorted(rows, key=lambda row: int(row["seed"]))
    ordered_schedules = sorted(
        schedules,
        key=lambda row: (int(row["seed"]), float(row["rho"]), int(row["t"])),
    )
    write_csv(out_dir / "proxy_transfer_raw.csv", ordered_rows)
    write_csv(out_dir / "proxy_transfer_schedules.csv", ordered_schedules)
    write_csv(out_dir / "proxy_transfer_summary.csv", summarize(ordered_rows))


def run_seed(
    seed: int,
    data_dir: Path,
    rho: float,
    q_min: float,
    label_fraction: float,
    feature_contrast: float,
    feature_noise_std: float,
    K: int,
    alpha: float,
) -> tuple[dict, list[dict]]:
    problem = fashion.make_problem(
        data_dir,
        seed=seed,
        n_train=8000,
        n_test=3000,
        horizon=45,
        batch_size=192,
        hidden=32,
        scenario="lr_restart",
        rho_max=rho,
        q_min=q_min,
        clip=1.75,
    )
    theta0 = problem.theta_design_ref[0]
    shifted_labels, changed = contaminate_labels(
        problem.y_design,
        label_fraction,
        seed=seed + 40_001,
    )
    shifted_features = shift_features(
        problem.x_design,
        feature_contrast,
        feature_noise_std,
        seed=seed + 50_001,
    )
    shifted_reference = fashion.compute_reference(
        theta0,
        shifted_features,
        shifted_labels,
        problem.batches,
        problem.active_masks,
        problem.eta,
        problem.shape,
        problem.l2,
        problem.clip,
    )
    shifted_problem = replace(
        problem,
        x_design=shifted_features,
        y_design=shifted_labels,
        theta_design_ref=shifted_reference,
    )
    xis_opt = np.random.default_rng(seed + 77).normal(
        size=(problem.T, 8, problem.shape.p)
    )
    method = fashion.run_pb(
        shifted_problem,
        theta0,
        xis_opt,
        K=K,
        alpha=alpha,
    )
    q = method["q"]
    uniform = np.full(problem.T, rho / problem.T)
    selected_eval, uniform_eval = fashion.simulate_eval_many(
        problem,
        [q, uniform],
        theta0,
        seed=seed + 1000,
        paths=256,
    )
    xis_val = np.random.default_rng(seed + 177).normal(
        size=(problem.T, 48, problem.shape.p)
    )
    schedule = fashion.summarize_schedule(q, rho)
    row = {
        "task": "fashion-lr_restart-proxy_shift",
        "seed": seed,
        "rho": rho,
        "label_contamination": label_fraction,
        "feature_contrast": feature_contrast,
        "feature_noise_std": feature_noise_std,
        "changed_public_labels": len(changed),
        "alpha": alpha,
        "K": K,
        "design_objective": method["design_objective"],
        "public_validation_objective": fashion.objective_only(
            shifted_problem, q, xis_val, theta0
        ),
        "terminal_mean": selected_eval["terminal_mean"],
        "uniform_terminal_mean": uniform_eval["terminal_mean"],
        "terminal_reduction_percent": 100.0 * (
            uniform_eval["terminal_mean"] - selected_eval["terminal_mean"]
        ) / uniform_eval["terminal_mean"],
        "test_acc": selected_eval["test_acc"],
        "uniform_test_acc": uniform_eval["test_acc"],
        "accuracy_difference_pp": 100.0 * (
            selected_eval["test_acc"] - uniform_eval["test_acc"]
        ),
        "late_over_early": schedule["late_over_early"],
        "rho_spent": schedule["rho_spent"],
        "rho_violation": schedule["rho_violation"],
    }
    schedule_rows = [
        {"seed": seed, "rho": rho, "t": t, "q": value}
        for t, value in enumerate(q)
    ]
    return row, schedule_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/submission_v5_proxy_transfer")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    parser.add_argument("--seeds", default="123,321,777,1001,2027,409,613,887,1301,1601")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--q-min", type=float, default=1e-5)
    parser.add_argument("--label-contamination", type=float, default=0.1)
    parser.add_argument("--feature-contrast", type=float, default=0.9)
    parser.add_argument("--feature-noise-std", type=float, default=0.15)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fashion.load_preprocessed_fashion_mnist(str(data_dir.resolve()))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != "resume"} | {
        "protocol_version": 5,
        "public_design_records": 8000,
        "fashion_parameters": 6634,
        "fashion_active_parameters_every_step": 6634,
        "fashion_gaussian_noise_dimension": 6634,
        "fashion_active_mask": "all-ones throughout",
        "shift": "attenuate normalized public-design features, add independent Gaussian measurement noise, and replace the prespecified fraction of public labels by a different class",
        "selection_data": "shifted public design split only",
        "public_pretraining": "unchanged disjoint public auxiliary split",
        "evaluation_data": "unchanged disjoint private split",
        "evaluation_randomness": "fresh paths shared with uniform allocation",
    }
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        protocol_fields = [
            "data_dir",
            "seeds",
            "rho",
            "q_min",
            "label_contamination",
            "feature_contrast",
            "feature_noise_std",
            "K",
            "alpha",
            "protocol_version",
            "shift",
            "selection_data",
            "evaluation_data",
            "evaluation_randomness",
        ]
        if any(previous.get(field) != config.get(field) for field in protocol_fields):
            raise ValueError("the saved proxy-transfer protocol does not match this resume request")
    temporary_config = config_path.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary_config.replace(config_path)
    jobs = [
        (
            int(seed),
            data_dir,
            args.rho,
            args.q_min,
            args.label_contamination,
            args.feature_contrast,
            args.feature_noise_std,
            args.K,
            args.alpha,
        )
        for seed in args.seeds.split(",")
        if seed
    ]
    rows: list[dict] = []
    schedules: list[dict] = []
    raw_path = out_dir / "proxy_transfer_raw.csv"
    schedule_path = out_dir / "proxy_transfer_schedules.csv"
    if args.resume and raw_path.exists() and schedule_path.exists():
        rows, schedules = retain_complete_cases(
            read_csv(raw_path),
            read_csv(schedule_path),
        )
        if rows:
            checkpoint(out_dir, rows, schedules)
    completed = {int(row["seed"]) for row in rows}
    pending_jobs = [job for job in jobs if int(job[0]) not in completed]

    if args.workers == 1:
        iterator = (run_seed(*job) for job in pending_jobs)
        for result in iterator:
            rows.append(result[0])
            schedules.extend(result[1])
            checkpoint(out_dir, rows, schedules)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(lambda job: run_seed(*job), pending_jobs):
                rows.append(result[0])
                schedules.extend(result[1])
                checkpoint(out_dir, rows, schedules)
    if not pending_jobs and rows:
        checkpoint(out_dir, rows, schedules)
    result_summary = summarize(rows)[0]
    result_summary["resumed_cases"] = len(completed)
    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()
