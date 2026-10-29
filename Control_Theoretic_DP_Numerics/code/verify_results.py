from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from experiment_utils import (
    holm_adjusted_pvalues,
    student_t_quantile,
    student_t_two_sided_pvalue,
)


TASKS = {"tabular", "fashion-standard", "fashion-lr_restart"}
FIXED_METHODS = {"Uniform", "Linear late", "Linear front", "Cosine late"}
DATA_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
SUMMARY_METRICS = [
    "design_objective",
    "validation_objective",
    "terminal_mean",
    "test_loss",
    "test_acc",
    "reference_test_loss",
    "reference_test_acc",
    "rho_violation",
    "rho_slack",
    "late_over_early",
    "max_over_avg",
    "epsilon_target",
    "epsilon_actual",
    "time_sec",
    "tuning_time_sec",
]
PAPER_ARTIFACTS = [
    "numerical_utility_tabular.dat",
    "numerical_utility_fashion_constant.dat",
    "numerical_utility_fashion_restart.dat",
    "numerical_penalty_tabular.dat",
    "numerical_penalty_fashion_constant.dat",
    "numerical_penalty_fashion_restart.dat",
    "numerical_cost_summary_table.tex",
    "numerical_direct_checks_table.tex",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(row: dict[str, str], field: str, source: Path) -> float:
    value = row.get(field, "")
    if value == "":
        raise AssertionError(f"missing {field} in {source}")
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"non-finite {field} in {source}")
    return result


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(actual), abs(expected))


def require_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not close(actual, expected, tolerance):
        raise AssertionError(f"{label}: expected {expected:.17g}, found {actual:.17g}")


def expected_epsilon(rho: float, delta: float) -> float:
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    half_width = student_t_quantile(0.975, len(values) - 1) * standard_error
    return mean, mean - half_width, mean + half_width


def verify_data(root: Path) -> dict[str, object]:
    data_dir = root / "data" / "fashion_mnist"
    verified = {}
    for name, expected in DATA_SHA256.items():
        path = data_dir / name
        if not path.is_file():
            raise AssertionError(f"missing Fashion-MNIST file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Fashion-MNIST checksum mismatch: {path}")
        verified[name] = actual
    return {"files": len(verified), "sha256": verified}


def verify_protocol_config(config: dict, source: Path) -> None:
    record = json.dumps(config).lower()
    for phrase in ["public", "certified upper bound", "delta_exact"]:
        if phrase not in record:
            raise AssertionError(f"protocol marker {phrase!r} missing from {source}")
    if "replace-one" not in record and "replacement adjacency" not in record:
        raise AssertionError(f"adjacency marker missing from {source}")
    if "2 * c_clip / batch_size" not in record and "2 * clip / batch_size" not in record:
        raise AssertionError(f"sensitivity calibration missing from {source}")
    if "privacy_amplification" not in config or "none" not in str(config["privacy_amplification"]).lower():
        raise AssertionError(f"privacy-amplification policy missing from {source}")


def verify_representative_case(path: Path, expected_rho: float) -> dict[str, object]:
    summary_path = path / "summary.csv"
    schedules_path = path / "schedules.csv"
    rows = read_csv(summary_path)
    config = read_json(path / "config.json")
    verify_protocol_config(config, path / "config.json")
    expected_counts = {"PB-AMD": 1, "PGD": 4, "Penalty": 6, "Uniform": 1}
    counts = Counter(row["family"] for row in rows)
    if len(rows) != 12 or counts != expected_counts:
        raise AssertionError(f"unexpected representative method grid in {path}: {counts}")
    if config.get("dataset") == "Fashion-MNIST":
        parameters = int(config["parameters"])
        if parameters != 6634 or int(config["gaussian_noise_dimension"]) != parameters:
            raise AssertionError(f"Fashion-MNIST run is not full-dimensional: {path}")
        if "all-ones" not in str(config["active_mask"]):
            raise AssertionError(f"Fashion-MNIST active mask is incomplete: {path}")
    rho = float(config["rho_max"])
    require_close(rho, expected_rho, f"representative rho in {path}")
    delta = float(config["delta"])
    sensitivity = 2.0 * float(config["clip"]) / int(config["batch_size"])
    require_close(float(config["sensitivity"]), sensitivity, f"sensitivity in {path}", 1e-13)
    require_close(
        float(config["epsilon_target"]),
        expected_epsilon(rho, delta),
        f"epsilon conversion in {path}",
        1e-12,
    )

    with schedules_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        schedules = list(reader)
        methods = [] if reader.fieldnames is None else reader.fieldnames[1:]
    if len(schedules) != int(config["T"]) or set(methods) != {row["method"] for row in rows}:
        raise AssertionError(f"schedule matrix does not match the representative summary: {path}")

    q_min = float(config["q_min"])
    maximum_prox_error = 0.0
    maximum_kkt_residual = 0.0
    for row in rows:
        for field in [
            "design_objective",
            "validation_objective",
            "terminal_mean",
            "test_loss",
            "test_acc",
            "rho_spent",
            "epsilon_actual",
        ]:
            finite(row, field, summary_path)
        require_close(
            float(row["epsilon_actual"]),
            expected_epsilon(float(row["rho_spent"]), delta),
            f"actual epsilon for {row['method']}",
            1e-12,
        )
        schedule = [float(item[row["method"]]) for item in schedules]
        require_close(sum(schedule), float(row["rho_spent"]), f"schedule sum for {row['method']}", 1e-11)
        require_close(min(schedule), float(row["min_q"]), f"schedule minimum for {row['method']}", 1e-11)
        if row["family"] in {"PB-AMD", "PGD", "Uniform"}:
            require_close(float(row["rho_spent"]), rho, f"exact budget for {row['method']}", 1e-9)
            if min(schedule) < q_min - 1e-11:
                raise AssertionError(f"schedule-floor violation for {row['method']}")
        if row["family"] in {"PB-AMD", "PGD"}:
            current_error = finite(row, "max_prox_feasibility_error", summary_path)
            maximum_prox_error = max(maximum_prox_error, current_error)
            if current_error > 1e-10:
                raise AssertionError(f"proximal feasibility failure for {row['method']}")
        if row["family"] == "PB-AMD":
            maximum_kkt_residual = max(
                maximum_kkt_residual,
                finite(row, "saa_kkt_residual", summary_path),
            )

    for family, key in [
        ("PB-AMD", "selected_pb_method"),
        ("PGD", "selected_pgd_method"),
        ("Penalty", "selected_penalty_method"),
    ]:
        candidates = [row for row in rows if row["family"] == family]
        selected = config.get(key)
        if selected not in {row["method"] for row in candidates}:
            raise AssertionError(f"missing selected {family} candidate in {path}")
        comparison = candidates
        if family == "Penalty":
            feasible = [row for row in candidates if float(row["rho_spent"]) <= rho + 1e-10]
            comparison = feasible or candidates
        winner = min(comparison, key=lambda row: float(row["validation_objective"]))
        if selected != winner["method"]:
            raise AssertionError(f"{family} was not selected by the public-validation rule in {path}")
    return {
        "methods": len(rows),
        "horizon": len(schedules),
        "maximum_prox_feasibility_error": maximum_prox_error,
        "pb_saa_kkt_residual": maximum_kkt_residual,
    }


def verify_selection(
    group: list[dict[str, str]], prefix: str, selected_name: str, expected_candidates: int
) -> None:
    candidates = [row for row in group if row["method_family"].startswith(prefix)]
    selected = [row for row in group if row["method_family"] == selected_name]
    if len(candidates) != expected_candidates or len(selected) != 1:
        raise AssertionError(f"incomplete selection record for {selected_name}")
    winner = min(candidates, key=lambda row: float(row["validation_objective"]))
    chosen = selected[0]
    for field in ["beta", "lr", "alpha", "initialization"]:
        if chosen[field] != winner[field]:
            raise AssertionError(f"{selected_name} does not preserve the winning {field}")
    for field in ["validation_objective", "terminal_mean", "test_acc", "rho_spent"]:
        require_close(float(chosen[field]), float(winner[field]), f"{selected_name} {field}")


def verify_comprehensive(path: Path) -> dict[str, object]:
    raw_path = path / "raw_results.csv"
    summary_path = path / "summary_by_task_budget.csv"
    paired_path = path / "paired_comparisons.csv"
    raw = read_csv(raw_path)
    summary = read_csv(summary_path)
    paired = read_csv(paired_path)
    config = read_json(path / "config.json")
    verify_protocol_config(config, path / "config.json")
    if int(config.get("protocol_version", 0)) < 5:
        raise AssertionError("comprehensive protocol predates certified PB-AMD stopping")
    stopping = config.get("pb_amd_stopping", {})
    kkt_tolerance = float(stopping.get("kkt_tolerance", 0.0))
    if kkt_tolerance <= 0.0:
        raise AssertionError("comprehensive PB-AMD KKT tolerance is missing")
    code_dir = path.parents[1] / "code"
    source_hashes = config.get("source_sha256", {})
    expected_sources = {
        "comprehensive_schedule_benchmark.py",
        "dp_sgd_schedule_compare.py",
        "fashion_mnist_mlp_schedule_compare.py",
        "experiment_utils.py",
    }
    if set(source_hashes) != expected_sources:
        raise AssertionError("comprehensive source hashes are incomplete")
    for name, expected in source_hashes.items():
        source = code_dir / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"comprehensive source mismatch: {name}")
    fashion = config["fashion_settings"]
    if int(fashion["parameters"]) != 6634 or int(fashion["active_parameters_every_step"]) != 6634:
        raise AssertionError("comprehensive Fashion-MNIST mechanism is not full-dimensional")

    groups: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        groups[(row["task"], int(row["seed"]), float(row["rho"]))].append(row)
    if len(raw) != 2160 or len(groups) != 90 or {key[0] for key in groups} != TASKS:
        raise AssertionError("comprehensive task-seed-budget grid is incomplete")
    pgd_counts = {"tabular": 4, "fashion-standard": 4, "fashion-lr_restart": 4}
    maximum_exact_budget_error = 0.0
    maximum_pb_prox_error = 0.0
    maximum_pb_kkt_residual = 0.0
    maximum_pb_variational_gap = 0.0
    maximum_pb_iterations = 0
    for key, group in groups.items():
        task, _, rho = key
        families = [row["method_family"] for row in group]
        if len(group) != 24 or len(set(families)) != 24:
            raise AssertionError(f"incomplete or duplicated method grid for {key}")
        if not FIXED_METHODS <= set(families):
            raise AssertionError(f"missing fixed schedule for {key}")
        verify_selection(group, "PGD alpha=", "PGD tuned", pgd_counts[task])
        verify_selection(group, "Penalty beta=", "Penalty raw-tuned", 6)
        verify_selection(group, "Penalty repaired beta=", "Penalty repaired-tuned", 6)
        for row in group:
            for field in [
                "design_objective",
                "validation_objective",
                "terminal_mean",
                "test_acc",
                "rho_spent",
                "epsilon_actual",
            ]:
                finite(row, field, raw_path)
            delta = float(row["delta"])
            require_close(
                float(row["epsilon_actual"]),
                expected_epsilon(float(row["rho_spent"]), delta),
                f"epsilon conversion for {key}",
                1e-12,
            )
            batch_size = 256 if task == "tabular" else 192
            require_close(
                float(row["sensitivity"]),
                2.0 * float(row["clip"]) / batch_size,
                f"sensitivity for {key}",
                1e-13,
            )
            family = row["method_family"]
            exact = (
                family == "PB-AMD"
                or family == "PGD tuned"
                or family.startswith("PGD alpha=")
                or family in FIXED_METHODS
                or family.startswith("Penalty repaired")
            )
            if exact:
                error = abs(float(row["rho_spent"]) - rho)
                maximum_exact_budget_error = max(maximum_exact_budget_error, error)
                if error > 1e-9:
                    raise AssertionError(f"exact-feasibility failure for {key}, {family}")
            if family == "PB-AMD":
                prox_error = float(row["max_prox_feasibility_error"])
                maximum_pb_prox_error = max(maximum_pb_prox_error, prox_error)
                if prox_error > 1e-10:
                    raise AssertionError(f"PB-AMD prox feasibility failure for {key}")
                kkt_residual = finite(row, "saa_kkt_residual", raw_path)
                gap = finite(row, "saa_variational_gap", raw_path)
                iterations = int(float(row["iterations"]))
                maximum_pb_kkt_residual = max(maximum_pb_kkt_residual, kkt_residual)
                maximum_pb_variational_gap = max(maximum_pb_variational_gap, gap)
                maximum_pb_iterations = max(maximum_pb_iterations, iterations)
                minimum_iterations = int(
                    stopping[
                        "tabular_minimum_iterations"
                        if task == "tabular"
                        else "fashion_minimum_iterations"
                    ]
                )
                maximum_iterations = int(
                    stopping[
                        "tabular_maximum_iterations"
                        if task == "tabular"
                        else "fashion_maximum_iterations"
                    ]
                )
                if not minimum_iterations <= iterations <= maximum_iterations:
                    raise AssertionError(f"invalid PB-AMD iteration count for {key}")
                if kkt_residual > kkt_tolerance * (1.0 + 1e-10):
                    raise AssertionError(
                        f"PB-AMD KKT certificate failure for {key}: "
                        f"{kkt_residual} > {kkt_tolerance}"
                    )
                gap_bound = 2.0 * rho * kkt_tolerance + 1e-8
                if gap > gap_bound:
                    raise AssertionError(
                        f"PB-AMD variational-gap failure for {key}: {gap} > {gap_bound}"
                    )

    summary_index = {}
    for row in summary:
        key = (row["task"], float(row["rho"]), row["method_family"])
        if key in summary_index or int(row["n_seeds"]) != 10:
            raise AssertionError(f"invalid comprehensive summary row: {key}")
        summary_index[key] = row
        sample = [
            item
            for item in raw
            if item["task"] == key[0]
            and float(item["rho"]) == key[1]
            and item["method_family"] == key[2]
        ]
        if len(sample) != 10:
            raise AssertionError(f"summary sample is incomplete: {key}")
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in sample]
            require_close(
                float(row[f"{metric}_mean"]),
                statistics.fmean(values),
                f"summary mean for {key}, {metric}",
            )
            require_close(
                float(row[f"{metric}_std"]),
                statistics.stdev(values),
                f"summary standard deviation for {key}, {metric}",
            )
    if len(summary) != 216:
        raise AssertionError(f"expected 216 summary rows, found {len(summary)}")

    primary_rows = {"terminal_mean": [], "test_acc": []}
    for row in paired:
        task = row["task"]
        rho = float(row["rho"])
        metric = row["metric"]
        comparator = row["comparator"]
        if int(row["n_pairs"]) != 10:
            raise AssertionError("paired comparison has the wrong sample size")
        pb = {
            int(item["seed"]): float(item[metric])
            for item in raw
            if item["task"] == task and float(item["rho"]) == rho and item["method_family"] == "PB-AMD"
        }
        baseline = {
            int(item["seed"]): float(item[metric])
            for item in raw
            if item["task"] == task and float(item["rho"]) == rho and item["method_family"] == comparator
        }
        differences = [pb[seed] - baseline[seed] for seed in sorted(pb)]
        mean, low, high = paired_interval(differences)
        require_close(float(row["mean_difference_pb_minus_comparator"]), mean, "paired mean")
        require_close(float(row["ci95_low"]), low, "paired lower confidence bound")
        require_close(float(row["ci95_high"]), high, "paired upper confidence bound")
        standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
        pvalue = (
            0.0
            if standard_error == 0.0 and mean != 0.0
            else 1.0
            if standard_error == 0.0
            else student_t_two_sided_pvalue(mean / standard_error, len(differences) - 1)
        )
        require_close(float(row["pvalue_two_sided"]), pvalue, "paired p-value")
        if comparator == "Uniform":
            primary_rows[metric].append(row)
    if len(paired) != 72:
        raise AssertionError(f"expected 72 paired comparisons, found {len(paired)}")
    for metric, rows in primary_rows.items():
        if len(rows) != 9:
            raise AssertionError(f"wrong Holm family size for {metric}")
        adjusted = holm_adjusted_pvalues([float(row["pvalue_two_sided"]) for row in rows])
        for row, expected in zip(rows, adjusted):
            require_close(float(row["holm_adjusted_pvalue"]), expected, "Holm-adjusted p-value")
            if int(row["holm_reject_0_05"]) != int(expected <= 0.05):
                raise AssertionError("Holm decision mismatch")

    raw_penalty = [row for row in raw if row["method_family"].startswith("Penalty beta=")]
    repaired_penalty = {
        (row["task"], int(row["seed"]), float(row["rho"]), float(row["beta"])): row
        for row in raw
        if row["method_family"].startswith("Penalty repaired beta=")
    }
    if len(raw_penalty) != 540 or len(repaired_penalty) != 540:
        raise AssertionError("raw/repaired penalty grids are incomplete")
    charge_ratios = [float(row["rho_spent"]) / float(row["rho"]) for row in raw_penalty]
    exact_raw = sum(
        abs(float(row["rho_spent"]) - float(row["rho"]))
        <= 1e-10 * max(1.0, float(row["rho"]))
        for row in raw_penalty
    )
    selected_betas = Counter(
        float(row["beta"]) for row in raw if row["method_family"] == "Penalty repaired-tuned"
    )
    terminal_differences = []
    for task in TASKS:
        for rho in [0.25, 0.5, 1.0]:
            pb_mean = statistics.fmean(
                float(row["terminal_mean"])
                for row in raw
                if row["task"] == task and float(row["rho"]) == rho and row["method_family"] == "PB-AMD"
            )
            penalty_mean = statistics.fmean(
                float(row["terminal_mean"])
                for row in raw
                if row["task"] == task
                and float(row["rho"]) == rho
                and row["method_family"] == "Penalty repaired-tuned"
            )
            terminal_differences.append(100.0 * abs(penalty_mean - pb_mean) / abs(pb_mean))
    return {
        "raw_rows": len(raw),
        "summary_rows": len(summary),
        "paired_rows": len(paired),
        "task_seed_budget_combinations": len(groups),
        "maximum_exact_budget_error": maximum_exact_budget_error,
        "maximum_pb_prox_feasibility_error": maximum_pb_prox_error,
        "maximum_pb_saa_kkt_residual": maximum_pb_kkt_residual,
        "maximum_pb_saa_variational_gap": maximum_pb_variational_gap,
        "maximum_pb_iterations": maximum_pb_iterations,
        "raw_penalty_candidates": len(raw_penalty),
        "raw_penalty_exact_budget_candidates": exact_raw,
        "raw_penalty_charge_ratio_minimum": min(charge_ratios),
        "raw_penalty_charge_ratio_median": statistics.median(charge_ratios),
        "raw_penalty_charge_ratio_maximum": max(charge_ratios),
        "selected_penalty_beta_counts": {
            str(beta): count for beta, count in sorted(selected_betas.items())
        },
        "maximum_repaired_penalty_vs_pb_terminal_difference_percent": max(terminal_differences),
    }


def verify_published(path: Path, comprehensive_path: Path) -> dict[str, object]:
    raw = read_csv(path / "published_schedule_raw.csv")
    candidates = read_csv(path / "published_schedule_candidates.csv")
    schedules = read_csv(path / "published_schedule_schedules.csv")
    summary = read_csv(path / "published_schedule_summary.csv")
    config = read_json(path / "config.json")
    if "10.1145/3531146.3533070" not in str(config.get("source_method", "")):
        raise AssertionError("published schedule source is missing")
    ratios = sorted(float(item) for item in str(config["ratios"]).split(","))
    raw_groups = {(row["task"], int(row["seed"]), float(row["rho"])): row for row in raw}
    candidate_groups: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    schedule_groups: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidate_groups[(row["task"], int(row["seed"]), float(row["rho"]))].append(row)
    for row in schedules:
        schedule_groups[(row["task"], int(row["seed"]), float(row["rho"]))].append(row)
    if len(raw) != 90 or len(raw_groups) != 90 or len(candidates) != 360 or len(schedules) != 4500:
        raise AssertionError("published-schedule grid is incomplete")

    comprehensive = read_csv(comprehensive_path / "raw_results.csv")
    uniform = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in comprehensive
        if row["method_family"] == "Uniform"
    }
    pb = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in comprehensive
        if row["method_family"] == "PB-AMD"
    }
    max_budget_error = 0.0
    max_shared_uniform_error = 0.0
    for key, row in raw_groups.items():
        current_candidates = candidate_groups[key]
        if sorted(float(item["ratio"]) for item in current_candidates) != ratios:
            raise AssertionError(f"published-schedule candidate grid mismatch for {key}")
        winner = min(current_candidates, key=lambda item: (float(item["validation_objective"]), float(item["ratio"])))
        if int(winner["selected"]) != 1 or float(winner["ratio"]) != float(row["selected_ratio"]):
            raise AssertionError(f"published-schedule public selection mismatch for {key}")
        current_schedule = schedule_groups[key]
        horizon = 60 if key[0] == "tabular" else 45
        if len(current_schedule) != horizon:
            raise AssertionError(f"published schedule has the wrong horizon for {key}")
        values = [float(item["q"]) for item in current_schedule]
        budget_error = abs(sum(values) - key[2])
        max_budget_error = max(max_budget_error, budget_error)
        if min(values) < float(config["q_min"]) - 1e-12 or budget_error > 1e-11:
            raise AssertionError(f"published schedule is infeasible for {key}")
        shared_error = max(
            abs(float(row["uniform_terminal_mean"]) - float(uniform[key]["terminal_mean"])),
            abs(float(row["uniform_test_acc"]) - float(uniform[key]["test_acc"])),
        )
        max_shared_uniform_error = max(max_shared_uniform_error, shared_error)
        if shared_error > 1e-12:
            raise AssertionError(f"published comparator does not share the evaluation path for {key}")
    if len(summary) != 9:
        raise AssertionError("published-schedule summary grid is incomplete")
    restart_reductions = []
    for rho in [0.25, 0.5, 1.0]:
        keys = [key for key in raw_groups if key[0] == "fashion-lr_restart" and key[2] == rho]
        published_mean = statistics.fmean(float(raw_groups[key]["terminal_mean"]) for key in keys)
        pb_mean = statistics.fmean(float(pb[key]["terminal_mean"]) for key in keys)
        restart_reductions.append(100.0 * (published_mean - pb_mean) / published_mean)
    return {
        "raw_rows": len(raw),
        "candidate_rows": len(candidates),
        "schedule_rows": len(schedules),
        "maximum_budget_error": max_budget_error,
        "maximum_shared_uniform_error": max_shared_uniform_error,
        "restart_pb_vs_published_reduction_percent_minimum": min(restart_reductions),
        "restart_pb_vs_published_reduction_percent_maximum": max(restart_reductions),
    }


def verify_proxy(path: Path, comprehensive_path: Path) -> dict[str, object]:
    raw = read_csv(path / "proxy_transfer_raw.csv")
    schedules = read_csv(path / "proxy_transfer_schedules.csv")
    summary = read_csv(path / "proxy_transfer_summary.csv")
    config = read_json(path / "config.json")
    if len(raw) != 10 or len(schedules) != 450 or len(summary) != 1:
        raise AssertionError("proxy-transfer result grid is incomplete")
    if int(config["fashion_parameters"]) != 6634:
        raise AssertionError("proxy-transfer Fashion-MNIST mechanism is not full-dimensional")
    comprehensive = read_csv(comprehensive_path / "raw_results.csv")
    uniform = {
        (int(row["seed"]), float(row["rho"])): row
        for row in comprehensive
        if row["task"] == "fashion-lr_restart" and row["method_family"] == "Uniform"
    }
    schedule_groups: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
    for row in schedules:
        schedule_groups[(int(row["seed"]), float(row["rho"]))].append(row)
    reductions = []
    accuracy_gains = []
    maximum_budget_error = 0.0
    maximum_shared_uniform_error = 0.0
    for row in raw:
        key = (int(row["seed"]), float(row["rho"]))
        values = [float(item["q"]) for item in schedule_groups[key]]
        if len(values) != 45 or min(values) < float(config["q_min"]) - 1e-12:
            raise AssertionError(f"invalid proxy-transfer schedule for {key}")
        maximum_budget_error = max(maximum_budget_error, abs(sum(values) - key[1]))
        reduction = 100.0 * (float(row["uniform_terminal_mean"]) - float(row["terminal_mean"])) / float(row["uniform_terminal_mean"])
        accuracy = 100.0 * (float(row["test_acc"]) - float(row["uniform_test_acc"]))
        require_close(float(row["terminal_reduction_percent"]), reduction, "proxy terminal reduction")
        require_close(float(row["accuracy_difference_pp"]), accuracy, "proxy accuracy difference")
        reductions.append(reduction)
        accuracy_gains.append(accuracy)
        shared_error = max(
            abs(float(row["uniform_terminal_mean"]) - float(uniform[key]["terminal_mean"])),
            abs(float(row["uniform_test_acc"]) - float(uniform[key]["test_acc"])),
        )
        maximum_shared_uniform_error = max(maximum_shared_uniform_error, shared_error)
    reduction_summary = paired_interval(reductions)
    accuracy_summary = paired_interval(accuracy_gains)
    summary_row = summary[0]
    expected = {
        "late_over_early_mean": statistics.fmean(float(row["late_over_early"]) for row in raw),
        "terminal_reduction_percent": reduction_summary[0],
        "terminal_reduction_ci95_low": reduction_summary[1],
        "terminal_reduction_ci95_high": reduction_summary[2],
        "accuracy_difference_pp": accuracy_summary[0],
        "accuracy_difference_ci95_low": accuracy_summary[1],
        "accuracy_difference_ci95_high": accuracy_summary[2],
    }
    for field, value in expected.items():
        require_close(float(summary_row[field]), value, f"proxy summary {field}")
    if maximum_budget_error > 1e-11 or maximum_shared_uniform_error > 1e-12:
        raise AssertionError("proxy-transfer feasibility or shared-path check failed")
    return {
        "raw_rows": len(raw),
        "schedule_rows": len(schedules),
        "maximum_budget_error": maximum_budget_error,
        "maximum_shared_uniform_error": maximum_shared_uniform_error,
        **expected,
    }


def compatible_runtime_source_digest(source: bytes, expected: str) -> str:
    actual = hashlib.sha256(source).hexdigest()
    if actual == expected:
        return actual
    marker = b"def exponential_privacy_schedule"
    next_marker = b"def project_lower_bounded_simplex"
    if marker in source and next_marker in source:
        start = source.index(marker)
        end = source.index(next_marker)
        compatible = hashlib.sha256(source[:start] + source[end:]).hexdigest()
        if compatible == expected:
            return compatible
    return actual


def verify_runtime(path: Path, code_dir: Path) -> dict[str, object]:
    rows = read_csv(path / "runtime_runs.csv")
    config = read_json(path / "config.json")
    if len(rows) != 27 or int(config["repeats"]) != 3:
        raise AssertionError("controlled runtime grid is incomplete")
    for name, expected in config["source_sha256"].items():
        actual = compatible_runtime_source_digest((code_dir / name).read_bytes(), expected)
        if actual != expected:
            raise AssertionError(f"runtime source mismatch: {name}")
    expected_candidates = {"PB-AMD": 1, "PGD grid": 4, "Penalty grid": 6}
    maximum_identity_error = 0.0
    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        family = row["family"]
        if int(row["candidates"]) != expected_candidates[family]:
            raise AssertionError(f"wrong runtime candidate count for {family}")
        if int(row["adjoint_iterations"]) != int(row["K"]) * int(row["candidates"]):
            raise AssertionError(f"wrong runtime adjoint count for {family}")
        solver = float(row["solver_time_sec"])
        validation = float(row["validation_time_sec"])
        pipeline = float(row["pipeline_time_sec"])
        maximum_identity_error = max(maximum_identity_error, abs(pipeline - solver - validation))
        repeat = int(row["repeat"])
        if repeat in grouped[row["task"]][family]:
            raise AssertionError(f"duplicate runtime repeat for {row['task']}, {family}, {repeat}")
        grouped[row["task"]][family][repeat] = pipeline
        require_close(float(row["selected_rho_spent"]), float(config["rho"]), "runtime selected budget", 1e-9)
    if maximum_identity_error > 1e-9:
        raise AssertionError("controlled pipeline timings do not add up")
    relative_medians = {}
    for task, families in grouped.items():
        repeats = sorted(families["PB-AMD"])
        if any(sorted(values) != repeats for values in families.values()):
            raise AssertionError(f"runtime repeats are not paired for {task}")
        relative_medians[task] = {
            family: statistics.median(
                values[repeat] / families["PB-AMD"][repeat] for repeat in repeats
            )
            for family, values in families.items()
        }
    return {
        "rows": len(rows),
        "repeats": int(config["repeats"]),
        "maximum_pipeline_identity_error": maximum_identity_error,
        "relative_pipeline_time_medians": relative_medians,
    }


def verify_diagnostics(diagnostics_path: Path, lq_path: Path) -> dict[str, object]:
    diagnostics = read_json(diagnostics_path)
    lq = read_json(lq_path)
    tabular = diagnostics["tabular"]
    fashion = diagnostics["fashion_mnist"]
    if int(fashion["parameters"]) != 6634 or int(fashion["horizon"]) != 45:
        raise AssertionError("Fashion-MNIST derivative diagnostic is not full-dimensional")
    derivative_errors = [
        float(tabular["directional_derivative_max_relative_error"]),
        float(fashion["schedule_directional_derivative_max_relative_error"]),
        float(fashion["gradient_vjp_max_relative_error"]),
    ]
    if max(derivative_errors) > 1e-4:
        raise AssertionError("adjoint/finite-difference validation failed")
    for record in [tabular, fashion]:
        if float(record["observed_replace_one_change"]) > float(record["calibrated_replace_one_bound"]):
            raise AssertionError("replace-one sensitivity check failed")
    if float(lq["alpha"]) > float(lq["relative_smoothness_step_limit"]):
        raise AssertionError("LQ validation step exceeds the theoretical limit")
    if (
        abs(float(lq["relative_objective_gap"])) > 1e-10
        or float(lq["schedule_l1_error"]) > 1e-6
        or float(lq["max_budget_error"]) > 1e-12
    ):
        raise AssertionError("LQ analytic-solution validation failed")
    for field in ["relative_objective_gap", "schedule_l1_error", "max_budget_error"]:
        require_close(float(diagnostics["linear_quadratic"][field]), float(lq[field]), f"LQ diagnostic {field}")
    return {
        "fashion_parameters": int(fashion["parameters"]),
        "fashion_horizon": int(fashion["horizon"]),
        "maximum_derivative_relative_error": max(derivative_errors),
        "lq_relative_objective_gap": float(lq["relative_objective_gap"]),
        "lq_schedule_l1_error": float(lq["schedule_l1_error"]),
        "lq_maximum_budget_error": float(lq["max_budget_error"]),
    }


def verify_paper_artifacts(root: Path, results: Path) -> dict[str, object]:
    exporter = root / "code" / "export_concise_numerical_latex.py"
    with tempfile.TemporaryDirectory(prefix="dp_sgd_export_") as temporary:
        output = Path(temporary)
        command = [
            sys.executable,
            str(exporter),
            "--raw",
            str(results / "comprehensive" / "raw_results.csv"),
            "--published",
            str(results / "published_schedule" / "published_schedule_raw.csv"),
            "--runtime",
            str(results / "controlled_runtime" / "runtime_runs.csv"),
            "--lq",
            str(results / "linear_quadratic" / "result.json"),
            "--out-dir",
            str(output),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        for name in PAPER_ARTIFACTS:
            archived = root / "paper_artifacts" / name
            generated = output / name
            if not archived.is_file() or generated.read_bytes() != archived.read_bytes():
                raise AssertionError(f"paper artifact is not regenerated exactly: {name}")
    return {"files_regenerated_exactly": len(PAPER_ARTIFACTS)}


def comparable_number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def compare_csv_to_reference(
    candidate: Path,
    reference: Path,
    key_fields: tuple[str, ...],
    ignored_fields: set[str] | None = None,
    tolerance: float = 1e-8,
) -> float:
    ignored = ignored_fields or set()
    candidate_rows = read_csv(candidate)
    reference_rows = read_csv(reference)

    def index(rows: list[dict[str, str]], source: Path) -> dict[tuple[str, ...], dict[str, str]]:
        result = {}
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            if key in result:
                raise AssertionError(f"duplicate comparison key {key} in {source}")
            result[key] = row
        return result

    candidate_index = index(candidate_rows, candidate)
    reference_index = index(reference_rows, reference)
    if candidate_index.keys() != reference_index.keys():
        raise AssertionError(f"candidate/reference key mismatch: {candidate} versus {reference}")
    maximum_scaled_error = 0.0
    for key in candidate_index:
        current = candidate_index[key]
        expected = reference_index[key]
        fields = (set(current) | set(expected)) - ignored
        for field in fields:
            if field not in current or field not in expected:
                raise AssertionError(f"field mismatch for {field} at {key}")
            if current[field] == expected[field]:
                continue
            current_number = comparable_number(current[field])
            expected_number = comparable_number(expected[field])
            if current_number is None or expected_number is None:
                raise AssertionError(
                    f"text mismatch for {field} at {key}: {current[field]!r} versus {expected[field]!r}"
                )
            scale = max(1.0, abs(current_number), abs(expected_number))
            scaled_error = abs(current_number - expected_number) / scale
            maximum_scaled_error = max(maximum_scaled_error, scaled_error)
            if scaled_error > tolerance:
                raise AssertionError(
                    f"scientific result mismatch for {field} at {key}: "
                    f"{current_number:.17g} versus {expected_number:.17g}"
                )
    return maximum_scaled_error


def compare_json_values(candidate: object, reference: object, label: str, tolerance: float = 1e-8) -> float:
    if isinstance(candidate, dict) and isinstance(reference, dict):
        if candidate.keys() != reference.keys():
            raise AssertionError(f"JSON field mismatch at {label}")
        return max(
            (compare_json_values(candidate[key], reference[key], f"{label}.{key}", tolerance) for key in candidate),
            default=0.0,
        )
    if isinstance(candidate, list) and isinstance(reference, list):
        if len(candidate) != len(reference):
            raise AssertionError(f"JSON length mismatch at {label}")
        return max(
            (compare_json_values(a, b, f"{label}[{index}]", tolerance) for index, (a, b) in enumerate(zip(candidate, reference))),
            default=0.0,
        )
    if isinstance(candidate, (int, float)) and isinstance(reference, (int, float)):
        scale = max(1.0, abs(float(candidate)), abs(float(reference)))
        error = abs(float(candidate) - float(reference)) / scale
        if error > tolerance:
            raise AssertionError(f"scientific result mismatch at {label}: {candidate} versus {reference}")
        return error
    if candidate != reference:
        raise AssertionError(f"value mismatch at {label}: {candidate!r} versus {reference!r}")
    return 0.0


def verify_against_reference(candidate: Path, reference: Path) -> dict[str, object]:
    timing_fields = {
        "time_sec",
        "tuning_time_sec",
        "time_sec_mean",
        "time_sec_std",
        "tuning_time_sec_mean",
        "tuning_time_sec_std",
        "solver_time_sec",
        "optimization_wall_time_sec",
        "validation_time_sec",
        "pipeline_time_sec",
    }
    specifications = [
        ("representative/tabular/summary.csv", ("method",), {"time_sec"}),
        ("representative/tabular/schedules.csv", ("t",), set()),
        ("representative/tabular/optimization_traces.csv", ("method", "iteration"), set()),
        ("representative/fashion_constant/summary.csv", ("method",), {"time_sec"}),
        ("representative/fashion_constant/schedules.csv", ("t",), set()),
        ("representative/fashion_constant/optimization_traces.csv", ("method", "iteration"), set()),
        ("representative/fashion_restart/summary.csv", ("method",), {"time_sec"}),
        ("representative/fashion_restart/schedules.csv", ("t",), set()),
        ("representative/fashion_restart/optimization_traces.csv", ("method", "iteration"), set()),
        ("comprehensive/raw_results.csv", ("task", "seed", "rho", "method_family"), timing_fields),
        ("comprehensive/summary_by_task_budget.csv", ("task", "rho", "method_family"), timing_fields),
        ("comprehensive/paired_comparisons.csv", ("task", "rho", "comparator", "metric"), set()),
        ("published_schedule/published_schedule_raw.csv", ("task", "seed", "rho"), set()),
        ("published_schedule/published_schedule_candidates.csv", ("task", "seed", "rho", "ratio"), set()),
        ("published_schedule/published_schedule_schedules.csv", ("task", "seed", "rho", "t"), set()),
        ("published_schedule/published_schedule_summary.csv", ("task", "rho"), set()),
        ("proxy_transfer/proxy_transfer_raw.csv", ("seed", "rho"), set()),
        ("proxy_transfer/proxy_transfer_schedules.csv", ("seed", "rho", "t"), set()),
        ("proxy_transfer/proxy_transfer_summary.csv", ("rho",), set()),
        ("controlled_runtime/runtime_runs.csv", ("task", "repeat", "family"), timing_fields),
        ("linear_quadratic/convergence.csv", ("iteration",), set()),
    ]
    maximum_scaled_error = 0.0
    for relative, keys, ignored in specifications:
        maximum_scaled_error = max(
            maximum_scaled_error,
            compare_csv_to_reference(candidate / relative, reference / relative, keys, ignored),
        )
    for relative in ["validation_diagnostics.json", "linear_quadratic/result.json"]:
        maximum_scaled_error = max(
            maximum_scaled_error,
            compare_json_values(
                read_json(candidate / relative),
                read_json(reference / relative),
                relative,
            ),
        )
    return {
        "scientific_files_compared": len(specifications) + 2,
        "wall_clock_fields_excluded": sorted(timing_fields),
        "maximum_scaled_numerical_difference": maximum_scaled_error,
        "relative_tolerance": 1e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the numerical evidence used in the paper.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/verification_report.json")
    parser.add_argument("--reference-results")
    parser.add_argument("--skip-artifact-check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    results = Path(args.results)
    if not results.is_absolute():
        results = root / results
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    report = {
        "status": "verified",
        "data": verify_data(root),
        "representative": {
            "tabular": verify_representative_case(results / "representative" / "tabular", 0.5),
            "fashion_constant": verify_representative_case(results / "representative" / "fashion_constant", 0.5),
            "fashion_restart": verify_representative_case(results / "representative" / "fashion_restart", 0.5),
        },
        "comprehensive": verify_comprehensive(results / "comprehensive"),
        "published_schedule": verify_published(results / "published_schedule", results / "comprehensive"),
        "proxy_transfer": verify_proxy(results / "proxy_transfer", results / "comprehensive"),
        "controlled_runtime": verify_runtime(results / "controlled_runtime", root / "code"),
        "diagnostics": verify_diagnostics(
            results / "validation_diagnostics.json",
            results / "linear_quadratic" / "result.json",
        ),
    }
    if not args.skip_artifact_check:
        report["paper_artifacts"] = verify_paper_artifacts(root, results)
    if args.reference_results:
        reference = Path(args.reference_results)
        if not reference.is_absolute():
            reference = root / reference
        report["reference_comparison"] = verify_against_reference(results, reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Verified numerical bundle. Report: {out}")


if __name__ == "__main__":
    main()
