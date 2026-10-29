from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from experiment_utils import holm_adjusted_pvalues, student_t_quantile


TASK_LABELS = {
    "tabular": "Tabular",
    "fashion-standard": "Fashion--constant",
    "fashion-lr_restart": "Fashion--restart",
}
TASK_ORDER = list(TASK_LABELS)
RUNTIME_FAMILIES = ["PB-AMD", "PGD grid", "Penalty grid"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def effect(value: float) -> str:
    digits = 4 if abs(value) < 1.0 else 3
    return f"{value:.{digits}f}"


def scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def interval(mean: float, low: float, high: float, scale: float = 1.0) -> str:
    return (
        f"${effect(scale * mean)}$ "
        f"$[{effect(scale * low)},\\,{effect(scale * high)}]$"
    )


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("a paired interval requires at least one value")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / len(values) ** 0.5
    half_width = student_t_quantile(0.975, len(values) - 1) * standard_error
    return mean, mean - half_width, mean + half_width


def write_space_table(
    path: Path,
    columns: list[str],
    rows: list[list[float]],
) -> None:
    lines = [" ".join(columns)]
    lines.extend(" ".join(f"{value:.12g}" for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_utility_plot_data(
    raw: list[dict[str, str]],
    published: list[dict[str, str]],
    out_dir: Path,
) -> None:
    pb = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in raw
        if row["method_family"] == "PB-AMD"
    }
    uniform = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in raw
        if row["method_family"] == "Uniform"
    }
    hwz = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in published
    }
    if len(pb) != 90 or len(uniform) != 90 or len(hwz) != 90:
        raise ValueError("utility plots require 90 unique PB-AMD, uniform, and HWZ cases")

    stems = {
        "tabular": "tabular",
        "fashion-standard": "fashion_constant",
        "fashion-lr_restart": "fashion_restart",
    }
    columns = [
        "rho",
        "uniform",
        "uniform_minus",
        "uniform_plus",
        "hwz",
        "hwz_minus",
        "hwz_plus",
    ]
    for task in TASK_ORDER:
        rows = []
        for rho in (0.25, 0.5, 1.0):
            keys = sorted(
                (
                    key
                    for key in pb
                    if key[0] == task
                    and key[2] == rho
                    and key in uniform
                    and key in hwz
                ),
                key=lambda key: key[1],
            )
            if len(keys) != 10:
                raise ValueError(f"incomplete utility plot data for {(task, rho)}")
            versus_uniform = paired_interval([
                100.0
                * (
                    float(uniform[key]["terminal_mean"])
                    - float(pb[key]["terminal_mean"])
                )
                / float(uniform[key]["terminal_mean"])
                for key in keys
            ])
            versus_hwz = paired_interval([
                100.0
                * (
                    float(hwz[key]["terminal_mean"])
                    - float(pb[key]["terminal_mean"])
                )
                / float(hwz[key]["terminal_mean"])
                for key in keys
            ])
            rows.append([
                rho,
                versus_uniform[0],
                versus_uniform[0] - versus_uniform[1],
                versus_uniform[2] - versus_uniform[0],
                versus_hwz[0],
                versus_hwz[0] - versus_hwz[1],
                versus_hwz[2] - versus_hwz[0],
            ])
        write_space_table(
            out_dir / f"numerical_utility_{stems[task]}.dat",
            columns,
            rows,
        )


def write_penalty_plot_data(
    raw: list[dict[str, str]],
    out_dir: Path,
) -> None:
    penalty = [
        row for row in raw if row["method_family"].startswith("Penalty beta=")
    ]
    if len(penalty) != 540:
        raise ValueError("penalty plots require all 540 raw candidates")
    betas = sorted({float(row["beta"]) for row in penalty})
    if betas != [1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0]:
        raise ValueError(f"unexpected penalty grid: {betas}")
    stems = {
        "tabular": "tabular",
        "fashion-standard": "fashion_constant",
        "fashion-lr_restart": "fashion_restart",
    }
    columns = ["beta", "median", "minimum", "maximum", "minus", "plus"]
    for task in TASK_ORDER:
        rows = []
        for beta in betas:
            values = [
                float(row["rho_spent"]) / float(row["rho"])
                for row in penalty
                if row["task"] == task and float(row["beta"]) == beta
            ]
            if len(values) != 30:
                raise ValueError(f"incomplete penalty plot data for {(task, beta)}")
            median = statistics.median(values)
            lower = min(values)
            upper = max(values)
            rows.append([beta, median, lower, upper, median - lower, upper - median])
        write_space_table(
            out_dir / f"numerical_penalty_{stems[task]}.dat",
            columns,
            rows,
        )


def write_core_effects(
    summary: list[dict[str, str]],
    paired: list[dict[str, str]],
    raw: list[dict[str, str]],
    published: list[dict[str, str]],
    path: Path,
) -> None:
    summary_index = {
        (row["task"], float(row["rho"]), row["method_family"]): row
        for row in summary
    }
    paired_index = {
        (row["task"], float(row["rho"]), row["comparator"], row["metric"]): row
        for row in paired
    }
    pb_raw = {
        (row["task"], float(row["rho"]), int(row["seed"])): row
        for row in raw
        if row["method_family"] == "PB-AMD"
    }
    published_raw = {
        (row["task"], float(row["rho"]), int(row["seed"])): row
        for row in published
    }
    primary_rows = [row for row in paired if row["comparator"] == "Uniform"]
    if len(primary_rows) != 18:
        raise ValueError("the primary analysis requires nine PB--uniform comparisons per endpoint")
    primary_adjusted: dict[tuple[str, float, str], float] = {}
    for metric in ("terminal_mean", "test_acc"):
        metric_rows = [row for row in primary_rows if row["metric"] == metric]
        if len(metric_rows) != 9:
            raise ValueError(f"unexpected primary family size for {metric}")
        adjusted = holm_adjusted_pvalues(
            [float(row["pvalue_two_sided"]) for row in metric_rows]
        )
        for row, value in zip(metric_rows, adjusted, strict=True):
            primary_adjusted[(row["task"], float(row["rho"]), metric)] = value
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Problem & $\\rho$ & PB late/early & $\\Delta_{\\rm term}^{\\rm unif}$ (paired 95\\% CI) & "
        "$\\Delta_{\\rm term}^{\\rm HWZ}$ (paired 95\\% CI) & "
        "$\\Delta_{\\rm acc}^{\\rm unif}$, p.p. (paired 95\\% CI) \\\\",
        "\\midrule",
    ]
    for task in TASK_ORDER:
        rhos = sorted({float(row["rho"]) for row in summary if row["task"] == task})
        if rhos != [0.25, 0.5, 1.0]:
            raise ValueError(f"unexpected budget grid for {task}: {rhos}")
        for index, rho in enumerate(rhos):
            pb = summary_index[(task, rho, "PB-AMD")]
            terminal = paired_index[(task, rho, "Uniform", "terminal_mean")]
            accuracy = paired_index[(task, rho, "Uniform", "test_acc")]
            keys = sorted(
                {
                    key
                    for key in pb_raw
                    if key[0] == task and key[1] == rho and key in published_raw
                },
                key=lambda key: key[2],
            )
            if len(keys) != 10:
                raise ValueError(
                    f"the HWZ comparison requires ten paired seeds for {(task, rho)}"
                )
            published_difference = paired_interval([
                float(pb_raw[key]["terminal_mean"])
                - float(published_raw[key]["terminal_mean"])
                for key in keys
            ])
            if int(terminal["n_pairs"]) != 10 or int(accuracy["n_pairs"]) != 10:
                raise ValueError("the concise table requires ten paired seeds")
            if primary_adjusted[(task, rho, "terminal_mean")] >= 0.05:
                raise ValueError("a primary terminal PB--uniform comparison no longer passes Holm correction")
            expected_accuracy_reject = task != "fashion-standard"
            if (
                primary_adjusted[(task, rho, "test_acc")] < 0.05
            ) != expected_accuracy_reject:
                raise ValueError("unexpected primary Holm decision for PB--uniform accuracy")
            task_cell = f"\\multirow{{3}}{{*}}{{{TASK_LABELS[task]}}}" if index == 0 else ""
            lines.append(
                f"{task_cell} & {rho:.2f} & {float(pb['late_over_early_mean']):.2f} & "
                f"{interval(float(terminal['mean_difference_pb_minus_comparator']), float(terminal['ci95_low']), float(terminal['ci95_high']))} & "
                f"{interval(*published_difference)} & "
                f"{interval(float(accuracy['mean_difference_pb_minus_comparator']), float(accuracy['ci95_low']), float(accuracy['ci95_high']), 100.0)} \\\\"
            )
        if task != TASK_ORDER[-1]:
            lines.append("\\addlinespace[0.25em]")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\caption{Core multi-seed effects.  The comparator columns use uniform allocation and the "
        "public-tuned, zCDP-matched geometric schedule adapted from Hong, Wang, and Zhou~\\cite{hong2022dynamic}.  "
        "Each terminal difference is PB-AMD minus the comparator, so negative values favor PB-AMD; "
        "$\\Delta_{\\rm acc}^{\\rm unif}$ is the accuracy difference in percentage points.  "
        "Each row uses ten paired seeds and common fresh evaluation paths.  Within the prespecified Holm-corrected "
        "PB-AMD--uniform family, all terminal and the tabular and restart accuracy comparisons are significant; "
        "the constant-stepsize accuracy comparisons are not.}",
        "\\label{tab:core-effects}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cost_summary(rows: list[dict[str, str]], path: Path) -> None:
    repeats = sorted({int(row["repeat"]) for row in rows})
    if repeats != [0, 1, 2]:
        raise ValueError(f"unexpected runtime repeats: {repeats}")
    ratios: dict[str, list[float]] = {family: [] for family in RUNTIME_FAMILIES}
    candidates: dict[str, int] = {}
    solves: dict[str, int] = {}
    for task in TASK_ORDER:
        indexed = {
            family: {
                int(row["repeat"]): row
                for row in rows
                if row["task"] == task and row["family"] == family
            }
            for family in RUNTIME_FAMILIES
        }
        if any(sorted(group) != repeats for group in indexed.values()):
            raise ValueError(f"incomplete runtime block for {task}")
        for repeat in repeats:
            base = float(indexed["PB-AMD"][repeat]["pipeline_time_sec"])
            for family in RUNTIME_FAMILIES:
                row = indexed[family][repeat]
                ratios[family].append(float(row["pipeline_time_sec"]) / base)
                family_candidates = int(row["candidates"])
                family_solves = int(row["adjoint_iterations"]) // int(row["K"])
                if family in candidates and candidates[family] != family_candidates:
                    raise ValueError(f"candidate count changed for {family}")
                if family in solves and solves[family] != family_solves:
                    raise ValueError(f"adjoint-solve count changed for {family}")
                candidates[family] = family_candidates
                solves[family] = family_solves
    feasibility = {
        "PB-AMD": "Exact throughout",
        "PGD grid": "Exact throughout",
        "Penalty grid": "Repair required",
    }
    labels = {
        "PB-AMD": "PB-AMD",
        "PGD grid": "PGD grid",
        "Penalty grid": "Penalty grid + repair",
    }
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcrrr}",
        "\\toprule",
        "Pipeline & Constraint treatment & Candidates & $K$-step adjoint solves & Relative time \\\\",
        "\\midrule",
    ]
    for family in RUNTIME_FAMILIES:
        median_by_task = []
        for task in TASK_ORDER:
            task_rows = [row for row in rows if row["task"] == task]
            by_repeat = {
                int(row["repeat"]): float(row["pipeline_time_sec"])
                for row in task_rows if row["family"] == family
            }
            pb_by_repeat = {
                int(row["repeat"]): float(row["pipeline_time_sec"])
                for row in task_rows if row["family"] == "PB-AMD"
            }
            median_by_task.append(statistics.median(by_repeat[r] / pb_by_repeat[r] for r in repeats))
        if family == "PB-AMD":
            time_text = "1.00"
        else:
            time_text = f"{min(median_by_task):.2f}--{max(median_by_task):.2f}"
        lines.append(
            f"{labels[family]} & {feasibility[family]} & {candidates[family]} & "
            f"{solves[family]} & {time_text} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Validated schedule-selection workload at $\\rho_{\\max}=0.5$.  "
        "Relative times are taskwise medians of three serial, single-threaded repeats; "
        "shared construction and final private evaluation are excluded.}",
        "\\label{tab:cost-summary}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_direct_numerical_checks(
    raw: list[dict[str, str]],
    lq: dict,
    path: Path,
) -> None:
    pb_rows = [row for row in raw if row["method_family"] == "PB-AMD"]
    if len(pb_rows) != 90:
        raise ValueError(f"expected 90 PB-AMD task-seed-budget rows, found {len(pb_rows)}")
    max_kkt = max(float(row["saa_kkt_residual"]) for row in pb_rows)
    max_feasibility = max(
        max(
            abs(float(row["rho_violation"])),
            abs(float(row["max_prox_feasibility_error"])),
        )
        for row in pb_rows
    )
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccc@{}}",
        "\\toprule",
        "Adjoint stationarity & Exact feasibility & Closed-form consistency \\\\ ",
        "\\midrule",
        f"${scientific(max_kkt)}$ & ${scientific(max_feasibility)}$ & "
        f"${scientific(float(lq['relative_objective_gap']))}$ \\\\ ",
        "Maximum fixed-path KKT residual & Maximum budget/prox error & Relative objective gap \\\\ ",
        "\\bottomrule",
        "\\end{tabular*}",
        "\\caption{Direct numerical checks of the adjoint and optimization calculations. "
        "The first two columns refer to all $90$ PB-AMD task--seed--budget "
        "solutions, and the third to the analytic linear--quadratic water-filling solution.  The KKT "
        "residual includes the interior gradient-level and one-sided floor conditions.}",
        "\\label{tab:direct-numerical-checks}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_macros(
    raw: list[dict[str, str]],
    published: list[dict[str, str]],
    proxy_summary: list[dict[str, str]],
    path: Path,
) -> None:
    if len(proxy_summary) != 1:
        raise ValueError("the proxy-transfer summary must contain one row")
    raw_penalty = [
        row for row in raw if row["method_family"].startswith("Penalty beta=")
    ]
    if len(raw_penalty) != 540:
        raise ValueError("the concise penalty audit requires 540 raw candidates")
    charge_ratios = [
        float(row["rho_spent"]) / float(row["rho"]) for row in raw_penalty
    ]
    exact_raw = sum(
        abs(float(row["rho_spent"]) - float(row["rho"]))
        <= 1e-10 * max(1.0, float(row["rho"]))
        for row in raw_penalty
    )
    selected_betas = {
        float(row["beta"])
        for row in raw
        if row["method_family"] == "Penalty repaired-tuned"
    }
    pb_index = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in raw
        if row["method_family"] == "PB-AMD"
    }
    repaired_index = {
        (row["task"], int(row["seed"]), float(row["rho"])): row
        for row in raw
        if row["method_family"] == "Penalty repaired-tuned"
    }
    terminal_differences = []
    for task in TASK_ORDER:
        for rho in (0.25, 0.5, 1.0):
            keys = [
                key for key in pb_index
                if key[0] == task and key[2] == rho and key in repaired_index
            ]
            if len(keys) != 10:
                raise ValueError(f"incomplete repaired-penalty comparison for {(task, rho)}")
            pb_mean = statistics.fmean(float(pb_index[key]["terminal_mean"]) for key in keys)
            penalty_mean = statistics.fmean(
                float(repaired_index[key]["terminal_mean"]) for key in keys
            )
            terminal_differences.append(100.0 * abs(penalty_mean - pb_mean) / abs(pb_mean))

    published_index = {
        (row["task"], int(row["seed"]), float(row["rho"])): row for row in published
    }
    published_gains = []
    published_ratios = []
    for rho in (0.25, 0.5, 1.0):
        keys = [
            key for key in published_index
            if key[0] == "fashion-lr_restart" and key[2] == rho and key in pb_index
        ]
        if len(keys) != 10:
            raise ValueError(f"incomplete published restart comparison at rho={rho}")
        pb_mean = statistics.fmean(float(pb_index[key]["terminal_mean"]) for key in keys)
        comparator_mean = statistics.fmean(
            float(published_index[key]["terminal_mean"]) for key in keys
        )
        published_gains.append(100.0 * (comparator_mean - pb_mean) / comparator_mean)
        published_ratios.extend(float(published_index[key]["selected_ratio"]) for key in keys)

    proxy = proxy_summary[0]
    lines = [
        "% Generated by code/export_concise_numerical_latex.py",
        f"\\newcommand{{\\PenaltyRawCandidateCount}}{{{len(raw_penalty)}}}",
        f"\\newcommand{{\\PenaltyExactRawCandidateCount}}{{{exact_raw}}}",
        f"\\newcommand{{\\PenaltyChargeRatioMinimum}}{{{min(charge_ratios):.4f}}}",
        f"\\newcommand{{\\PenaltyChargeRatioMedian}}{{{statistics.median(charge_ratios):.2f}}}",
        f"\\newcommand{{\\PenaltyChargeRatioMaximum}}{{{max(charge_ratios):.2f}}}",
        f"\\newcommand{{\\PenaltySelectedBetaCount}}{{{len(selected_betas)}}}",
        f"\\newcommand{{\\PenaltyMaximumTerminalDifference}}{{{max(terminal_differences):.3f}}}",
        f"\\newcommand{{\\PublishedRestartRatio}}{{{statistics.fmean(published_ratios):.0f}}}",
        f"\\newcommand{{\\PublishedRestartAdditionalReduction}}{{{statistics.fmean(published_gains):.2f}}}",
        f"\\newcommand{{\\ProxyLateEarly}}{{{float(proxy['late_over_early_mean']):.2f}}}",
        f"\\newcommand{{\\ProxyTerminalReduction}}{{{float(proxy['terminal_reduction_percent']):.2f}}}",
        f"\\newcommand{{\\ProxyTerminalReductionLow}}{{{float(proxy['terminal_reduction_ci95_low']):.2f}}}",
        f"\\newcommand{{\\ProxyTerminalReductionHigh}}{{{float(proxy['terminal_reduction_ci95_high']):.2f}}}",
        f"\\newcommand{{\\ProxyAccuracyGain}}{{{float(proxy['accuracy_difference_pp']):.3f}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--published", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--lq", required=True)
    parser.add_argument("--out-dir", default="output/pdf")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_csv(Path(args.raw))
    published = read_csv(Path(args.published))
    lq = read_json(Path(args.lq))
    write_utility_plot_data(raw, published, out_dir)
    write_penalty_plot_data(raw, out_dir)
    write_cost_summary(
        read_csv(Path(args.runtime)),
        out_dir / "numerical_cost_summary_table.tex",
    )
    write_direct_numerical_checks(
        raw,
        lq,
        out_dir / "numerical_direct_checks_table.tex",
    )
    for name in (
        "numerical_utility_tabular.dat",
        "numerical_utility_fashion_constant.dat",
        "numerical_utility_fashion_restart.dat",
        "numerical_penalty_tabular.dat",
        "numerical_penalty_fashion_constant.dat",
        "numerical_penalty_fashion_restart.dat",
    ):
        print(out_dir / name)
    print(out_dir / "numerical_cost_summary_table.tex")
    print(out_dir / "numerical_direct_checks_table.tex")


if __name__ == "__main__":
    main()
