from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_utils import (
    floor_aware_entropy_step,
    mirror_residual,
    variational_gap,
)


def make_coefficients(horizon: int) -> np.ndarray:
    t = np.arange(horizon, dtype=float)
    transition = 0.91 + 0.055 * np.sin(2.0 * np.pi * (t + 1.0) / horizon)
    injection = 0.035 + 0.02 * (t / max(horizon - 1, 1)) ** 2
    future_gain = np.ones(horizon)
    running = 1.0
    for index in range(horizon - 1, -1, -1):
        future_gain[index] = running
        running *= transition[index]
    coefficients = 0.5 * (injection * future_gain) ** 2
    return coefficients


def objective(coefficients: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(coefficients / q))


def gradient(coefficients: np.ndarray, q: np.ndarray) -> np.ndarray:
    return -coefficients / (q * q)


def relative_smoothness_step_limit(coefficients: np.ndarray, q_min: float) -> float:
    """Return 1/L for the explicit relative-smoothness bound in the paper."""
    return float(q_min * q_min / (2.0 * np.max(coefficients)))


def exact_water_filling(coefficients: np.ndarray, rho: float, q_min: float) -> np.ndarray:
    def allocation(log_lambda: float) -> np.ndarray:
        return np.maximum(q_min, np.sqrt(coefficients / np.exp(log_lambda)))

    lower, upper = -80.0, 80.0
    for _ in range(240):
        midpoint = 0.5 * (lower + upper)
        if float(allocation(midpoint).sum()) > rho:
            lower = midpoint
        else:
            upper = midpoint
    q = allocation(0.5 * (lower + upper))
    q[int(np.argmax(q))] += rho - float(q.sum())
    return q


def run_validation(
    horizon: int = 80,
    rho: float = 1.0,
    q_min: float = 1e-3,
    alpha: float = 3e-4,
    iterations: int = 200000,
) -> tuple[list[dict], dict]:
    coefficients = make_coefficients(horizon)
    theory_step_limit = relative_smoothness_step_limit(coefficients, q_min)
    if alpha > theory_step_limit:
        raise ValueError(
            f"alpha={alpha:g} exceeds the theorem limit {theory_step_limit:g}"
        )
    exact = exact_water_filling(coefficients, rho, q_min)

    # Start on a boundary face with the largest coefficient at the floor. This
    # specifically checks that the direct-entropy prox can reactivate it.
    q = np.full(horizon, q_min)
    target_index = int(np.argmax(coefficients))
    donor_index = (target_index + horizon // 2) % horizon
    q[donor_index] += rho - horizon * q_min

    rows = []
    max_budget_error = 0.0
    for iteration in range(iterations + 1):
        value = objective(coefficients, q)
        grad = gradient(coefficients, q)
        max_budget_error = max(max_budget_error, abs(float(q.sum()) - rho))
        if iteration % 50 == 0 or iteration == iterations:
            rows.append({
                "iteration": iteration,
                "objective": value,
                "relative_objective_gap": (value - objective(coefficients, exact)) / objective(coefficients, exact),
                "variational_gap": variational_gap(q, grad, rho, q_min),
                "mirror_residual": mirror_residual(q, grad, alpha, rho, q_min),
                "schedule_l1_error": float(np.sum(np.abs(q - exact))),
                "target_coordinate": float(q[target_index]),
                "budget_error": abs(float(q.sum()) - rho),
            })
        if iteration < iterations:
            q, _, prox_error = floor_aware_entropy_step(q, grad, alpha, rho, q_min)
            max_budget_error = max(max_budget_error, prox_error)

    report = {
        "horizon": horizon,
        "rho": rho,
        "q_min": q_min,
        "alpha": alpha,
        "relative_smoothness_step_limit": theory_step_limit,
        "alpha_over_step_limit": alpha / theory_step_limit,
        "iterations": iterations,
        "exact_objective": objective(coefficients, exact),
        "final_objective": objective(coefficients, q),
        "relative_objective_gap": (objective(coefficients, q) - objective(coefficients, exact)) / objective(coefficients, exact),
        "schedule_l1_error": float(np.sum(np.abs(q - exact))),
        "floor_coordinates_exact": int(np.sum(exact <= q_min + 1e-12)),
        "reactivated_target_coordinate": float(q[target_index]),
        "max_budget_error": max_budget_error,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/submission_v4_lq_validation")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--q-min", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=3e-4)
    parser.add_argument("--iterations", type=int, default=200000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, report = run_validation(args.horizon, args.rho, args.q_min, args.alpha, args.iterations)
    with (out_dir / "convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
