from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dp_sgd_schedule_compare as tabular
import fashion_mnist_mlp_schedule_compare as fashion
from linear_quadratic_validation import run_validation


def relative_error(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-15)


def tangent_direction(rng: np.random.Generator, size: int) -> np.ndarray:
    direction = rng.normal(size=size)
    direction -= direction.mean()
    return direction / np.linalg.norm(direction)


def tabular_diagnostics() -> dict[str, object]:
    problem = tabular.make_problem(17)
    rng = np.random.default_rng(404)
    xis = rng.normal(size=(problem.T, 3, problem.d))
    q = np.full(problem.T, problem.rho_max / problem.T)
    objective, gradient = tabular.objective_and_grad(problem, q, xis)
    step = 1e-6
    directions = [tangent_direction(rng, problem.T) for _ in range(5)]
    adjoints = [float(np.dot(gradient, direction)) for direction in directions]
    finite_differences = [
        (
            tabular.objective_only(problem, q + step * direction, xis)
            - tabular.objective_only(problem, q - step * direction, xis)
        )
        / (2.0 * step)
        for direction in directions
    ]
    errors = [
        relative_error(adjoint, finite_difference)
        for adjoint, finite_difference in zip(adjoints, finite_differences)
    ]

    batch = problem.batches[0]
    x = problem.x_design[batch].copy()
    y = problem.y_design[batch].copy()
    theta = problem.theta_design_ref[0][None, :]
    first = tabular.batch_grad(theta, x, y, problem.l2, problem.clip)[0]
    replacement = next(index for index in range(len(problem.x_design)) if index not in set(batch))
    x[0] = problem.x_design[replacement]
    y[0] = problem.y_design[replacement]
    second = tabular.batch_grad(theta, x, y, problem.l2, problem.clip)[0]
    calibrated = 2.0 * problem.clip / len(batch)
    return {
        "objective": objective,
        "directional_derivative_adjoint": adjoints[0],
        "directional_derivative_finite_difference": finite_differences[0],
        "directional_derivative_relative_error": errors[0],
        "directional_derivative_checks": len(errors),
        "directional_derivative_relative_errors": errors,
        "directional_derivative_max_relative_error": max(errors),
        "observed_replace_one_change": float(np.linalg.norm(first - second)),
        "calibrated_replace_one_bound": calibrated,
        "observed_over_bound": float(np.linalg.norm(first - second)) / calibrated,
        "sensitivity_check_clip": problem.clip,
        "sensitivity_check_batch_size": len(batch),
    }


def fashion_diagnostics(data_dir: Path) -> dict[str, object]:
    problem = fashion.make_problem(
        data_dir,
        seed=321,
        n_train=8000,
        n_test=3000,
        horizon=45,
        batch_size=192,
        hidden=32,
        scenario="lr_restart",
    )
    rng = np.random.default_rng(505)
    theta0 = problem.theta_design_ref[0]
    xis = rng.normal(size=(problem.T, 2, problem.shape.p))
    q = np.full(problem.T, problem.rho_max / problem.T)
    objective, gradient = fashion.objective_and_grad(problem, q, xis, theta0)
    step = 1e-5
    directions = [tangent_direction(rng, problem.T) for _ in range(3)]
    adjoints = [float(np.dot(gradient, direction)) for direction in directions]
    finite_differences = [
        (
            fashion.objective_only(problem, q + step * direction, xis, theta0)
            - fashion.objective_only(problem, q - step * direction, xis, theta0)
        )
        / (2.0 * step)
        for direction in directions
    ]
    schedule_errors = [
        relative_error(adjoint, finite_difference)
        for adjoint, finite_difference in zip(adjoints, finite_differences)
    ]

    batch = problem.batches[0]
    x = problem.x_design[batch].copy()
    y = problem.y_design[batch].copy()
    theta = theta0[None, :]
    cotangent = rng.normal(size=theta.shape)
    vjp = fashion.mlp_grad_vjp(
        theta,
        cotangent,
        x,
        y,
        problem.shape,
        problem.l2,
        problem.clip,
        problem.active_masks[0],
    )
    vjp_step = 1e-6
    parameter_directions = []
    for _ in range(3):
        direction = rng.normal(size=theta.shape)
        parameter_directions.append(direction / np.linalg.norm(direction))
    vjp_analytics = []
    vjp_finite_differences = []
    for parameter_direction in parameter_directions:
        plus = fashion.mlp_grad(
            theta + vjp_step * parameter_direction,
            x,
            y,
            problem.shape,
            problem.l2,
            problem.clip,
            problem.active_masks[0],
        )
        minus = fashion.mlp_grad(
            theta - vjp_step * parameter_direction,
            x,
            y,
            problem.shape,
            problem.l2,
            problem.clip,
            problem.active_masks[0],
        )
        vjp_finite_differences.append(
            float(np.sum(cotangent * (plus - minus)) / (2.0 * vjp_step))
        )
        vjp_analytics.append(float(np.sum(vjp * parameter_direction)))
    vjp_errors = [
        relative_error(analytic, finite_difference)
        for analytic, finite_difference in zip(vjp_analytics, vjp_finite_differences)
    ]

    first = fashion.mlp_grad(
        theta, x, y, problem.shape, problem.l2, problem.clip, problem.active_masks[0]
    )[0]
    replacement = next(index for index in range(len(problem.x_design)) if index not in set(batch))
    x[0] = problem.x_design[replacement]
    y[0] = problem.y_design[replacement]
    second = fashion.mlp_grad(
        theta, x, y, problem.shape, problem.l2, problem.clip, problem.active_masks[0]
    )[0]
    calibrated = 2.0 * problem.clip / len(batch)
    return {
        "objective": objective,
        "schedule_directional_derivative_adjoint": adjoints[0],
        "schedule_directional_derivative_finite_difference": finite_differences[0],
        "schedule_directional_derivative_relative_error": schedule_errors[0],
        "schedule_directional_derivative_checks": len(schedule_errors),
        "schedule_directional_derivative_relative_errors": schedule_errors,
        "schedule_directional_derivative_max_relative_error": max(schedule_errors),
        "gradient_vjp_analytic": vjp_analytics[0],
        "gradient_vjp_finite_difference": vjp_finite_differences[0],
        "gradient_vjp_relative_error": vjp_errors[0],
        "gradient_vjp_checks": len(vjp_errors),
        "gradient_vjp_relative_errors": vjp_errors,
        "gradient_vjp_max_relative_error": max(vjp_errors),
        "observed_replace_one_change": float(np.linalg.norm(first - second)),
        "calibrated_replace_one_bound": calibrated,
        "observed_over_bound": float(np.linalg.norm(first - second)) / calibrated,
        "sensitivity_check_clip": problem.clip,
        "sensitivity_check_batch_size": len(batch),
        "parameters": problem.shape.p,
        "horizon": problem.T,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/submission_v4_validation_diagnostics.json")
    parser.add_argument("--data-dir", default="data/fashion_mnist")
    args = parser.parse_args()
    _, lq = run_validation()
    report = {
        "tabular": tabular_diagnostics(),
        "fashion_mnist": fashion_diagnostics(Path(args.data_dir)),
        "linear_quadratic": lq,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
