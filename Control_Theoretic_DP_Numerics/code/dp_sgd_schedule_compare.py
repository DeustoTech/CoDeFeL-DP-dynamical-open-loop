from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment_utils import (
    active_kkt_residual,
    floor_aware_entropy_step,
    floor_diagnostics,
    inequality_budget_kkt_residual,
    mirror_residual,
    privacy_epsilon,
    project_lower_bounded_simplex,
    projected_gradient_residual,
    variational_gap,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    nonnegative = x >= 0.0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


@dataclass
class Problem:
    x_design: np.ndarray
    y_design: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    batches: list[np.ndarray]
    theta_design_ref: np.ndarray
    theta_ref: np.ndarray
    eta: np.ndarray
    sensitivity: float
    clip: float
    l2: float
    rho_max: float
    q_min: float

    @property
    def T(self) -> int:
        return len(self.batches)

    @property
    def d(self) -> int:
        return self.x_train.shape[1]


def make_problem(
    seed: int = 17,
    rho_max: float = 0.5,
    q_min: float = 1.0e-5,
    clip: float = 4.0,
) -> Problem:
    n_design, n_train, n_test, d = 4500, 4500, 1500, 64
    rho_corr = 0.65
    idx = np.arange(d)
    cov = rho_corr ** np.abs(idx[:, None] - idx[None, :])
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
    w_true = np.zeros(d)
    informative = np.array([0, 1, 2, 3, 4, 5, 9, 13, 21, 34, 45, 55])
    w_true[informative] = np.array([1.7, -1.4, 1.1, -1.0, 0.9, -0.8, 0.7, -0.65, 0.6, -0.55, 0.5, -0.45])

    def sample_population(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d)) @ chol.T
        logit = (
            x @ w_true
            + 0.85 * x[:, 0] * x[:, 1]
            - 0.65 * x[:, 2] * x[:, 3]
            + 0.45 * np.sin(x[:, 4])
            + 0.35 * (x[:, 5] > 0.0)
        ) / 2.43
        p = sigmoid(1.35 * logit)
        y = np.where(rng.uniform(size=n) < p, 1.0, -1.0)
        y[rng.uniform(size=n) < 0.05] *= -1.0
        return x, y

    # The schedule is designed on an independent public proxy sample.  The
    # private training and test samples are never read by the schedule solver.
    design_rng = np.random.default_rng(seed + 10_000)
    private_rng = np.random.default_rng(seed)
    batch_rng = np.random.default_rng(seed + 20_000)
    x_design, y_design = sample_population(design_rng, n_design)
    x_private, y_private = sample_population(private_rng, n_train + n_test)
    perm = private_rng.permutation(n_train + n_test)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    x_train, y_train = x_private[train_idx], y_private[train_idx]
    x_test, y_test = x_private[test_idx], y_private[test_idx]

    # The synthetic population already has zero mean and unit marginal
    # variances, so no private-data-dependent normalization is required.

    T = 60
    batch_size = 256
    batches = [batch_rng.choice(n_train, size=batch_size, replace=False) for _ in range(T)]
    eta = np.full(T, 0.18)
    # Under replace-one adjacency, one contribution is replaced.  The
    # difference of the old and new clipped vectors has norm at most 2*clip,
    # so the averaged query has sensitivity 2*clip/batch_size.
    sensitivity = 2.0 * clip / batch_size
    l2 = 2.0e-3
    if T * q_min >= rho_max:
        raise ValueError("q_min must satisfy T * q_min < rho_max")

    theta_design_ref = compute_reference(x_design, y_design, batches, eta, l2, clip)
    theta_ref = compute_reference(x_train, y_train, batches, eta, l2, clip)
    return Problem(
        x_design,
        y_design,
        x_train,
        y_train,
        x_test,
        y_test,
        batches,
        theta_design_ref,
        theta_ref,
        eta,
        sensitivity,
        clip,
        l2,
        rho_max,
        q_min,
    )


def batch_grad(
    theta: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    l2: float,
    clip: float,
) -> np.ndarray:
    """Average smoothly clipped per-example logistic gradients.

    Regularization is deterministic and is therefore added after clipping.
    """
    yz = (theta @ xb.T) * yb[None, :]
    s = sigmoid(-yz)
    norm_sq = s * s * np.sum(xb * xb, axis=1)[None, :]
    scales = 1.0 / np.sqrt(1.0 + norm_sq / (clip * clip))
    weights = -s * yb[None, :] * scales
    return weights @ xb / xb.shape[0] + l2 * theta


def batch_grad_vjp(
    theta: np.ndarray,
    cotangent: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    l2: float,
    clip: float,
) -> np.ndarray:
    """Exact VJP of the smoothly clipped minibatch gradient."""
    yz = (theta @ xb.T) * yb[None, :]
    s = sigmoid(-yz)
    x_norm_sq = np.sum(xb * xb, axis=1)
    norm_sq = s * s * x_norm_sq[None, :]
    scales = 1.0 / np.sqrt(1.0 + norm_sq / (clip * clip))
    raw_coefficient = -s * yb[None, :]
    x_dot_cotangent = cotangent @ xb.T
    inner = raw_coefficient * x_dot_cotangent
    x_dot_raw_gradient = raw_coefficient * x_norm_sq[None, :]
    x_dot_direction = scales * (
        x_dot_cotangent
        - inner * x_dot_raw_gradient / (clip * clip + norm_sq)
    )
    curvature = s * (1.0 - s)
    vjp = (curvature * x_dot_direction) @ xb / xb.shape[0]
    return vjp + l2 * cotangent


def compute_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batches: list[np.ndarray],
    eta: np.ndarray,
    l2: float,
    clip: float,
) -> np.ndarray:
    T = len(batches)
    d = x_train.shape[1]
    theta = np.zeros((T + 1, d))
    for t, batch in enumerate(batches):
        xb, yb = x_train[batch], y_train[batch]
        grad = batch_grad(theta[t][None, :], xb, yb, l2, clip)[0]
        theta[t + 1] = theta[t] - eta[t] * grad
    return theta


def objective_and_grad(problem: Problem, q: np.ndarray, xis: np.ndarray) -> tuple[float, np.ndarray]:
    T, M, d = xis.shape
    theta = np.zeros((T + 1, M, d))
    for t, batch in enumerate(problem.batches):
        xb, yb = problem.x_design[batch], problem.y_design[batch]
        grad = batch_grad(theta[t], xb, yb, problem.l2, problem.clip)
        amp = problem.sensitivity * problem.eta[t] / math.sqrt(2.0 * q[t])
        theta[t + 1] = theta[t] - problem.eta[t] * grad - amp * xis[t]

    terminal_diff = theta[T] - problem.theta_design_ref[T][None, :]
    obj = float(np.mean(np.sum(terminal_diff * terminal_diff, axis=1)))
    p = 2.0 * terminal_diff
    h = np.zeros(T)

    for t in range(T - 1, -1, -1):
        h[t] = np.mean(problem.sensitivity * problem.eta[t] * (2.0 * q[t]) ** (-1.5) * np.sum(p * xis[t], axis=1))
        xb, yb = problem.x_design[problem.batches[t]], problem.y_design[problem.batches[t]]
        vjp = batch_grad_vjp(theta[t], p, xb, yb, problem.l2, problem.clip)
        p = p - problem.eta[t] * vjp
    return obj, h


def objective_only(problem: Problem, q: np.ndarray, xis: np.ndarray) -> float:
    _, M, d = xis.shape
    theta = np.zeros((M, d))
    for t, batch in enumerate(problem.batches):
        grad = batch_grad(
            theta,
            problem.x_design[batch],
            problem.y_design[batch],
            problem.l2,
            problem.clip,
        )
        amp = problem.sensitivity * problem.eta[t] / math.sqrt(2.0 * q[t])
        theta = theta - problem.eta[t] * grad - amp * xis[t]
    terminal_diff = theta - problem.theta_design_ref[problem.T][None, :]
    return float(np.mean(np.sum(terminal_diff * terminal_diff, axis=1)))


def simulate_many(problem: Problem, schedules: list[np.ndarray], xis: np.ndarray) -> list[dict[str, float]]:
    """Evaluate schedules together with identical Gaussian paths."""
    T, M, d = xis.shape
    q_matrix = np.asarray(schedules, dtype=float)
    if q_matrix.ndim != 2 or q_matrix.shape[1] != T:
        raise ValueError("every schedule must have one coordinate per time step")
    n_methods = q_matrix.shape[0]
    theta = np.zeros((n_methods, M, d))
    path_dev = np.zeros((n_methods, M))
    for t, batch in enumerate(problem.batches):
        xb, yb = problem.x_train[batch], problem.y_train[batch]
        flat_theta = theta.reshape(n_methods * M, d)
        grad = batch_grad(flat_theta, xb, yb, problem.l2, problem.clip).reshape(n_methods, M, d)
        amp = problem.sensitivity * problem.eta[t] / np.sqrt(2.0 * q_matrix[:, t])
        theta = theta - problem.eta[t] * grad - amp[:, None, None] * xis[t][None, :, :]
        diff_t = theta - problem.theta_ref[t + 1][None, None, :]
        path_dev += np.sum(diff_t * diff_t, axis=2)

    terminal = np.sum((theta - problem.theta_ref[T][None, None, :]) ** 2, axis=2)
    flat_theta = theta.reshape(n_methods * M, d)
    train_loss, train_acc = logistic_loss_acc(problem.x_train, problem.y_train, flat_theta, problem.l2)
    test_loss, test_acc = logistic_loss_acc(problem.x_test, problem.y_test, flat_theta, problem.l2)
    train_loss = train_loss.reshape(n_methods, M)
    train_acc = train_acc.reshape(n_methods, M)
    test_loss = test_loss.reshape(n_methods, M)
    test_acc = test_acc.reshape(n_methods, M)
    return [
        {
            "terminal_mean": float(np.mean(terminal[index])),
            "terminal_std": float(np.std(terminal[index])),
            "path_mean": float(np.mean(path_dev[index])),
            "path_std": float(np.std(path_dev[index])),
            "train_loss": float(np.mean(train_loss[index])),
            "train_acc": float(np.mean(train_acc[index])),
            "test_loss": float(np.mean(test_loss[index])),
            "test_acc": float(np.mean(test_acc[index])),
            "test_acc_std": float(np.std(test_acc[index])),
        }
        for index in range(n_methods)
    ]


def simulate(problem: Problem, q: np.ndarray, xis: np.ndarray) -> dict[str, float]:
    return simulate_many(problem, [q], xis)[0]


def logistic_loss_acc(x: np.ndarray, y: np.ndarray, theta: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    logits = theta @ x.T
    yz = logits * y[None, :]
    loss = np.mean(np.logaddexp(0.0, -yz), axis=1) + 0.5 * l2 * np.sum(theta * theta, axis=1)
    acc = np.mean(np.where(logits >= 0.0, 1.0, -1.0) == y[None, :], axis=1)
    return loss, acc


def kkt_residual(problem: Problem, q: np.ndarray, xis: np.ndarray) -> float:
    """KKT residual for the lower-bounded exact-budget simplex."""
    _, h = objective_and_grad(problem, q, xis)
    feasibility = max(
        abs(float(np.sum(q) - problem.rho_max)),
        max(float(problem.q_min - np.min(q)), 0.0),
    )
    return max(active_kkt_residual(q, h, problem.q_min), feasibility)


def run_pb_amd(
    problem: Problem,
    xis_opt: np.ndarray,
    alpha: float,
    K: int,
    record_trace: bool = True,
    kkt_tolerance: float | None = None,
    max_iterations: int | None = None,
) -> dict:
    if kkt_tolerance is not None:
        if kkt_tolerance <= 0.0:
            raise ValueError("kkt_tolerance must be positive")
        if max_iterations is None:
            max_iterations = 10 * K
        if K < 1 or max_iterations < K:
            raise ValueError("max_iterations must be at least K >= 1")
        return _run_pb_amd_to_kkt(
            problem,
            xis_opt,
            alpha,
            K,
            max_iterations,
            kkt_tolerance,
            record_trace,
        )

    T = problem.T
    q = np.full(T, problem.rho_max / T)
    hist = []
    trace = []
    best_obj = float("inf")
    best_q = None
    max_feasibility_error = 0.0
    start = time.perf_counter()
    for k in range(K):
        obj, h = objective_and_grad(problem, q, xis_opt)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
        if record_trace and (k % 5 == 0 or k == K - 1):
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": mirror_residual(
                    q,
                    h,
                    alpha,
                    problem.rho_max,
                    problem.q_min,
                ),
                "kkt_residual": active_kkt_residual(q, h, problem.q_min),
            })
        q, _, feasibility_error = floor_aware_entropy_step(
            q,
            h,
            alpha,
            problem.rho_max,
            problem.q_min,
        )
        max_feasibility_error = max(max_feasibility_error, feasibility_error)
        if record_trace and (k % 5 == 0 or k == K - 1):
            hist.append((k, obj, float(np.sum(q)), feasibility_error))
    obj = objective_only(problem, q, xis_opt)
    if obj < best_obj:
        best_obj = obj
        best_q = q.copy()
    elapsed = time.perf_counter() - start
    assert best_q is not None
    return {
        "method": f"PB-AMD alpha={alpha:g}",
        "family": "PB-AMD",
        "alpha": alpha,
        "beta": "",
        "lr": "",
        "q": best_q,
        "design_objective": best_obj,
        "history": hist,
        "trace": trace,
        "time_sec": elapsed,
        "max_prox_feasibility_error": max_feasibility_error,
        "iterations": K,
    }


def _run_pb_amd_to_kkt(
    problem: Problem,
    xis_opt: np.ndarray,
    alpha: float,
    minimum_iterations: int,
    max_iterations: int,
    kkt_tolerance: float,
    record_trace: bool,
) -> dict:
    """Run at least ``minimum_iterations`` updates and certify the returned iterate."""
    q = np.full(problem.T, problem.rho_max / problem.T)
    history = []
    trace = []
    best_obj = float("inf")
    best_q = None
    best_kkt = float("inf")
    max_feasibility_error = 0.0
    converged = False
    iterations = 0
    start = time.perf_counter()

    for k in range(max_iterations + 1):
        obj, gradient = objective_and_grad(problem, q, xis_opt)
        candidate_kkt = active_kkt_residual(q, gradient, problem.q_min)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
            best_kkt = candidate_kkt

        should_record = (
            record_trace
            and (
                k % 5 == 0
                or k == minimum_iterations
                or k == max_iterations
                or (k >= minimum_iterations and best_kkt <= kkt_tolerance)
            )
        )
        if should_record:
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": mirror_residual(
                    q,
                    gradient,
                    alpha,
                    problem.rho_max,
                    problem.q_min,
                ),
                "kkt_residual": candidate_kkt,
            })

        if k >= minimum_iterations and best_kkt <= kkt_tolerance:
            converged = True
            iterations = k
            break
        if k == max_iterations:
            iterations = k
            break

        q, _, feasibility_error = floor_aware_entropy_step(
            q,
            gradient,
            alpha,
            problem.rho_max,
            problem.q_min,
        )
        max_feasibility_error = max(max_feasibility_error, feasibility_error)
        if should_record:
            history.append((k, obj, float(q.sum()), feasibility_error))

    elapsed = time.perf_counter() - start
    assert best_q is not None
    return {
        "method": f"PB-AMD alpha={alpha:g}",
        "family": "PB-AMD",
        "alpha": alpha,
        "beta": "",
        "lr": "",
        "q": best_q,
        "design_objective": best_obj,
        "history": history,
        "trace": trace,
        "time_sec": elapsed,
        "max_prox_feasibility_error": max_feasibility_error,
        "iterations": iterations,
        "gradient_evaluations": iterations + 1,
        "kkt_tolerance": kkt_tolerance,
        "certified_kkt_residual": best_kkt,
        "converged": converged,
        "stopping_reason": "kkt_tolerance" if converged else "maximum_iterations",
    }


def run_projected_gradient(
    problem: Problem,
    xis_opt: np.ndarray,
    alpha: float,
    K: int,
    record_trace: bool = True,
) -> dict:
    """Euclidean projected gradient on the exact lower-bounded budget set."""
    q = np.full(problem.T, problem.rho_max / problem.T)
    history = []
    trace = []
    best_obj = float("inf")
    best_q = None
    start = time.perf_counter()
    for k in range(K):
        obj, gradient = objective_and_grad(problem, q, xis_opt)
        if obj < best_obj:
            best_obj = obj
            best_q = q.copy()
        if record_trace and (k % 5 == 0 or k == K - 1):
            residual = projected_gradient_residual(
                q,
                gradient,
                alpha,
                problem.rho_max,
                problem.q_min,
            )
            history.append((
                k,
                obj,
                float(q.sum()),
                residual,
            ))
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": obj,
                "rho_spent": float(q.sum()),
                "rho_violation": max(float(q.sum() - problem.rho_max), 0.0),
                "rho_slack": max(float(problem.rho_max - q.sum()), 0.0),
                "optimizer_residual": residual,
                "kkt_residual": active_kkt_residual(q, gradient, problem.q_min),
            })
        q = project_lower_bounded_simplex(
            q - alpha * gradient,
            problem.rho_max,
            problem.q_min,
        )
    obj = objective_only(problem, q, xis_opt)
    if obj < best_obj:
        best_obj = obj
        best_q = q.copy()
    elapsed = time.perf_counter() - start
    assert best_q is not None
    return {
        "method": f"PGD alpha={alpha:g}",
        "family": "PGD",
        "alpha": alpha,
        "beta": "",
        "lr": "",
        "q": best_q,
        "design_objective": best_obj,
        "history": history,
        "trace": trace,
        "time_sec": elapsed,
        "max_prox_feasibility_error": abs(float(best_q.sum()) - problem.rho_max),
        "iterations": K,
    }


def run_penalty(
    problem: Problem,
    xis_opt: np.ndarray,
    beta: float,
    lr: float,
    K: int,
    initial_q: np.ndarray | None = None,
    initialization: str = "uniform",
    compute_diagnostics: bool = True,
    record_trace: bool = True,
) -> dict:
    T = problem.T
    q0 = (
        np.full(T, problem.rho_max / T)
        if initial_q is None
        else np.asarray(initial_q, dtype=float).copy()
    )
    if q0.shape != (T,) or np.any(q0 <= 0.0):
        raise ValueError("the penalty initialization must be a positive schedule")
    v = -np.log(2.0 * q0)
    m = np.zeros(T)
    s = np.zeros(T)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    hist = []
    trace = []
    best_penalized = float("inf")
    best_q = None
    best_obj = float("inf")
    start = time.perf_counter()
    for k in range(K):
        q = 0.5 * np.exp(-v)
        q = np.clip(q, 1e-8, 2.0 * problem.rho_max)
        obj, h = objective_and_grad(problem, q, xis_opt)
        phi = float(np.sum(q) - problem.rho_max)
        grad_v = -q * h - beta * max(phi, 0.0) * q
        m = beta1 * m + (1.0 - beta1) * grad_v
        s = beta2 * s + (1.0 - beta2) * grad_v * grad_v
        m_hat = m / (1.0 - beta1 ** (k + 1))
        s_hat = s / (1.0 - beta2 ** (k + 1))
        v = v - lr * m_hat / (np.sqrt(s_hat) + eps)
        v = np.clip(v, -math.log(2.0 * 2.0 * problem.rho_max), -math.log(2.0 * 1e-8))
        penalized = obj + 0.5 * beta * max(phi, 0.0) ** 2
        if penalized < best_penalized:
            best_penalized = penalized
            best_q = q.copy()
            best_obj = obj
        if record_trace and (k % 5 == 0 or k == K - 1):
            hist.append((k, penalized, float(np.sum(q)), max(phi, 0.0)))
            trace.append({
                "iteration": k,
                "objective": obj,
                "penalized_objective": penalized,
                "rho_spent": float(q.sum()),
                "rho_violation": max(phi, 0.0),
                "rho_slack": max(-phi, 0.0),
                "optimizer_residual": float(np.linalg.norm(grad_v)),
                "kkt_residual": inequality_budget_kkt_residual(
                    q,
                    h,
                    problem.rho_max,
                ),
            })
    q = 0.5 * np.exp(-v)
    q = np.clip(q, 1e-8, 2.0 * problem.rho_max)
    obj = objective_only(problem, q, xis_opt)
    phi = float(np.sum(q) - problem.rho_max)
    penalized = obj + 0.5 * beta * max(phi, 0.0) ** 2
    if penalized < best_penalized:
        best_q = q.copy()
        best_obj = obj
    elapsed = time.perf_counter() - start
    assert best_q is not None
    q = best_q
    obj = best_obj
    phi = float(np.sum(q) - problem.rho_max)
    optimizer_grad_norm: float | str = ""
    if compute_diagnostics:
        _, h = objective_and_grad(problem, q, xis_opt)
        penalized_gradient = -q * h - beta * max(phi, 0.0) * q
        optimizer_grad_norm = float(np.linalg.norm(penalized_gradient))
    return {
        "method": f"Penalty beta={beta:g}, lr={lr:g}",
        "family": "Penalty",
        "alpha": "",
        "beta": beta,
        "lr": lr,
        "q": q,
        "design_objective": obj,
        "history": hist,
        "trace": trace,
        "time_sec": elapsed,
        "violation": max(phi, 0.0),
        "optimizer_grad_norm": optimizer_grad_norm,
        "iterations": K,
        "initialization": initialization,
    }


def summarize_schedule(q: np.ndarray, rho: float) -> dict[str, float]:
    T = len(q)
    avg = rho / T
    early_values = q[: T // 3]
    middle_values = q[T // 3 : 2 * T // 3]
    late_values = q[2 * T // 3 :]
    early = float(np.sum(early_values))
    middle = float(np.sum(middle_values))
    late = float(np.sum(late_values))
    normalized = q / np.sum(q)
    entropy = -float(np.sum(normalized * np.log(normalized + 1e-300))) / math.log(T)
    return {
        "rho_spent": float(np.sum(q)),
        "rho_violation": max(float(np.sum(q) - rho), 0.0),
        "rho_slack": max(float(rho - np.sum(q)), 0.0),
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
        "max_over_avg": float(np.max(q) / avg),
        "early_budget": early,
        "middle_budget": middle,
        "late_budget": late,
        "late_over_early": float(np.mean(late_values)) / max(float(np.mean(early_values)), 1e-12),
        "schedule_entropy": entropy,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def svg_line_plot(series: list[tuple[str, np.ndarray]], path: Path, title: str, ylabel: str) -> None:
    width, height = 920, 360
    ml, mr, mt, mb = 70, 22, 38, 48
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    dash_arrays = ["", "9 5", "3 3", "10 3 2 3", "1.5 3", "14 5", "5 2 1 2"]
    all_y = np.concatenate([y for _, y in series])
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    xmax = max(len(y) - 1 for _, y in series)

    def xy(i: int, val: float) -> tuple[float, float]:
        x = ml + (width - ml - mr) * i / max(xmax, 1)
        y = mt + (height - mt - mb) * (1.0 - (val - ymin) / (ymax - ymin))
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Helvetica" font-size="17" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#333"/>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Helvetica" font-size="13">{html.escape(ylabel)}</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = ymin + frac * (ymax - ymin)
        _, yy = xy(0, yv)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{width-mr}" y2="{yy:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-family="Helvetica" font-size="11">{yv:.3g}</text>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ix = int(round(frac * xmax))
        xx, _ = xy(ix, ymin)
        parts.append(f'<text x="{xx:.2f}" y="{height-18}" text-anchor="middle" font-family="Helvetica" font-size="11">{ix}</text>')
    legend_x = ml + 12
    legend_y = mt + 12
    for j, (name, yvals) in enumerate(series):
        pts = " ".join(f"{xy(i, float(v))[0]:.2f},{xy(i, float(v))[1]:.2f}" for i, v in enumerate(yvals))
        color = colors[j % len(colors)]
        dash = dash_arrays[j % len(dash_arrays)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        parts.append(f'<line x1="{legend_x}" y1="{legend_y+18*j}" x2="{legend_x+22}" y2="{legend_y+18*j}" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        parts.append(f'<text x="{legend_x+28}" y="{legend_y+18*j+4}" font-family="Helvetica" font-size="12">{html.escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_history_plot(histories: list[tuple[str, list[tuple[int, float, float, float]]]], path: Path, title: str) -> None:
    series = []
    for name, hist in histories:
        y = np.array([row[1] for row in hist], dtype=float)
        series.append((name, y))
    svg_line_plot(series, path, title, "optimization objective")


def make_report(out_dir: Path, summary_rows: list[dict], selected: list[dict], config: dict) -> None:
    schedule_svg = "schedules.svg"
    objective_svg = "objectives.svg"
    rows_html = []
    display_cols = [
        "method",
        "design_objective",
        "terminal_mean",
        "path_mean",
        "test_acc",
        "rho_spent",
        "rho_violation",
        "rho_slack",
        "late_over_early",
        "saa_kkt_residual",
        "time_sec",
    ]
    for r in summary_rows:
        cells = []
        for c in display_cols:
            val = r[c]
            if isinstance(val, float):
                cells.append(f"<td>{val:.5g}</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    table = "\n".join(rows_html)
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DP-SGD schedule optimization comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif; margin: 36px; color: #222; }}
h1, h2 {{ margin-bottom: 0.35rem; }}
table {{ border-collapse: collapse; font-size: 13px; margin-top: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f5f5f5; }}
.note {{ max-width: 980px; line-height: 1.45; }}
img {{ max-width: 960px; display: block; margin: 18px 0; border: 1px solid #eee; }}
code {{ background: #f5f5f5; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>DP-SGD Open-Loop Schedule Optimization</h1>
<p class="note">
The task is a high-dimensional correlated tabular binary classification problem with sparse signal,
nonlinear interactions, and 5% label noise. The model is L2-regularized logistic regression trained by
the same stochastic mini-batch sequence over {config["T"]} steps. Schedules are optimized for terminal
tracking of the non-private reference trajectory under the zCDP budget rho={config["rho_max"]}.
</p>
<h2>Main comparison</h2>
<table>
<thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in display_cols)}</tr></thead>
<tbody>
{table}
</tbody>
</table>
<h2>Privacy budget schedules</h2>
<img src="{schedule_svg}" alt="privacy budget schedules">
<h2>Optimization curves</h2>
<img src="{objective_svg}" alt="optimization curves">
<h2>Files</h2>
<p class="note">
Raw summary: <code>summary.csv</code>. Schedules: <code>schedules.csv</code>.
Optimization histories: <code>histories.csv</code>. Configuration: <code>config.json</code>.
</p>
</body>
</html>
"""
    (out_dir / "report.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/dp_sgd_schedule")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--K", type=int, default=180)
    parser.add_argument("--M-opt", type=int, default=256)
    parser.add_argument("--M-val", type=int, default=512)
    parser.add_argument("--M-eval", type=int, default=512)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--q-min", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--alphas", default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--pgd-alphas", default="3e-5,1e-4,3e-4,1e-3")
    parser.add_argument(
        "--penalty-grid",
        default="1:0.04,3:0.04,10:0.035,100:0.025,1000:0.015,10000:0.008",
        help="comma-separated beta:learning-rate pairs; use an empty string to skip penalties",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = make_problem(args.seed, rho_max=args.rho, q_min=args.q_min, clip=args.clip)
    opt_seed = args.seed + 101
    val_seed = args.seed + 151
    eval_seed = args.seed + 202
    xis_opt = np.random.default_rng(opt_seed).normal(size=(problem.T, args.M_opt, problem.d))
    xis_val = np.random.default_rng(val_seed).normal(size=(problem.T, args.M_val, problem.d))
    xis_eval = np.random.default_rng(eval_seed).normal(size=(problem.T, args.M_eval, problem.d))

    methods = []
    alphas = [float(value) for value in args.alphas.split(",") if value]
    pgd_alphas = [float(value) for value in args.pgd_alphas.split(",") if value]
    penalty_grid = []
    for item in args.penalty_grid.split(","):
        if item:
            beta, lr = item.split(":", maxsplit=1)
            penalty_grid.append((float(beta), float(lr)))
    # PB-AMD step sizes: enough to show robustness, with the best one selected in the table.
    for alpha in alphas:
        methods.append(run_pb_amd(problem, xis_opt, alpha=alpha, K=args.K))
    for alpha in pgd_alphas:
        methods.append(run_projected_gradient(problem, xis_opt, alpha=alpha, K=args.K))
    # Penalty method with a small grid of penalty parameters. Learning rates are scaled down for large beta.
    for beta, lr in penalty_grid:
        methods.append(run_penalty(problem, xis_opt, beta=beta, lr=lr, K=args.K))

    q_uniform = np.full(problem.T, problem.rho_max / problem.T)
    methods.append({
        "method": "Uniform feasible",
        "family": "Uniform",
        "alpha": "",
        "beta": "",
        "lr": "",
        "q": q_uniform,
        "design_objective": objective_and_grad(problem, q_uniform, xis_opt)[0],
        "history": [(0, objective_and_grad(problem, q_uniform, xis_opt)[0], float(np.sum(q_uniform)), 0.0)],
        "time_sec": 0.0,
    })

    for method in methods:
        method["validation_objective"] = objective_only(problem, method["q"], xis_val)

    # Select a compact set for plots: best PB-AMD, best feasible penalty, representative low- and high-beta penalties, uniform.
    summary_rows = []
    schedules = []
    histories = []
    evaluation_metrics = simulate_many(problem, [m["q"] for m in methods], xis_eval)
    reference_test_loss, reference_test_acc = logistic_loss_acc(
        problem.x_test,
        problem.y_test,
        problem.theta_ref[-1][None, :],
        problem.l2,
    )
    for m, eval_metrics in zip(methods, evaluation_metrics):
        q = m["q"]
        sched = summarize_schedule(q, problem.rho_max)
        exact_feasible = (
            abs(float(np.sum(q)) - problem.rho_max) <= 1e-10
            and float(np.min(q)) >= problem.q_min - 1e-12
        )
        report_stationarity = exact_feasible and m["family"] in {"PB-AMD", "PGD"}
        if report_stationarity:
            _, saa_gradient = objective_and_grad(problem, q, xis_opt)
            _, validation_gradient = objective_and_grad(problem, q, xis_val)
        floor_stats = floor_diagnostics(q, problem.q_min)
        row = {
            "method": m["method"],
            "family": m["family"],
            "alpha": m["alpha"],
            "beta": m["beta"],
            "lr": m["lr"],
            "initialization": m.get("initialization", ""),
            "design_objective": float(m["design_objective"]),
            "validation_objective": float(m["validation_objective"]),
            **eval_metrics,
            "reference_test_loss": float(reference_test_loss[0]),
            "reference_test_acc": float(reference_test_acc[0]),
            **sched,
            "delta": args.delta,
            "epsilon_target": privacy_epsilon(problem.rho_max, args.delta),
            "epsilon_actual": privacy_epsilon(sched["rho_spent"], args.delta),
            "saa_variational_gap": variational_gap(q, saa_gradient, problem.rho_max, problem.q_min) if report_stationarity else "",
            "saa_mirror_residual": mirror_residual(q, saa_gradient, float(m["alpha"]), problem.rho_max, problem.q_min) if m["family"] == "PB-AMD" else "",
            "saa_projected_residual": projected_gradient_residual(q, saa_gradient, float(m["alpha"]), problem.rho_max, problem.q_min) if m["family"] == "PGD" else "",
            "saa_kkt_residual": active_kkt_residual(q, saa_gradient, problem.q_min) if report_stationarity else "",
            "validation_variational_gap": variational_gap(q, validation_gradient, problem.rho_max, problem.q_min) if report_stationarity else "",
            **floor_stats,
            "optimizer_grad_norm": m.get("optimizer_grad_norm", ""),
            "iterations": m.get("iterations", ""),
            "max_prox_feasibility_error": m.get("max_prox_feasibility_error", ""),
            "time_sec": float(m["time_sec"]),
        }
        summary_rows.append(row)
        schedules.append((m["method"], q.copy()))
        histories.append((m["method"], m["history"]))

    write_csv(out_dir / "summary.csv", summary_rows)

    with (out_dir / "schedules.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [m["method"] for m in methods])
        for t in range(problem.T):
            writer.writerow([t] + [float(m["q"][t]) for m in methods])

    with (out_dir / "histories.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "iteration", "objective_or_penalized", "rho_spent", "violation"])
        for m in methods:
            for k, obj, spent, viol in m["history"]:
                writer.writerow([m["method"], k, obj, spent, viol])

    trace_rows = []
    for method in methods:
        for trace in method.get("trace", []):
            trace_rows.append({"method": method["method"], "family": method["family"], **trace})
    if trace_rows:
        write_csv(out_dir / "optimization_traces.csv", trace_rows)

    # Plot a readable subset.
    best_pb = min([m for m in methods if m["family"] == "PB-AMD"], key=lambda z: z["validation_objective"])
    best_pgd = min([m for m in methods if m["family"] == "PGD"], key=lambda z: z["validation_objective"])
    feasibility_tol = 1e-10 * max(1.0, problem.rho_max)
    feasible_penalties = [
        m for m in methods
        if m["family"] == "Penalty" and np.sum(m["q"]) <= problem.rho_max + feasibility_tol
    ]
    penalties = [m for m in methods if m["family"] == "Penalty"]
    best_pen = min(feasible_penalties or penalties, key=lambda z: z["validation_objective"]) if penalties else None
    uniform = [m for m in methods if m["family"] == "Uniform"][0]
    plot_methods = [best_pb, best_pgd]
    if best_pen is not None:
        plot_methods.extend([best_pen, penalties[0], penalties[-1]])
    plot_methods.append(uniform)
    svg_line_plot([(m["method"], m["q"]) for m in plot_methods], out_dir / "schedules.svg", "Privacy budget allocation q_t", "q_t")
    svg_history_plot([(m["method"], m["history"]) for m in plot_methods if len(m["history"]) > 1], out_dir / "objectives.svg", "Optimization history")

    config = {
        "seed": args.seed,
        "T": problem.T,
        "d": problem.d,
        "n_public_design": int(problem.x_design.shape[0]),
        "n_train": int(problem.x_train.shape[0]),
        "n_test": int(problem.x_test.shape[0]),
        "batch_size": int(len(problem.batches[0])),
        "rho_max": problem.rho_max,
        "q_min": problem.q_min,
        "sensitivity": problem.sensitivity,
        "clip": problem.clip,
        "clipping_operator": "g / sqrt(1 + ||g||^2 / clip^2)",
        "aggregation": "average of smoothly clipped per-example gradients",
        "sensitivity_model": "replacement adjacency; public calibration bound 2 * clip / batch_size",
        "privacy_charge_interpretation": "q_t is a certified upper bound; exact worst-case charge is (Delta_exact / Delta_hat)^2 q_t",
        "privacy_amplification": "none; accountant conditions on public batches and uses direct adaptive composition",
        "noise_gradient_std": "sensitivity / sqrt(2 * q_t)",
        "schedule_design_data": "independent public proxy sample",
        "selection": "independent public validation Gaussian paths",
        "final_evaluation": "fresh private-training Gaussian paths shared across methods",
        "l2": problem.l2,
        "eta": float(problem.eta[0]),
        "K": args.K,
        "M_opt": args.M_opt,
        "M_val": args.M_val,
        "M_eval": args.M_eval,
        "alphas": alphas,
        "pgd_alphas": pgd_alphas,
        "penalty_grid": penalty_grid,
        "optimization_seed": opt_seed,
        "validation_seed": val_seed,
        "evaluation_seed": eval_seed,
        "selected_pb_method": best_pb["method"],
        "selected_pgd_method": best_pgd["method"],
        "selected_penalty_method": best_pen["method"] if best_pen else None,
        "delta": args.delta,
        "epsilon_target": privacy_epsilon(problem.rho_max, args.delta),
        "q_min_over_uniform": problem.q_min / (problem.rho_max / problem.T),
        "test_set_role": "independent held-out evaluation only; never used for schedule or hyperparameter selection",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_report(out_dir, summary_rows, plot_methods, config)

    print(json.dumps({
        "out_dir": str(out_dir),
        "best_pb": best_pb["method"],
        "best_penalty": best_pen["method"] if best_pen else None,
        "summary_csv": str(out_dir / "summary.csv"),
        "report": str(out_dir / "report.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
