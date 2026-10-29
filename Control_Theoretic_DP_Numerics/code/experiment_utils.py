from __future__ import annotations

import math

import numpy as np


def exponential_privacy_schedule(
    horizon: int,
    rho: float,
    q_min: float,
    end_to_start_ratio: float,
) -> np.ndarray:
    """Exact-budget geometric charge schedule with decreasing noise variance."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if q_min <= 0.0 or rho <= horizon * q_min:
        raise ValueError("invalid lower-bounded simplex parameters")
    if not np.isfinite(end_to_start_ratio) or end_to_start_ratio < 1.0:
        raise ValueError("end_to_start_ratio must be finite and at least one")

    if horizon == 1:
        weights = np.ones(1)
    else:
        exponents = np.arange(horizon, dtype=float) / float(horizon - 1)
        weights = np.exp(np.log(end_to_start_ratio) * exponents)
    residual_budget = rho - horizon * q_min
    schedule = q_min + residual_budget * weights / float(weights.sum())
    schedule[int(np.argmax(schedule))] += rho - float(schedule.sum())
    return schedule


def project_lower_bounded_simplex(
    values: np.ndarray,
    rho: float,
    q_min: float,
    tolerance: float = 1e-13,
) -> np.ndarray:
    """Euclidean projection onto {q: q >= q_min, sum(q) = rho}."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector")
    residual_budget = rho - len(values) * q_min
    if q_min <= 0.0 or residual_budget <= 0.0:
        raise ValueError("invalid lower-bounded simplex parameters")

    shifted = values - q_min
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered) - residual_budget
    indices = np.arange(1, len(values) + 1)
    active = np.flatnonzero(ordered - cumulative / indices > 0.0)
    if len(active) == 0:
        raise RuntimeError("failed to identify the simplex projection threshold")
    last = int(active[-1])
    threshold = cumulative[last] / float(last + 1)
    projected = q_min + np.maximum(shifted - threshold, 0.0)
    projected[int(np.argmax(projected))] += rho - float(projected.sum())
    error = abs(float(projected.sum()) - rho)
    if np.min(projected) < q_min - tolerance or error > tolerance * max(1.0, rho):
        raise RuntimeError("lower-bounded simplex projection failed its feasibility check")
    return projected


def floor_aware_entropy_step(
    q: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    rho: float,
    q_min: float,
    tolerance: float = 1e-13,
    max_bisection: int = 160,
) -> tuple[np.ndarray, float, float]:
    """Solve the lower-bounded entropic prox in the log domain."""
    q = np.asarray(q, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if q.shape != gradient.shape:
        raise ValueError("q and gradient must have the same shape")
    if alpha <= 0.0 or q_min <= 0.0 or rho <= len(q) * q_min:
        raise ValueError("invalid floor-aware prox parameters")
    if np.any(q < q_min) or not np.all(np.isfinite(gradient)):
        raise ValueError("the prox input must be feasible and finite")

    log_unscaled = np.log(q) - alpha * gradient
    log_floor = math.log(q_min)

    def candidate(log_tau: float) -> np.ndarray:
        exponent = np.clip(log_tau + log_unscaled, -745.0, 700.0)
        return np.maximum(q_min, np.exp(exponent))

    # At the lower endpoint every coordinate is at the floor. At the upper
    # endpoint at least one unconstrained coordinate exceeds the full budget.
    lower = float(np.min(log_floor - log_unscaled) - 2.0)
    upper = float(math.log(rho) - np.max(log_unscaled) + 2.0)
    if float(candidate(lower).sum()) >= rho or float(candidate(upper).sum()) <= rho:
        raise RuntimeError("failed to bracket the entropy normalization scalar")

    scale = max(1.0, rho)
    for _ in range(max_bisection):
        midpoint = 0.5 * (lower + upper)
        total = float(candidate(midpoint).sum())
        if abs(total - rho) <= tolerance * scale:
            lower = upper = midpoint
            break
        if total < rho:
            lower = midpoint
        else:
            upper = midpoint

    log_tau = 0.5 * (lower + upper)
    q_next = candidate(log_tau)
    # Remove the final roundoff error without changing the active set.
    q_next[int(np.argmax(q_next))] += rho - float(q_next.sum())
    feasibility_error = abs(float(q_next.sum()) - rho)
    if np.min(q_next) < q_min - tolerance or feasibility_error > 10.0 * tolerance * scale:
        raise RuntimeError("floor-aware entropy step failed its feasibility check")
    return q_next, log_tau, feasibility_error


def entropy_divergence(z: np.ndarray, q: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    q = np.asarray(q, dtype=float)
    return max(float(np.sum(z * np.log(z / q) - z + q)), 0.0)


def variational_gap(
    q: np.ndarray,
    gradient: np.ndarray,
    rho: float,
    q_min: float,
) -> float:
    """Return max_{z in Q} <gradient, q-z> for the bounded simplex."""
    q = np.asarray(q, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    residual_budget = rho - len(q) * q_min
    minimum_linear_value = q_min * float(gradient.sum()) + residual_budget * float(gradient.min())
    return max(float(np.dot(gradient, q)) - minimum_linear_value, 0.0)


def mirror_residual(
    q: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    rho: float,
    q_min: float,
) -> float:
    prox, _, _ = floor_aware_entropy_step(q, gradient, alpha, rho, q_min)
    return entropy_divergence(prox, q) / (alpha * alpha)


def projected_gradient_residual(
    q: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    rho: float,
    q_min: float,
) -> float:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    projected = project_lower_bounded_simplex(q - alpha * gradient, rho, q_min)
    return float(np.linalg.norm(q - projected) / alpha)


def active_kkt_residual(
    q: np.ndarray,
    gradient: np.ndarray,
    q_min: float,
    active_tolerance: float = 1e-9,
) -> float:
    q = np.asarray(q, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    active = q > q_min + active_tolerance
    if not np.any(active):
        return float("inf")
    level = float(np.mean(gradient[active]))
    active_error = float(np.max(np.abs(gradient[active] - level)))
    floor_error = float(np.max(np.maximum(level - gradient[~active], 0.0))) if np.any(~active) else 0.0
    return max(active_error, floor_error)


def inequality_budget_kkt_residual(
    q: np.ndarray,
    gradient: np.ndarray,
    rho: float,
) -> float:
    """KKT residual for positive q and the inequality sum(q) <= rho."""
    q = np.asarray(q, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if q.shape != gradient.shape or np.any(q <= 0.0):
        raise ValueError("q must be positive and match the gradient")
    constraint = float(q.sum() - rho)
    multiplier = max(-float(gradient.mean()), 0.0)
    stationarity = float(np.max(np.abs(gradient + multiplier)))
    primal = max(constraint, 0.0)
    complementarity = abs(multiplier * constraint)
    return max(stationarity, primal, complementarity)


def floor_diagnostics(q: np.ndarray, q_min: float, tolerance: float = 1e-9) -> dict[str, float | int]:
    q = np.asarray(q, dtype=float)
    excess = q - q_min
    return {
        "floor_count": int(np.sum(excess <= tolerance)),
        "min_floor_excess": float(np.min(excess)),
    }


def privacy_epsilon(rho: float, delta: float) -> float:
    if rho < 0.0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid privacy parameters")
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    value = d
    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        value *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        update = d * c
        value *= update
        if abs(update - 1.0) <= epsilon:
            return value
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0 or not 0.0 <= x <= 1.0:
        raise ValueError("invalid incomplete-beta parameters")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_incomplete_beta(
        x,
        0.5 * degrees_of_freedom,
        0.5,
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_quantile(1.0 - probability, degrees_of_freedom)
    lower, upper = 0.0, 1.0
    while student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def student_t_two_sided_pvalue(value: float, degrees_of_freedom: int) -> float:
    """Return the two-sided Student-t tail without subtractive cancellation."""
    if degrees_of_freedom <= 0 or not math.isfinite(value):
        raise ValueError("invalid Student-t parameters")
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    return regularized_incomplete_beta(
        x,
        0.5 * degrees_of_freedom,
        0.5,
    )


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Return step-down Holm adjusted p-values in the original order."""
    if any(not 0.0 <= value <= 1.0 for value in pvalues):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
