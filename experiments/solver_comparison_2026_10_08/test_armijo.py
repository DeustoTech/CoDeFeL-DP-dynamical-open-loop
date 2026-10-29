"""Check secondary line-search bookkeeping and the original PB-AMD recurrence."""
from dataclasses import replace
import json

import armijo as a
b, np = a.b, a.np


def toy_check():
    class Problem:
        T, rho_max, q_min = 8, .5, 1e-5
    class Context:
        problem, task, seed = Problem(), "unit", 0
        xi = np.zeros((8, 1, 1))
        uniform = np.full(8, .5 / 8)
        def fg(self, q):
            weights = np.linspace(1., 4., 8)
            return float(np.sum(weights / q)), -weights / q**2
    old_objective = a.objective
    a.objective = lambda ctx, q: ctx.fg(q)[0]
    try:
        for method in b.METHODS:
            result = a.solve(Context(), method)
            trace = result["trace"]
            assert trace[-1]["relative_gap"] <= 1e-6
            assert trace[-1]["forward_only_evaluations"] == sum(len(p["line_search"]) for p in trace)
            for left, right in zip(trace, trace[1:]):
                assert left["line_search"][-1]["accepted"]
                assert all(not x["accepted"] for x in left["line_search"][:-1])
                np.testing.assert_allclose(left["line_search"][-1]["objective"], right["objective"], rtol=1e-12)
                expected = b.update(np.array(left["q"]), np.array(left["gradient"]), method,
                                    left["line_search"][-1]["alpha"], .5, 1e-5)
                np.testing.assert_allclose(expected, right["q"], atol=1e-13, rtol=1e-12)
    finally:
        a.objective = old_objective


def original_recurrence_check():
    ctx = b.make_context("nonlinear_head", 225001, paths=4)
    p = ctx.problem
    short = replace(p, batches=p.batches[:5], eta=p.eta[:5],
                    active_masks=p.active_masks[:5], theta_design_ref=p.theta_design_ref[:6],
                    theta_ref=p.theta_ref[:6])
    ctx = b.Context("nonlinear_head", ctx.seed, short, ctx.xi[:5])
    result = a.solve(ctx, "PB-AMD")
    steps = min(5, len(result["trace"]) - 1)
    candidates, history, _ = b.head.from_uniform(short, ctx.xi, ctx.uniform, steps)
    np.testing.assert_allclose(candidates[-1], result["trace"][steps]["q"], atol=2e-12, rtol=1e-10)
    for k, row in enumerate(history):
        np.testing.assert_allclose(row["objective"], result["trace"][k]["objective"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(row["alpha"], result["trace"][k]["line_search"][-1]["alpha"], atol=1e-12, rtol=1e-12)
    return float(np.max(np.abs(candidates[-1] - result["trace"][steps]["q"])))


if __name__ == "__main__":
    toy_check()
    error = original_recurrence_check()
    b.save(b.HERE / "armijo_unit_test_report.json", {"status": "passed",
           "original_pb_recurrence_max_error": error, "source_sha256": a.sources()})
    print("Armijo unit tests and original PB-AMD recurrence passed", error)
