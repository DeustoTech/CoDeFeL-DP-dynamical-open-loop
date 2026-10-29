"""Check recurrence equivalence, repaired diagnostics, and penalty derivatives."""
from dataclasses import replace
import penalty_comparison as a
b, np = a.b, a.np


def check(task, seed):
    ctx = b.make_context(task, seed, paths=4)
    if task != "tabular":
        p = ctx.problem
        p = replace(p, batches=p.batches[:5], eta=p.eta[:5],
                    active_masks=p.active_masks[:5], theta_design_ref=p.theta_design_ref[:6],
                    theta_ref=p.theta_ref[:6])
        ctx = b.Context(task, seed, p, ctx.xi[:5])
    beta, lr = a.candidates(task)[2]
    result = a.solve(ctx, beta, lr, 8, diagnostics_every_step=True)
    p = ctx.problem
    errors = []
    for k in (1, 3, 8):
        if task == "tabular":
            reference = b.tabular.run_penalty(p, ctx.xi, beta, lr, k, compute_diagnostics=False, record_trace=False)
        else:
            reference = b.fashion.run_penalty(p, p.theta_design_ref[0], ctx.xi, k,
                                             beta, lr, compute_diagnostics=False, record_trace=False)
        expected = a.original.repair_exact_budget(reference["q"], p.rho_max, p.q_min)
        np.testing.assert_allclose(result["trace"][k]["q"], expected, rtol=1e-12, atol=1e-13)
        errors.append(float(np.max(np.abs(np.array(result["trace"][k]["q"]) - expected))))
    for row in result["trace"]:
        f, g = ctx.serial_fg(np.array(row["q"]))
        np.testing.assert_allclose(row["objective"], f, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(row["gradient"], g, rtol=1e-10, atol=1e-11)
        assert row["gradient_evaluations"] == row["raw_gradient_evaluations"] + row["repair_gradient_evaluations"]
    q = 1.3 * ctx.uniform
    v = -np.log(2 * q)
    f, g = ctx.serial_fg(q)
    dv = np.linspace(-1., 1., p.T)
    dv /= np.linalg.norm(dv)
    analytic = float((-q * g - beta * (q.sum() - p.rho_max) * q) @ dv)
    def objective(w):
        r = .5 * np.exp(-w)
        return ctx.serial_fg(r)[0] + .5 * beta * max(r.sum() - p.rho_max, 0.)**2
    difference = (objective(v + 1e-5 * dv) - objective(v - 1e-5 * dv)) / 2e-5
    error = abs(difference - analytic) / max(1e-6, abs(analytic))
    assert error < 2e-4, error
    return {"task": task, "recurrence_max_error": max(errors), "penalty_directional_relative_error": error}


if __name__ == "__main__":
    checks = [check(task, 229001 + i) for i, task in enumerate(a.TASKS)]
    b.save(b.HERE / "penalty_unit_test_report.json", {"status": "passed", "checks": checks,
                                                    "source_sha256": a.sources()})
    print(checks)
