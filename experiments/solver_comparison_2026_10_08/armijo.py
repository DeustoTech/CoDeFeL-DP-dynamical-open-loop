"""Secondary, symmetric Armijo comparison on all ten nonlinear-head cases."""
import gc
import json
import time

import benchmark as b
np = b.np
OUT = b.HERE / "armijo_results"


def sources():
    return b.sources() | {str(b.Path(__file__).relative_to(b.ROOT)): b.digest(__file__)}


def objective(ctx, q):
    chunks = np.array_split(ctx.xi, min(4, ctx.xi.shape[1]), axis=1)
    futures = [b.POOL.submit(b.fashion.objective_only, ctx.problem, q, xi,
                             ctx.problem.theta_design_ref[0]) for xi in chunks]
    return sum(xi.shape[1] / ctx.xi.shape[1] * future.result()
               for xi, future in zip(chunks, futures))


def protocol():
    return {"version": 1, "task": "nonlinear_head", "rho": b.RHO,
        "seeds": b.seeds("nonlinear_head", False), "paths": 128,
        "max_updates": 60, "stop_relative_gap": 1e-6,
        "initial_gamma": .3, "maximum_gamma": 2., "armijo_c": .1,
        "shrink": .5, "growth": 1.1, "max_trials": 30,
        "roundoff_slack": "1e-12*max(1,F(q)) as in the original neural PB-AMD implementation",
        "step_scaling": "PB alpha=gamma*rho/F(q0); PGD alpha=gamma*rho^2/(T*F(q0))",
        "timing": "Same serial alternating-order protocol and four-worker neural oracle as the primary study; every forward-only line-search call is counted and timed.",
        "status": "Secondary analysis, not preregistered with the primary fixed-step study.",
        "reason": "Initiated after fixed-step PB-AMD failed the tolerance on head seed 213102. The manuscript head utility experiments use Armijo, so both methods receive matched Armijo adaptation on every one of the same ten instances; all fixed-step results remain in the primary report.",
        "no_reselection": "No seeds, SAA paths, thresholds or model settings changed; no extra stepsize grid.",
        "source_sha256": sources()}


def lock():
    expected = protocol()
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected
    else:
        b.save(path, expected)


def solve(ctx, method):
    p, q = ctx.problem, ctx.uniform
    trace = []
    forward_calls = 0
    gc.collect()
    start = time.perf_counter()
    initial = scale = alpha = None
    for k in range(61):
        value, gradient = ctx.fg(q)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        if k == 0:
            initial = value
            scale = p.rho_max / initial
            if method == "PGD":
                scale *= p.rho_max / p.T
            alpha = .3 * scale
        gap = b.utils.variational_gap(q, gradient, p.rho_max, p.q_min)
        kkt = b.utils.active_kkt_residual(q, gradient, p.q_min)
        error = abs(float(q.sum()) - p.rho_max)
        assert error < 1e-12 and q.min() >= p.q_min - 1e-14
        point = {"iteration": k, "gradient_evaluations": k + 1,
            "forward_only_evaluations": forward_calls, "objective": float(value),
            "relative_objective": float(value / initial), "gap": float(gap),
            "relative_gap": float(gap / initial), "kkt": float(kkt),
            "scaled_kkt": float(p.rho_max * kkt / initial),
            "budget_error": error, "q": q.tolist(), "gradient": gradient.tolist(),
            "seconds": time.perf_counter() - start, "line_search": []}
        trace.append(point)
        if gap / initial <= 1e-6 or k == 60:
            break
        for attempt in range(30):
            trial = b.update(q, gradient, method, alpha, p.rho_max, p.q_min)
            trial_f = objective(ctx, trial)
            forward_calls += 1
            directional = float(gradient @ (trial - q))
            accepted = bool(trial_f <= value + .1 * directional + 1e-12 * max(1., value))
            point["line_search"].append({"alpha": alpha, "objective": trial_f,
                                          "directional": directional, "accepted": accepted})
            if accepted:
                break
            alpha *= .5
        else:
            raise RuntimeError(f"line search failed: {ctx.seed} {method}")
        q = trial
        alpha = min(alpha * 1.1, 2. * scale)
    return {"task": ctx.task, "seed": ctx.seed, "method": method, "rho": p.rho_max,
        "q_min": p.q_min, "T": p.T, "paths": ctx.xi.shape[1],
        "max_updates": 60, "updates": len(trace) - 1, "trace": trace,
        "source_sha256": sources()}


def run():
    lock()
    for i, seed in enumerate(b.seeds("nonlinear_head", False)):
        order = b.METHODS if i % 2 == 0 else b.METHODS[::-1]
        if all((OUT / f"{seed}_{m}.json").exists() for m in order):
            continue
        print(f"BUILD Armijo seed={seed}", flush=True)
        ctx = b.make_context("nonlinear_head", seed)
        f, _ = ctx.fg(ctx.uniform)
        np.testing.assert_allclose(f, objective(ctx, ctx.uniform), rtol=1e-12, atol=1e-12)
        for method in order:
            path = OUT / f"{seed}_{method}.json"
            if path.exists():
                assert json.loads(path.read_text())["source_sha256"] == sources()
                continue
            result = solve(ctx, method)
            result["order_position"] = list(order).index(method)
            b.save(path, result)
            last = result["trace"][-1]
            print(f"ARMIJO {seed} {method}: G={last['gradient_evaluations']} F={last['forward_only_evaluations']} gap={last['relative_gap']:.4g} time={last['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    import sys
    if "--lock-only" in sys.argv:
        lock()
    else:
        run()
