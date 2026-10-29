"""Add the existing penalty-and-repair baseline to three frozen comparisons."""
import argparse
import gc
import json
import math
import time

import benchmark as b
import controlled_runtime_benchmark as original

np = b.np
OUT = b.HERE / "penalty_results"
TASKS = ("tabular", "fashion_restart", "nonlinear_head")


def sources():
    return b.sources() | {
        str(b.Path(__file__).relative_to(b.ROOT)): b.digest(__file__),
        str(b.Path(original.__file__).relative_to(b.ROOT)): b.digest(original.__file__)}


def candidates(task):
    return original.TABULAR_PENALTY if task == "tabular" else original.FASHION_PENALTY


def protocol():
    return {"version": 1, "tasks": TASKS, "rho": b.RHO,
        "paths": {t: b.PATHS[t] for t in TASKS},
        "calibration_seeds": {t: b.seeds(t, True) for t in TASKS},
        "test_seeds": {t: b.seeds(t, False) for t in TASKS},
        "candidates": {t: candidates(t) for t in TASKS},
        "calibration_updates": 30, "test_updates": {t: b.UPDATES[t] for t in TASKS},
        "selection": "Minimum mean log repaired relative variational gap after 30 updates on three public calibration instances; numerical floor 1e-16, smaller beta breaks ties.",
        "algorithm": "Original log-variance Adam recurrence, same clipping bounds, original best-penalized-iterate return rule, and original radial exact-budget repair.",
        "calibration_diagnostics": "The repaired gradient is evaluated only at the final returned candidate; intermediate raw adjoints are recorded.",
        "test_diagnostics": "Repair the best penalized iterate at every prefix; evaluate the same feasible variational gap. Reuse only exactly equal cached points. Stop at gap/F(q0)<=1e-6 or the same raw-update cap.",
        "timing": "Only new penalty solves; prior PB-AMD/PGD curves unchanged. Same host, backend, paths and instances. Serial timing includes raw adjoints, repair, repaired-candidate diagnostic adjoints, and common diagnostics; setup, warmup and output excluded. Additional diagnostic calls are counted separately.",
        "scope": "Only the three manuscript convergence panels; does not replace historical penalty results or retune the other methods.",
        "source_sha256": sources()}


def lock():
    # JSON normalization makes tuple-valued imported grids stable on resume.
    expected = json.loads(json.dumps(protocol()))
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected, "penalty source/protocol mismatch"
    else:
        b.save(path, expected)


def case_path(phase, task, seed, beta):
    return OUT / phase / f"{task}_{seed}_beta{beta:g}.json"


def solve(ctx, beta, lr, updates, *, diagnostics_every_step, tolerance=None):
    p = ctx.problem
    v = -np.log(2. * ctx.uniform)
    m, s = np.zeros(p.T), np.zeros(p.T)
    trace, best, cache = [], None, None
    repair_calls, diagnostic_seconds = 0, 0.
    gc.collect()
    start = time.perf_counter()
    for k in range(updates + 1):
        raw = np.clip(.5 * np.exp(-v), 1e-8, 2. * p.rho_max)
        value, gradient = ctx.fg(raw)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        if k == 0:
            F0 = float(value)
            assert F0 > 0
        phi = float(raw.sum() - p.rho_max)
        penalized = float(value + .5 * beta * max(phi, 0.)**2)
        if best is None or penalized < best[0]:
            best = (penalized, k, raw.copy(), float(value), gradient.copy())
        point = {"iteration": k, "raw_q": raw.tolist(), "raw_objective": float(value),
            "raw_gradient": gradient.tolist(), "penalized_objective": penalized,
            "raw_budget_violation": max(phi, 0.), "best_iteration": best[1],
            "relative_gap": None, "raw_gradient_evaluations": k + 1}
        if diagnostics_every_step or k == updates:
            repaired = original.repair_exact_budget(best[2], p.rho_max, p.q_min)
            if cache is not None and np.array_equal(repaired, cache[0]):
                repaired_value, repaired_gradient = cache[1:]
            elif np.array_equal(repaired, best[2]):
                repaired_value, repaired_gradient = best[3:]
                cache = (repaired.copy(), repaired_value, repaired_gradient.copy())
            else:
                tick = time.perf_counter()
                repaired_value, repaired_gradient = ctx.fg(repaired)
                diagnostic_seconds += time.perf_counter() - tick
                repair_calls += 1
                cache = (repaired.copy(), repaired_value, repaired_gradient.copy())
            gap = b.utils.variational_gap(repaired, repaired_gradient, p.rho_max, p.q_min)
            budget_error = abs(float(repaired.sum()) - p.rho_max)
            assert budget_error < 1e-12 and repaired.min() >= p.q_min - 1e-14
            point.update({"q": repaired.tolist(), "gradient": repaired_gradient.tolist(),
                "objective": float(repaired_value), "relative_objective": float(repaired_value / F0),
                "relative_gap": float(gap / F0), "budget_error": budget_error,
                "scaled_kkt": float(p.rho_max * b.utils.active_kkt_residual(
                    repaired, repaired_gradient, p.q_min) / F0)})
        point.update({"repair_gradient_evaluations": repair_calls,
                      "gradient_evaluations": k + 1 + repair_calls,
                      "forward_only_evaluations": 0,
                      "diagnostic_seconds": diagnostic_seconds,
                      "seconds": time.perf_counter() - start})
        trace.append(point)
        if tolerance is not None and point["relative_gap"] is not None and point["relative_gap"] <= tolerance:
            break
        if k < updates:
            grad_v = -raw * gradient - beta * max(phi, 0.) * raw
            m = .9 * m + .1 * grad_v
            s = .999 * s + .001 * grad_v**2
            v -= lr * (m / (1. - .9**(k + 1))) / (np.sqrt(s / (1. - .999**(k + 1))) + 1e-8)
            v = np.clip(v, -math.log(4. * p.rho_max), -math.log(2e-8))
    return {"task": ctx.task, "seed": ctx.seed, "method": "Penalty + repair",
            "beta": beta, "lr": lr, "rho": p.rho_max, "q_min": p.q_min,
            "paths": ctx.xi.shape[1], "max_updates": updates, "updates": len(trace) - 1,
            "diagnostics_every_step": diagnostics_every_step,
            "stopping_tolerance": tolerance, "initial_objective": F0,
            "best_raw_q": best[2].tolist(), "trace": trace,
            "source_sha256": sources()}


def calibrate(tasks):
    for task in tasks:
        for seed in b.seeds(task, True):
            if all(case_path("calibration", task, seed, beta).exists() for beta, _ in candidates(task)):
                continue
            print(f"BUILD penalty calibration {task} {seed}", flush=True)
            ctx = b.make_context(task, seed)
            ctx.fg(ctx.uniform)
            for beta, lr in candidates(task):
                path = case_path("calibration", task, seed, beta)
                if path.exists():
                    continue
                result = solve(ctx, beta, lr, 30, diagnostics_every_step=False)
                b.save(path, result)
                print(f"CAL penalty {task} {seed} beta={beta:g}: gap={result['trace'][-1]['relative_gap']:.5g}", flush=True)
            del ctx
        scores = []
        for beta, lr in candidates(task):
            values = [json.loads(case_path("calibration", task, seed, beta).read_text())["trace"][-1]["relative_gap"]
                      for seed in b.seeds(task, True)]
            scores.append({"beta": beta, "lr": lr, "gaps": values,
                           "log_score": float(np.mean(np.log(np.maximum(values, 1e-16))))})
        best = min(scores, key=lambda x: (x["log_score"], x["beta"]))
        b.save(OUT / "selection" / f"{task}.json", {"selected": best, "candidates": scores})
        print(f"SELECT penalty {task}: beta={best['beta']:g}, lr={best['lr']:g}", flush=True)


def test(tasks):
    for task in tasks:
        selection_path = OUT / "selection" / f"{task}.json"
        selected = json.loads(selection_path.read_text())["selected"]
        beta, lr = selected["beta"], selected["lr"]
        for seed in b.seeds(task, False):
            path = case_path("test", task, seed, beta)
            if path.exists():
                continue
            print(f"BUILD penalty test {task} {seed}", flush=True)
            ctx = b.make_context(task, seed)
            ctx.fg(ctx.uniform)
            result = solve(ctx, beta, lr, b.UPDATES[task], diagnostics_every_step=True, tolerance=1e-6)
            result["selection_sha256"] = b.digest(selection_path)
            b.save(path, result)
            last = result["trace"][-1]
            print(f"TEST penalty {task} {seed}: gap={last['relative_gap']:.5g}, G={last['gradient_evaluations']}, time={last['seconds']:.1f}s", flush=True)
            del ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("all", "calibrate", "test", "lock"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    lock()
    if args.phase in ("all", "calibrate"):
        calibrate(args.tasks)
    if args.phase in ("all", "test"):
        test(args.tasks)
