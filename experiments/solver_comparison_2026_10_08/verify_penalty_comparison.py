"""Audit all penalty updates and independently replay feasible test endpoints."""
import json
import math
import penalty_comparison as a
b, np = a.b, a.np


def read(path):
    return json.loads(path.read_text())


def check_trace(result):
    rows, rho, floor = result["trace"], result["rho"], result["q_min"]
    n = len(rows[0]["raw_q"])
    beta, lr = result["beta"], result["lr"]
    v = -np.log(2 * np.full(n, rho / n))
    m, s, best, cache = np.zeros(n), np.zeros(n), None, None
    repair_calls, max_budget_error = 0, 0.
    assert result["source_sha256"] == a.sources()
    assert result["updates"] == len(rows) - 1
    assert result["initial_objective"] == rows[0]["raw_objective"]
    for k, row in enumerate(rows):
        raw = np.clip(.5 * np.exp(-v), 1e-8, 2 * rho)
        np.testing.assert_allclose(row["raw_q"], raw, atol=2e-13, rtol=2e-12)
        raw = np.array(row["raw_q"])
        g = np.array(row["raw_gradient"])
        assert np.all(np.isfinite(g))
        excess = max(raw.sum() - rho, 0.)
        penalized = row["raw_objective"] + .5 * beta * excess**2
        np.testing.assert_allclose(row["penalized_objective"], penalized, rtol=1e-13, atol=1e-13)
        if best is None or row["penalized_objective"] < best[0]:
            best = (row["penalized_objective"], k, raw.copy())
        assert row["best_iteration"] == best[1]
        if row["relative_gap"] is not None:
            z = np.maximum(best[2] - floor, 0.)
            if z.sum() == 0:
                z = np.ones(n)
            q = floor + (rho - n * floor) * z / z.sum()
            np.testing.assert_allclose(row["q"], q, rtol=2e-13, atol=2e-14)
            if cache is not None and np.array_equal(q, cache):
                pass
            elif np.array_equal(q, best[2]):
                cache = q.copy()
            else:
                repair_calls += 1
                cache = q.copy()
            q, grad = np.array(row["q"]), np.array(row["gradient"])
            vertices = np.full((n, n), floor) + (rho - n * floor) * np.eye(n)
            gap = max(0., float(np.max((q - vertices) @ grad)))
            np.testing.assert_allclose(row["relative_gap"], gap / result["initial_objective"], rtol=1e-10, atol=1e-11)
            max_budget_error = max(max_budget_error, abs(float(q.sum()) - rho))
            assert max_budget_error < 1e-12 and q.min() >= floor - 1e-14
        else:
            assert not result["diagnostics_every_step"] and k < result["max_updates"]
        assert row["raw_gradient_evaluations"] == k + 1
        assert row["repair_gradient_evaluations"] == repair_calls
        assert row["gradient_evaluations"] == k + 1 + repair_calls
        assert 0 <= row["diagnostic_seconds"] <= row["seconds"]
        if k:
            assert rows[k - 1]["seconds"] <= row["seconds"]
        if k < len(rows) - 1:
            if result["stopping_tolerance"] is not None:
                assert row["relative_gap"] > result["stopping_tolerance"]
            gv = -raw * g - beta * excess * raw
            m, s = .9 * m + .1 * gv, .999 * s + .001 * gv**2
            v -= lr * (m / (1 - .9**(k + 1))) / (np.sqrt(s / (1 - .999**(k + 1))) + 1e-8)
            v = np.clip(v, -math.log(4 * rho), -math.log(2e-8))
    np.testing.assert_allclose(result["best_raw_q"], best[2], rtol=0, atol=0)
    assert result["updates"] == result["max_updates"] or rows[-1]["relative_gap"] <= result["stopping_tolerance"]
    return len(rows), max_budget_error


def main():
    a.lock()
    unit = read(b.HERE / "penalty_unit_test_report.json")
    assert unit["status"] == "passed" and unit["source_sha256"] == a.sources()
    for name, expected in read(b.HERE / "data_sha256.json").items():
        assert b.digest(b.ROOT / name) == expected
    points, max_budget, records = 0, 0., []
    for task in a.TASKS:
        selection_path = a.OUT / "selection" / f"{task}.json"
        selection = read(selection_path)
        scores = []
        for beta, lr in a.candidates(task):
            values = []
            for seed in b.seeds(task, True):
                result = read(a.case_path("calibration", task, seed, beta))
                count, error = check_trace(result)
                points, max_budget = points + count, max(max_budget, error)
                values.append(result["trace"][-1]["relative_gap"])
            scores.append({"beta": beta, "lr": lr, "gaps": values,
                           "log_score": float(np.mean(np.log(np.maximum(values, 1e-16))))})
        assert scores == selection["candidates"]
        assert min(scores, key=lambda x: (x["log_score"], x["beta"])) == selection["selected"]
        beta = selection["selected"]["beta"]
        for seed in b.seeds(task, False):
            path = a.case_path("test", task, seed, beta)
            result = read(path)
            assert result["selection_sha256"] == b.digest(selection_path)
            count, error = check_trace(result)
            points, max_budget = points + count, max(max_budget, error)
            baseline_selection = read(b.HERE / "results/selection" / f"{task}.json")
            baseline = read(b.case_path(b.HERE / "results", "test", task, seed, "PB-AMD",
                                       baseline_selection["PB-AMD"]["selected_gamma"]))
            np.testing.assert_allclose(result["initial_objective"], baseline["trace"][0]["objective"], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(result["trace"][0]["raw_gradient"], baseline["trace"][0]["gradient"], atol=1e-10, rtol=1e-11)
            ctx = b.make_context(task, seed)
            last = result["trace"][-1]
            f, g = ctx.serial_fg(np.array(last["q"]))
            np.testing.assert_allclose(f, last["objective"], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(g, last["gradient"], atol=1e-9, rtol=1e-8)
            record = {"task": task, "seed": seed, "case_sha256": b.digest(path),
                "replay_objective_scaled_error": abs(f - last["objective"]) / max(1., abs(f)),
                "replay_gradient_scaled_error": float(np.max(np.abs(g - last["gradient"]))) / max(1., float(np.max(np.abs(g))))}
            records.append(record)
            b.save(a.OUT / "replay" / f"{task}_{seed}.json", record)
            print(f"VERIFIED penalty {task} {seed}", flush=True)
            del ctx
    assert len(records) == 30
    b.save(a.OUT / "verification.json", {"status": "passed", "solves": 84,
        "trajectory_points": points, "max_budget_error": max_budget,
        "replayed_test_endpoints": 30, "records": records, "source_sha256": a.sources()})


if __name__ == "__main__":
    main()
