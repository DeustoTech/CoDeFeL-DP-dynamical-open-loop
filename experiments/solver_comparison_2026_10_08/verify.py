"""Audit all traces and independently replay the final objective/adjoint."""
import argparse
import json
from pathlib import Path

import benchmark as b
np = b.np


def read(path):
    return json.loads(path.read_text())


def verify_trace(path):
    result = read(path)
    assert result["source_sha256"] == b.sources()
    trace = result["trace"]
    assert len(trace) == result["updates"] + 1
    assert trace[0]["iteration"] == 0
    p = result
    F0 = trace[0]["objective"]
    expected_alpha = p["gamma"] * p["rho"] / F0
    if p["method"] == "PGD":
        expected_alpha *= p["rho"] / p["T"]
    assert abs(p["alpha"] / expected_alpha - 1) < 1e-14
    np.testing.assert_allclose(trace[0]["q"], np.full(p["T"], p["rho"] / p["T"]), atol=1e-15, rtol=0.)
    max_error = 0.
    for i, point in enumerate(trace):
        assert point["iteration"] == i and point["gradient_evaluations"] == i + 1
        assert point["forward_only_evaluations"] == 0
        q, g = np.array(point["q"]), np.array(point["gradient"])
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(g))
        assert q.shape == g.shape == (p["T"],)
        error = abs(float(q.sum()) - p["rho"])
        assert error < 1e-12 and q.min() >= p["q_min"] - 1e-14
        max_error = max(max_error, error)
        # Direct simplex-vertex calculation is independent of the shared helper.
        slack = p["rho"] - p["T"] * p["q_min"]
        vertex_values = float(g @ q) - p["q_min"] * g.sum() - slack * g
        direct_gap = max(0., float(vertex_values.max()))
        np.testing.assert_allclose(point["gap"], direct_gap, atol=1e-10 * max(1., F0), rtol=1e-12)
        np.testing.assert_allclose(point["relative_gap"], point["gap"] / F0, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(point["relative_objective"], point["objective"] / F0, atol=1e-14, rtol=1e-13)
        assert point["seconds"] > 0
        if i:
            previous = trace[i - 1]
            assert point["seconds"] >= previous["seconds"]
            expected = b.update(np.array(previous["q"]), np.array(previous["gradient"]),
                                p["method"], p["alpha"], p["rho"], p["q_min"])
            np.testing.assert_allclose(q, expected, atol=2e-13, rtol=1e-12)
    tolerance = p["stopping_tolerance"]
    if tolerance is not None:
        assert all(x["relative_gap"] > tolerance for x in trace[:-1])
        assert len(trace) == p["max_updates"] + 1 or trace[-1]["relative_gap"] <= tolerance
    else:
        assert len(trace) == p["max_updates"] + 1
    return {"file": str(path.relative_to(b.HERE)), "sha256": b.digest(path),
            "rows": len(trace), "max_budget_error": max_error}


def verify_all(out):
    assert read(out / "protocol.json") == b.protocol()
    unit = read(b.HERE / "unit_test_report.json")
    assert unit["status"] == "passed" and unit["source_sha256"] == b.sources()
    for path, expected in read(b.HERE / "data_sha256.json").items():
        assert b.digest(b.ROOT / path) == expected
    assert len(list((out / "calibration").glob("*.json"))) == 96
    assert len(list((out / "test").glob("*.json"))) == 80
    summaries = []
    for task in b.TASKS:
        selection_path = out / "selection" / f"{task}.json"
        selection = read(selection_path)
        assert not set(b.seeds(task, True)) & set(b.seeds(task, False))
        for method in b.METHODS:
            scores = []
            for gamma in b.GAMMAS:
                values = []
                for seed in b.seeds(task, True):
                    path = b.case_path(out, "calibration", task, seed, method, gamma)
                    summaries.append(verify_trace(path))
                    result = read(path)
                    assert (result["task"], result["seed"], result["method"], result["gamma"]) == (task, seed, method, gamma)
                    assert result["paths"] == b.PATHS[task]
                    assert result["updates"] == 30
                    values.append(result["trace"][-1]["relative_gap"])
                scores.append((float(np.mean(np.log(np.maximum(values, 1e-16)))), gamma))
            assert min(scores)[1] == selection[method]["selected_gamma"]
        for seed in b.seeds(task, False):
            pair = []
            for method in b.METHODS:
                gamma = selection[method]["selected_gamma"]
                path = b.case_path(out, "test", task, seed, method, gamma)
                summaries.append(verify_trace(path))
                result = read(path)
                assert (result["task"], result["seed"], result["method"], result["gamma"]) == (task, seed, method, gamma)
                assert result["selection_sha256"] == b.digest(selection_path)
                assert result["paths"] == b.PATHS[task]
                assert result["max_updates"] == b.UPDATES[task]
                assert selection_path.stat().st_mtime <= path.stat().st_mtime
                pair.append(result)
            np.testing.assert_allclose(pair[0]["trace"][0]["gradient"], pair[1]["trace"][0]["gradient"], atol=1e-12, rtol=1e-12)
            assert pair[0]["trace"][0]["objective"] == pair[1]["trace"][0]["objective"]
            i = b.seeds(task, False).index(seed)
            assert pair[0]["order_position"] == i % 2
            assert pair[1]["order_position"] == 1 - i % 2
    assert len(summaries) == 176
    return summaries


def replay(out):
    folder = out / "replay"
    reports = []
    for task in b.TASKS:
        selection = read(out / "selection" / f"{task}.json")
        for seed in b.seeds(task, False):
            target = folder / f"{task}_{seed}.json"
            if target.exists():
                previous = read(target)
                assert previous["source_sha256"] == b.sources()
                for method in b.METHODS:
                    gamma = selection[method]["selected_gamma"]
                    assert previous[method]["case_sha256"] == b.digest(b.case_path(out, "test", task, seed, method, gamma))
                reports.append(previous)
                continue
            ctx = b.make_context(task, seed)
            report = {"task": task, "seed": seed, "source_sha256": b.sources()}
            for method in b.METHODS:
                gamma = selection[method]["selected_gamma"]
                path = b.case_path(out, "test", task, seed, method, gamma)
                result = read(path)
                last = result["trace"][-1]
                q = np.array(last["q"])
                f, g = ctx.serial_fg(q)
                np.testing.assert_allclose(f, last["objective"], rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(g, last["gradient"], rtol=1e-8, atol=1e-9)
                report[method] = {"objective_scaled_error": abs(f - last["objective"]) / max(1., abs(f)),
                                  "gradient_scaled_error": float(np.max(np.abs(g - last["gradient"]))) / max(1., float(np.max(np.abs(g)))),
                                  "case_sha256": b.digest(path)}
            b.save(target, report)
            reports.append(report)
            print(f"REPLAY {task} {seed} passed", flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    out = b.HERE / "results"
    traces = verify_all(out)
    report = {"status": "passed", "traces": len(traces), "trajectory_points": sum(x["rows"] for x in traces),
              "max_budget_error": max(x["max_budget_error"] for x in traces),
              "test_instances": 40, "calibration_problems": 12,
              "unit_tests_sha256": b.digest(b.HERE / "unit_test_report.json"),
              "source_sha256": b.sources(), "files": traces,
              "scope": "Recorded updates, common residuals, budgets, source hashes, calibration selection and stopping; optional full-path endpoint replay."}
    if args.replay:
        reports = replay(out)
        report["replayed_test_endpoints"] = 80
        for metric in ("objective_scaled_error", "gradient_scaled_error"):
            report["max_replay_" + metric] = max(r[m][metric] for r in reports for m in b.METHODS)
    b.save(out / "verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
