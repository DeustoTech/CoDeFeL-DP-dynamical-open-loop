"""Audit every secondary Armijo trial and replay both final adjoints."""
import json
import armijo as a
b, np = a.b, a.np


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(a.OUT / "protocol.json") == a.protocol()
    unit = read(b.HERE / "armijo_unit_test_report.json")
    assert unit["status"] == "passed" and unit["source_sha256"] == a.sources()
    records = []
    for seed in b.seeds("nonlinear_head", False):
        ctx = b.make_context("nonlinear_head", seed)
        for method in b.METHODS:
            path = a.OUT / f"{seed}_{method}.json"
            result = read(path)
            assert result["source_sha256"] == a.sources()
            assert (result["seed"], result["method"], result["paths"]) == (seed, method, 128)
            trace = result["trace"]
            assert result["updates"] == len(trace) - 1
            original = read(b.case_path(b.HERE / "results", "test", "nonlinear_head", seed, method, .3))
            np.testing.assert_allclose(trace[0]["gradient"], original["trace"][0]["gradient"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(trace[0]["objective"], original["trace"][0]["objective"], rtol=1e-12)
            F0, count = trace[0]["objective"], 0
            scale = .5 / F0 * (1. if method == "PB-AMD" else .5 / 120)
            expected_alpha = .3 * scale
            maximum_budget_error = 0.
            for k, p in enumerate(trace):
                assert p["gradient_evaluations"] == k + 1 and p["forward_only_evaluations"] == count
                q, g = np.array(p["q"]), np.array(p["gradient"])
                error = abs(float(q.sum()) - .5)
                assert error < 1e-12 and q.min() >= 1e-5 - 1e-14
                maximum_budget_error = max(maximum_budget_error, error)
                vertices = np.full((120, 120), 1e-5) + (.5 - 120e-5) * np.eye(120)
                gap = max(0., float(((q[None, :] - vertices) @ g).max()))
                np.testing.assert_allclose(gap / F0, p["relative_gap"], atol=1e-11, rtol=1e-11)
                for i, trial in enumerate(p["line_search"]):
                    assert abs(trial["alpha"] / expected_alpha - 1) < 1e-12
                    qt = b.update(q, g, method, trial["alpha"], .5, 1e-5)
                    direction = float(g @ (qt - q))
                    np.testing.assert_allclose(direction, trial["directional"], rtol=1e-10, atol=1e-12)
                    accepted = trial["objective"] <= p["objective"] + .1 * direction + 1e-12 * max(1., p["objective"])
                    assert trial["accepted"] == accepted
                    if accepted:
                        assert i == len(p["line_search"]) - 1
                        expected_alpha = min(1.1 * trial["alpha"], 2. * scale)
                    else:
                        expected_alpha *= .5
                count += len(p["line_search"])
                if k + 1 < len(trace):
                    assert p["relative_gap"] > 1e-6
                    assert p["line_search"][-1]["accepted"]
                    np.testing.assert_allclose(qt, trace[k + 1]["q"], atol=2e-13, rtol=1e-12)
                    np.testing.assert_allclose(p["line_search"][-1]["objective"], trace[k + 1]["objective"], atol=1e-10, rtol=1e-10)
                    assert trace[k + 1]["seconds"] >= p["seconds"]
                else:
                    assert len(trace) == 61 or p["relative_gap"] <= 1e-6
                    assert not p["line_search"]
            last = trace[-1]
            f, g = ctx.serial_fg(np.array(last["q"]))
            np.testing.assert_allclose(f, last["objective"], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(g, last["gradient"], atol=1e-9, rtol=1e-8)
            records.append({"seed": seed, "method": method, "file_sha256": b.digest(path),
                "points": len(trace), "forward_calls": count,
                "max_budget_error": maximum_budget_error,
                "replay_objective_scaled_error": abs(f - last["objective"]) / max(1., abs(f)),
                "replay_gradient_scaled_error": float(np.max(np.abs(g - last["gradient"]))) / max(1., float(np.max(np.abs(g))))})
            print(f"VERIFIED Armijo {seed} {method}", flush=True)
    assert len(records) == 20
    b.save(a.OUT / "verification.json", {"status": "passed", "solves": 20,
        "replayed_test_endpoints": 20, "source_sha256": a.sources(), "records": records,
        "max_budget_error": max(r["max_budget_error"] for r in records),
        "max_replay_objective_scaled_error": max(r["replay_objective_scaled_error"] for r in records),
        "max_replay_gradient_scaled_error": max(r["replay_gradient_scaled_error"] for r in records)})


if __name__ == "__main__":
    main()
