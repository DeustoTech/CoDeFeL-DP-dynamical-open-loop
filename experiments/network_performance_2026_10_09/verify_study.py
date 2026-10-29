"""Validate all outputs and independently replay prespecified individual networks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import run_study as s
from test_study import direct_metrics
import numpy as np


def replay(case):
    saved = json.loads((s.RESULTS / "cases" / (s.stem(case) + ".json")).read_text())
    schedule = json.loads((s.RESULTS / "schedules" / (s.stem(case) + ".json")).read_text())
    p = s.make_problem(case)
    indices = [0, 127, 255]
    # Reconstruct the original chunk-major Gaussian stream, without changing path identity.
    rng = np.random.default_rng(case[1] + 1000)
    innovations = np.empty((3, p.T, p.shape.p))
    for start in range(0, 256, 16):
        for t in range(p.T):
            block = rng.normal(size=(16, p.shape.p))
            for j, k in enumerate(indices):
                if start <= k < start + 16:
                    innovations[j, t] = block[k-start]
    errors = []
    for method, q in schedule["schedules"].items():
        theta = np.repeat(p.theta_design_ref[0][None], 3, axis=0)
        for t, idx in enumerate(p.batches):
            grad = s.f.mlp_grad(theta, p.x_train[idx], p.y_train[idx], p.shape,
                                p.l2, p.clip, p.active_masks[t])
            theta = theta - p.eta[t] * grad - p.eta[t] * p.sensitivity / np.sqrt(2*q[t]) * innovations[:, t] * p.active_masks[t]
        for j, k in enumerate(indices):
            ce, acc = direct_metrics(theta[j], p.x_test, p.y_test, p.shape)
            errors.append(s.close(ce, saved["metrics"][method]["cross_entropy"][k], "replay CE"))
            errors.append(s.close(acc, saved["metrics"][method]["accuracy"][k], "replay accuracy", 1e-14))
            errors.append(s.close(np.sum((theta[j] - p.theta_ref[-1])**2),
                                  saved["metrics"][method]["tracking"][k], "replay tracking"))
    # The matched reference uses the same training update with its noise removed.
    theta = p.theta_design_ref[0].copy()
    for t, idx in enumerate(p.batches):
        grad = s.f.mlp_grad(theta[None], p.x_train[idx], p.y_train[idx], p.shape, p.l2, p.clip, p.active_masks[t])[0]
        theta -= p.eta[t] * grad
    ce, acc = direct_metrics(theta, p.x_test, p.y_test, p.shape)
    errors.append(s.close(ce, saved["no_noise"]["cross_entropy"], "reference CE"))
    errors.append(s.close(acc, saved["no_noise"]["accuracy"], "reference accuracy", 1e-14))
    report = {"case": case, "individual_networks": 9, "no_noise_networks": 1,
              "max_scaled_error": max(errors)}
    s.save(s.RESULTS / "replay" / (s.stem(case) + ".json"), report)
    print("VERIFIED", s.stem(case), report["max_scaled_error"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads((s.RESULTS / "protocol.json").read_text())
    assert protocol == json.loads(json.dumps(s.protocol()))
    locked = s.sha(s.RESULTS / "protocol.json")
    budget_error, archive_error = 0., 0.
    cases = s.cases()
    assert len(list((s.RESULTS / "cases").glob("*.json"))) == len(cases) == 60
    hashes = {}
    for case in cases:
        path = s.RESULTS / "cases" / (s.stem(case) + ".json")
        cp = s.RESULTS / "schedules" / path.name
        data, sched = json.loads(path.read_text()), json.loads(cp.read_text())
        assert data["protocol_sha256"] == sched["protocol_sha256"] == locked
        assert data["schedule_sha256"] == s.sha(cp)
        assert data["case"] == sched["case"] == list(case)
        for q in sched["schedules"].values():
            assert len(q) == 45 and min(q) >= 1e-5
            budget_error = max(budget_error, abs(sum(q)-case[2]))
        assert budget_error < 1e-12 and sched["kkt_residual"] <= 5e-5*(1+1e-10)
        for method in s.METHODS:
            values = data["metrics"][method]
            assert all(len(a) == 256 and np.all(np.isfinite(a)) for a in values.values())
            assert np.min(values["accuracy"]) >= 0 and np.max(values["accuracy"]) <= 1
            assert np.min(values["cross_entropy"]) >= 0
            assert np.min(np.array(values["regularized_loss"]) - values["cross_entropy"]) >= 0
            # Each path is one model evaluated on exactly 3,000 examples.
            correct = np.array(values["accuracy"]) * 3000
            np.testing.assert_allclose(correct, np.rint(correct), rtol=0, atol=1e-10)
        if case[0] == "standard":
            assert sched["schedules"]["LR"] == sched["schedules"]["Uniform"]
            assert data["metrics"]["LR"] == data["metrics"]["Uniform"]
        archive_error = max(archive_error, max(data["archive_checks"].values()))
        lrpath = s.ROOT / "experiments/lr_baseline_2026_10_08/results" / ("fashion-" + path.name)
        assert s.sha(lrpath) == data["lr_archive_sha256"]
        hashes[str(path.relative_to(s.ROOT))] = s.sha(path)
        hashes[str(cp.relative_to(s.ROOT))] = s.sha(cp)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replay, c) for c in cases]
        replays = [future.result() for future in as_completed(futures)]
    report = {"status": "passed", "cases": 60, "seed_budget_scenario_method_records": 180,
        "evaluated_noisy_networks": 60*3*256, "independently_replayed_noisy_networks": 60*9,
        "independently_replayed_no_noise_networks": 60,
        "max_budget_error": budget_error, "max_archive_scaled_error": archive_error,
        "max_replay_scaled_error": max(r["max_scaled_error"] for r in replays),
        "files_sha256": hashes, "protocol_sha256": locked,
        "verifier_sha256": s.sha(__file__)}
    s.save(s.RESULTS / "verification.json", report)
    print({k: v for k, v in report.items() if k != "files_sha256"})


if __name__ == "__main__":
    main()
