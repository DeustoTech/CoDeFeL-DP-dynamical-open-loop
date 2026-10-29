"""Independently reconstruct new comparisons from saved case records."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import run_comparison as run


def main() -> None:
    out = Path(__file__).resolve().parent / "results"
    completion = json.loads((out / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["completed_cases"] == 90
    manifest = json.loads((out / "manifest.json").read_text())
    assert run.digest(run.HERE / "run_comparison.py") == manifest["runner_sha256"]
    config = json.loads((run.ROOT / "results/comprehensive/config.json").read_text())
    expected = set()
    cases = {}
    for task in ("tabular", "fashion-standard", "fashion-lr_restart"):
        prefix = "tabular" if task == "tabular" else "fashion"
        for seed, rho in itertools.product(map(int, config[prefix + "_seeds"].split(",")),
                                          map(float, config[prefix + "_rhos"].split(","))):
            expected.add((task, seed, rho))
    for path in out.glob("*_seed*_rho*.json"):
        case = json.loads(path.read_text())
        r = case["comparison"]
        key = (r["task"], r["seed"], r["rho"])
        assert key not in cases
        cases[key] = case
        q = np.asarray(case["lr_schedule"])
        eta = np.asarray(case["eta"])
        rho, floor = r["rho"], case["floor"]
        assert abs(q.sum() - rho) < 1e-12 and q.min() >= floor
        # All current cases have inactive floors: this is an independent formula.
        np.testing.assert_allclose(q, rho * eta / eta.sum(), rtol=1e-14, atol=1e-16)
        if r["task"] != "fashion-lr_restart":
            assert case["lr_equals_uniform"]
            assert case["lr_metrics"] == case["uniform_metrics"]
            assert r["lr_vs_uniform_percent"] == 0
        pb, lr, uniform = case["archived_pb_row"], case["lr_metrics"], case["uniform_metrics"]
        np.testing.assert_allclose(r["pb_vs_lr_percent"], 100 * (lr["terminal_mean"] - float(pb["terminal_mean"])) / lr["terminal_mean"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(r["pb_minus_lr_accuracy_pp"], 100 * (float(pb["test_acc"]) - lr["test_acc"]), rtol=0, atol=1e-14)
        assert max(case["uniform_reproduction_checks"].values()) < 1e-10
    assert set(cases) == expected
    compact = []
    summary = run.read_csv(out / "summary.csv")
    assert len(summary) == 9
    # Critical value is tabulated Student t(9), independently of the package CDF.
    critical = 2.262157162798205
    for s in summary:
        task, rho = s["task"], float(s["rho"])
        group = [c for (t, _, r), c in cases.items() if t == task and r == rho]
        assert len(group) == 10
        for name in ("pb_vs_lr_percent", "pb_minus_lr_error", "pb_minus_lr_accuracy_pp", "lr_vs_uniform_percent", "pb_vs_uniform_percent"):
            values = np.array([c["comparison"][name] for c in group])
            mean = values.mean()
            half = critical * values.std(ddof=1) / np.sqrt(10)
            np.testing.assert_allclose([float(s[name + "_mean"]), float(s[name + "_low"]), float(s[name + "_high"])],
                                       [mean, mean-half, mean+half], rtol=1e-9, atol=1e-10)
        compact.append({"task": task, "rho": rho,
            "LR_vs_uniform_percent": s["lr_vs_uniform_percent_mean"],
            "PB_vs_uniform_percent": s["pb_vs_uniform_percent_mean"],
            "PB_vs_LR_percent": s["pb_vs_lr_percent_mean"],
            "PB_vs_LR_CI95_low": s["pb_vs_lr_percent_low"],
            "PB_vs_LR_CI95_high": s["pb_vs_lr_percent_high"],
            "PB_minus_LR_accuracy_pp": s["pb_minus_lr_accuracy_pp_mean"],
            "accuracy_CI95_low": s["pb_minus_lr_accuracy_pp_low"],
            "accuracy_CI95_high": s["pb_minus_lr_accuracy_pp_high"]})
    run.write_csv(out / "comparison_table.csv", compact)
    run.write_json(out / "verification.json", {"status": "verified", "cases": 90, "groups": 9,
        "source_and_reference_checks": manifest,
        "independent_checks": ["case completeness", "closed-form LR allocation", "budget and floor",
            "constant LR equals uniform", "seed-level differences", "t(9) confidence intervals"],
        "max_uniform_reproduction_error": max(c["comparison"]["uniform_reproduction_error"] for c in cases.values()),
        "max_budget_error": max(c["budget_error"] for c in cases.values()),
        "summary_sha256": run.digest(out / "summary.csv"), "table_sha256": run.digest(out / "comparison_table.csv")})
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
