"""Aggregate per-network predictions at the independent problem-seed level."""
import csv
import json

import run_study as s
import numpy as np
from experiment_utils import student_t_quantile, student_t_two_sided_pvalue, holm_adjusted_pvalues


def interval(a):
    a = np.asarray(a, dtype=float)
    assert len(a) == 10
    mean = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    half = student_t_quantile(.975, len(a)-1) * se
    return {"mean": mean, "ci_low": mean-half, "ci_high": mean+half,
            "pvalue": student_t_two_sided_pvalue(mean/se, len(a)-1) if se else float(mean == 0.)}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit = json.loads((s.RESULTS / "verification.json").read_text())
    assert audit["status"] == "passed"
    for name, expected in audit["files_sha256"].items():
        assert s.sha(s.ROOT / name) == expected
    rows = []
    for case in s.cases():
        data = json.loads((s.RESULTS / "cases" / (s.stem(case)+".json")).read_text())
        scenario, seed, rho = case
        for method in s.METHODS + ["No noise"]:
            metrics = data["no_noise"] if method == "No noise" else data["metrics"][method]
            rows.append({"scenario": scenario, "seed": seed, "rho": rho, "method": method,
                "accuracy_percent": float(100*np.mean(metrics["accuracy"])),
                "cross_entropy": float(np.mean(metrics["cross_entropy"])),
                "regularized_loss": float(np.mean(metrics["regularized_loss"]))})
    summary, differences = [], []
    for scenario in ("standard", "lr_restart"):
        for rho in (.25, .5, 1.):
            group = [r for r in rows if r["scenario"] == scenario and r["rho"] == rho]
            values = {m: {metric: [r[metric] for r in sorted(group, key=lambda r:r["seed"]) if r["method"]==m]
                          for metric in ("accuracy_percent", "cross_entropy")}
                      for m in s.METHODS + ["No noise"]}
            for method, metrics in values.items():
                for metric, a in metrics.items():
                    summary.append({"scenario": scenario, "rho": rho, "method": method,
                                    "metric": metric, "n_seeds": 10, **interval(a)})
            for comparator in ("Uniform", "LR", "No noise"):
                for metric in ("accuracy_percent", "cross_entropy"):
                    diff = np.array(values["PB-AMD"][metric]) - values[comparator][metric]
                    differences.append({"scenario": scenario, "rho": rho, "comparator": comparator,
                        "metric": metric, "n_seeds": 10, **interval(diff), "holm_pvalue": None})
    primary = [r for r in differences if r["comparator"] == "Uniform"]
    for row, p in zip(primary, holm_adjusted_pvalues([r["pvalue"] for r in primary])):
        row["holm_pvalue"] = p
    write_csv(s.RESULTS / "seed_metrics.csv", rows)
    write_csv(s.RESULTS / "summary.csv", summary)
    write_csv(s.RESULTS / "paired_differences.csv", differences)
    report = {"summary": summary, "paired_differences": differences,
        "inference": "Means over 256 paths within seed, then ten-seed t intervals; differences PB minus comparator",
        "multiplicity": "Holm jointly over twelve PB-vs-Uniform task-budget-endpoint tests",
        "verification_sha256": s.sha(s.RESULTS / "verification.json")}
    s.save(s.RESULTS / "summary.json", report)
    for row in primary:
        print(row)


if __name__ == "__main__":
    main()
