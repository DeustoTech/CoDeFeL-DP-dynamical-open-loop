"""Resume the frozen comparison runner using independent case workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import run_comparison as experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = experiment.HERE / "results"
    manifest = json.loads((out / "manifest.json").read_text())
    assert experiment.digest(experiment.HERE / "run_comparison.py") == manifest["runner_sha256"]
    root = experiment.ROOT
    config_path = root / "results/comprehensive/config.json"
    raw_path = root / "results/comprehensive/raw_results.csv"
    assert experiment.digest(config_path) == manifest["archive_config_sha256"]
    assert experiment.digest(raw_path) == manifest["archive_raw_sha256"]
    config = json.loads(config_path.read_text())
    for name, expected in manifest["source_sha256"].items():
        assert experiment.digest(root / "code" / name) == expected
    archived = {(r["task"], int(r["seed"]), float(r["rho"]), r["method_family"]): r
                for r in experiment.read_csv(raw_path)}
    experiment.write_json(out / "parallel_execution.json", {
        "workers": args.workers, "driver_sha256": experiment.digest(experiment.HERE / "run_parallel.py"),
        "timing_scope": "evaluation only, concurrent case times are not solver speed benchmarks"})
    experiment.fashion.load_preprocessed_fashion_mnist(str((root / "data/fashion_mnist").resolve()))
    cases, pending = [], []
    for task in ("fashion-lr_restart", "tabular", "fashion-standard"):
        prefix = "tabular" if task == "tabular" else "fashion"
        for rho in map(float, config[prefix + "_rhos"].split(",")):
            for seed in map(int, config[prefix + "_seeds"].split(",")):
                path = out / f"{task}_seed{seed}_rho{rho:g}.json"
                if path.exists():
                    cases.append(json.loads(path.read_text()))
                else:
                    pending.append((task, seed, rho, path))
    print(f"RESUME {len(cases)}/90 complete; {len(pending)} pending; workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(experiment.run_case, task, seed, rho, config, archived): (task, seed, rho, path)
                   for task, seed, rho, path in pending}
        for future in as_completed(futures):
            task, seed, rho, path = futures[future]
            result = future.result()
            experiment.write_json(path, result)
            cases.append(result)
            experiment.summarize(cases, out)
            r = result["comparison"]
            print(f"DONE {len(cases)}/90 {task} seed={seed} rho={rho} "
                  f"LR gain={r['lr_vs_uniform_percent']:.7f}% PB gain={r['pb_vs_uniform_percent']:.7f}% "
                  f"PB-vs-LR={r['pb_vs_lr_percent']:.7f}% eval={r['elapsed_seconds']:.1f}s", flush=True)
    experiment.write_json(out / "completion.json", {"completed_cases": len(cases), "expected_cases": 90,
        "status": "complete" if len(cases) == 90 else "incomplete", "all_uniform_reproductions_passed": True,
        "max_uniform_scaled_error": max(c["comparison"]["uniform_reproduction_error"] for c in cases),
        "max_budget_error": max(c["budget_error"] for c in cases)})


if __name__ == "__main__":
    main()
