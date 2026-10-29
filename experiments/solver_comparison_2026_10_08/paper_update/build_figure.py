"""Export the compact manuscript figure from audited, unchanged trajectories."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
import analyze as primary
import armijo as secondary
import penalty_comparison as penalty

b, np = primary.b, primary.np
PANELS = (("tabular", "(a) Tabular", False),
          ("fashion_restart", "(b) Fashion--restart", False),
          ("nonlinear_head", "(c) Nonlinear head", True))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclude_intermediate_monitoring(result):
    """Count one terminal repair diagnostic, not every plotting-only diagnostic."""
    result = dict(result)
    result["trace"] = [dict(row) for row in result["trace"]]
    costs, previous_diagnostic = {}, 0.
    for row in result["trace"]:
        key = tuple(row["q"])
        increment = row["diagnostic_seconds"] - previous_diagnostic
        assert increment >= 0
        if increment > 0:
            costs[key] = increment
        elif key not in costs:
            best = result["trace"][row["best_iteration"]]
            assert np.array_equal(row["q"], best["raw_q"])
            costs[key] = 0.
        row["gross_seconds"] = row["seconds"]
        row["seconds"] = row["seconds"] - row["diagnostic_seconds"] + costs[key]
        assert 0 < row["seconds"] <= row["gross_seconds"] + 1e-10
        previous_diagnostic = row["diagnostic_seconds"]
    assert all(x["seconds"] < y["seconds"] for x, y in zip(result["trace"], result["trace"][1:]))
    return result


def main():
    for folder, endpoints in ((b.HERE / "results", 80), (secondary.OUT, 20),
                              (penalty.OUT, 30)):
        audit = primary.read(folder / "verification.json")
        assert audit["status"] == "passed"
        assert audit["replayed_test_endpoints"] == endpoints
    fixed = primary.load_all()
    source = r"""\documentclass[tikz,border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{pbblue}{RGB}{0,32,235}
\definecolor{pgdgreen}{RGB}{0,125,30}
\definecolor{penred}{RGB}{190,55,25}
\pgfplotsset{paper axis/.style={width=5.15cm,height=3.7cm,
 axis line style={line width=.45pt},tick style={line width=.4pt},
 tick label style={font=\fontsize{6}{6.5}\selectfont},
 title style={font=\scriptsize},label style={font=\scriptsize},
 grid=major,major grid style={gray!30,line width=.3pt},scaled ticks=false,
 legend style={draw=none,font=\scriptsize,fill=white},
 xmode=log,ymode=log,ymin=1e-7,ymax=10,ytick={1e-7,1e-5,1e-3,1e-1,10},
 xlabel={Solver time (s)}}}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[paper axis,
 group style={group size=3 by 1,horizontal sep=.7cm}]
"""
    summaries, used_files = [], {}
    for panel, (task, title, adaptive) in enumerate(PANELS):
        results = {}
        selected = primary.read(penalty.OUT / "selection" / f"{task}.json")["selected"]
        for seed in b.seeds(task, False):
            for method in (*b.METHODS, "Penalty + repair"):
                if method == "Penalty + repair":
                    path = penalty.case_path("test", task, seed, selected["beta"])
                    result = primary.read(path)
                    assert result["source_sha256"] == penalty.sources()
                    result = exclude_intermediate_monitoring(result)
                elif adaptive:
                    path = secondary.OUT / f"{seed}_{method}.json"
                    result = primary.read(path)
                    assert result["source_sha256"] == secondary.sources()
                else:
                    result = fixed[task, seed, method]
                    path = b.case_path(b.HERE / "results", "test", task,
                                       seed, method, result["gamma"])
                results[seed, method] = result
                used_files[str(path.relative_to(b.ROOT))] = sha(path)
        hits = {method: [primary.first_hit(results[seed, method], 1e-4)
                         for seed in b.seeds(task, False)] for method in b.METHODS}
        pairs = [(x, y) for x, y in zip(hits["PB-AMD"], hits["PGD"])
                 if x is not None and y is not None]
        ratio = statistics.median(y["seconds"] / x["seconds"] for x, y in pairs)
        summaries.append({"task": task, "armijo": adaptive,
            "successes": {m: sum(x is not None for x in v) for m, v in hits.items()},
            "paired_time_ratio_PGD_over_PB": ratio, "joint_successes": len(pairs),
            "PB_time_saving_percent": 100 * (1 - 1 / ratio),
            "median_curve_threshold_times": {}})
        penalty_hits = [primary.first_hit(results[seed, "Penalty + repair"], 1e-4)
                        for seed in b.seeds(task, False)]
        summaries[-1]["successes"]["Penalty + repair"] = sum(x is not None for x in penalty_hits)
        pairs_penalty = [(x, y) for x, y in zip(hits["PB-AMD"], penalty_hits)
                         if x is not None and y is not None]
        summaries[-1]["paired_time_ratio_penalty_over_PB"] = (
            statistics.median(y["seconds"] / x["seconds"] for x, y in pairs_penalty)
            if pairs_penalty else None)
        summaries[-1]["penalty_joint_successes"] = len(pairs_penalty)
        xmax = max(r["trace"][-1]["seconds"] for r in results.values())
        # All ten runs of each method must have an observed value at the
        # first plotted time; do not show a future initial diagnostic early.
        xmin = max(r["trace"][0]["seconds"] for r in results.values())
        grid = np.geomspace(xmin, xmax, 161)
        extra = (r"ylabel={Relative stationarity gap},legend to name=solverlegend,"
                 r"legend columns=3,legend style={/tikz/every even column/.append style={column sep=.3cm}}"
                 if panel == 0 else "yticklabels={}")
        title += rf"\\{100 * (1 - 1 / ratio):.1f}\% less time than PGD"
        source += rf"\nextgroupplot[title={{{title}}},title style={{align=center}},xmin={xmin:.10g},xmax={xmax:.10g},{extra}]" + "\n"
        curves = []
        for method in (*b.METHODS, "Penalty + repair"):
            values = []
            for seed in b.seeds(task, False):
                trace = results[seed, method]["trace"]
                times = np.array([p["seconds"] for p in trace])
                gaps = np.array([p["relative_gap"] for p in trace])
                index = np.searchsorted(times, grid, side="right") - 1
                assert np.all(index >= 0)
                values.append(np.maximum(gaps[index], 1e-16))
            low, med, high = np.quantile(values, [.25, .5, .75], axis=0)
            assert low.min() >= 1e-7 and high.max() <= 10
            crossings = np.flatnonzero(med <= 1e-4)
            summaries[-1]["median_curve_threshold_times"][method] = (
                float(grid[crossings[0]]) if len(crossings) else None)
            name, color, style = {"PB-AMD": ("pb", "pbblue", "solid,mark=*"),
                                  "PGD": ("pg", "pgdgreen", "dashed,mark=square*"),
                                  "Penalty + repair": ("pen", "penred", "dashdotted,mark=triangle*")}[method]
            for end, vals in (("lo", low), ("hi", high)):
                source += rf"\addplot[name path={name}{end},draw=none,forget plot] coordinates {{{primary.coords(grid, vals)}}};" + "\n"
            source += rf"\addplot[{color}!10,forget plot] fill between[of={name}lo and {name}hi];" + "\n"
            curves.append(rf"\addplot[{color},{style},line width=.85pt,mark size=1pt,mark repeat=18] coordinates {{{primary.coords(grid, med)}}};" + "\n")
            if panel == 0:
                curves.append(rf"\addlegendentry{{{method}}}" + "\n")
        source += "".join(curves)
        assert (summaries[-1]["median_curve_threshold_times"]["PB-AMD"]
                < summaries[-1]["median_curve_threshold_times"]["PGD"])
        source += (rf"\addplot[black,densely dotted,line width=.5pt,forget plot] "
                   rf"coordinates {{({xmin:.10g},1e-4) ({xmax:.10g},1e-4)}};" + "\n")
        if panel == 0:
            source += (rf"\node[font=\tiny,anchor=south east,fill=white,inner sep=1pt] "
                       rf"at (axis cs:{xmax:.10g},1e-4) {{$10^{{-4}}$}};" + "\n")
    source += r"""\end{groupplot}
\node at ($(group c1r1.south)!0.5!(group c3r1.south)+(0,-1.23cm)$)
 {\pgfplotslegendfromname{solverlegend}};
\end{tikzpicture}
\end{document}
"""
    path = HERE / "figure_solver_efficiency.tex"
    path.write_text(source)
    audit = {"status": "passed", "scope": "Figure export from all ten seeds per panel; no new experiments.",
             "plotted_results_sha256": used_files, "panels": summaries,
             "source_sha256": sha(Path(__file__)), "figure_tex_sha256": sha(path),
             "timing_plan_sha256": sha(STUDY / "penalty_reporting_plan.json")}
    b.save(HERE / "figure_data_audit.json", audit)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
