"""Report all locked test cases, including failures and unfavorable outcomes."""
import csv
import json
from pathlib import Path
import statistics

import benchmark as b
np = b.np
OUT = b.HERE / "results"
FIG = b.HERE / "figures"
TITLES = {"tabular": "Tabular", "fashion_constant": "Fashion--constant",
          "fashion_restart": "Fashion--restart", "nonlinear_head": "Nonlinear head"}
STYLES = {"PB-AMD": "blue,solid,mark=*", "PGD": "black,dashed,mark=square*"}


def read(path):
    return json.loads(Path(path).read_text())


def first_hit(result, tolerance):
    return next((p for p in result["trace"] if p["relative_gap"] <= tolerance), None)


def load_all():
    assert read(OUT / "protocol.json") == b.protocol()
    results = {}
    for task in b.TASKS:
        selection = read(OUT / "selection" / f"{task}.json")
        for seed in b.seeds(task, False):
            for method in b.METHODS:
                gamma = selection[method]["selected_gamma"]
                value = read(b.case_path(OUT, "test", task, seed, method, gamma))
                assert value["source_sha256"] == b.sources()
                assert value["selection_sha256"] == b.digest(OUT / "selection" / f"{task}.json")
                results[task, seed, method] = value
    assert len(results) == 80
    return results


def summarize(results):
    rows = []
    for task in b.TASKS:
        for tolerance in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            hits = {method: [first_hit(results[task, seed, method], tolerance)
                             for seed in b.seeds(task, False)] for method in b.METHODS}
            paired = [(pb, pg) for pb, pg in zip(hits["PB-AMD"], hits["PGD"])
                      if pb is not None and pg is not None]
            row = {"task": task, "relative_gap_tolerance": tolerance,
                   "PB_successes": sum(h is not None for h in hits["PB-AMD"]),
                   "PGD_successes": sum(h is not None for h in hits["PGD"]),
                   "both_successes": len(paired)}
            for method, key in (("PB-AMD", "PB"), ("PGD", "PGD")):
                successful = [h for h in hits[method] if h is not None]
                for metric in ("gradient_evaluations", "seconds"):
                    row[f"{key}_median_{metric}_successful"] = (
                        statistics.median(h[metric] for h in successful) if successful else None)
            for metric in ("gradient_evaluations", "seconds"):
                ratios = [pg[metric] / pb[metric] for pb, pg in paired]
                row[f"median_PGD_over_PB_{metric}_paired"] = statistics.median(ratios) if ratios else None
                row[f"PB_faster_{metric}_paired"] = sum(r > 1 for r in ratios)
            rows.append(row)
    end = []
    for task in b.TASKS:
        for method in b.METHODS:
            trajectories = [results[task, seed, method]["trace"] for seed in b.seeds(task, False)]
            final = [trace[-1] for trace in trajectories]
            step_seconds = [statistics.median(b_["seconds"] - a["seconds"]
                                             for a, b_ in zip(trace, trace[1:]))
                            for trace in trajectories if len(trace) > 1]
            end.append({"task": task, "method": method,
                        "median_final_relative_gap": statistics.median(r["relative_gap"] for r in final),
                        "median_final_scaled_kkt": statistics.median(r["scaled_kkt"] for r in final),
                        "median_final_relative_objective": statistics.median(r["relative_objective"] for r in final),
                        "median_total_seconds": statistics.median(r["seconds"] for r in final),
                        "median_seconds_per_update_and_adjoint": statistics.median(step_seconds) if step_seconds else None})
    b.save(OUT / "summary.json", {"thresholds": rows, "final": end,
        "instances": 40, "paired_solves": 80,
        "interpretation": "Ratios above one favor PB-AMD. Ratios are conditional on both methods reaching the tolerance; successes are reported separately."})
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, end


PREAMBLE = r"""\documentclass[tikz,border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\pgfplotsset{common/.style={width=7.25cm,height=4.35cm,
 tick label style={font=\fontsize{6}{7}\selectfont},
 title style={font=\scriptsize},label style={font=\scriptsize},
 legend style={font=\scriptsize,draw=none,fill=white},
 grid=major,major grid style={gray!25},axis line style={line width=.4pt},
 tick style={line width=.4pt},scaled ticks=false}}
\begin{document}
\begin{tikzpicture}
"""
END = "\\end{tikzpicture}\n\\end{document}\n"


def coords(x, y):
    return " ".join(f"({float(a):.9g},{float(c):.9g})" for a, c in zip(x, y))


def curves(results):
    all_gaps = [max(p["relative_gap"], 1e-16) for r in results.values() for p in r["trace"]]
    ymin = 10. ** np.floor(np.log10(min(all_gaps)))
    ymax = 10. ** np.ceil(np.log10(max(all_gaps)))
    source = PREAMBLE + rf"\begin{{groupplot}}[common,ymode=log,ymin={ymin:.9g},ymax={ymax:.9g}," + r"""
 group style={group size=2 by 4,horizontal sep=1.5cm,vertical sep=1.45cm}]
"""
    for task in b.TASKS:
        for metric in ("gradient_evaluations", "seconds"):
            limits = [r["trace"][-1][metric] for (t, _, _), r in results.items() if t == task]
            xmax = max(limits)
            grid = np.arange(1, xmax + 1) if metric == "gradient_evaluations" else np.linspace(0, xmax, 101)
            title = TITLES[task]
            label = "Adjoint evaluations" if metric == "gradient_evaluations" else "Solver time (s)"
            source += rf"\nextgroupplot[title={{{title}}},xlabel={{{label}}},ylabel={{Relative stationarity gap}},xmin=0,xmax={xmax:.9g}]" + "\n"
            for method in b.METHODS:
                traces = [results[task, seed, method]["trace"] for seed in b.seeds(task, False)]
                values = []
                for trace in traces:
                    x = np.array([p[metric] for p in trace])
                    y = np.array([p["relative_gap"] for p in trace])
                    index = np.searchsorted(x, grid, side="right") - 1
                    vals = y[np.maximum(index, 0)].copy()
                    # q0 is available initially; the first adjoint diagnoses it.
                    values.append(np.maximum(vals, 1e-16))
                lower, median, upper = np.quantile(values, [.25, .5, .75], axis=0)
                name = "pb" if method == "PB-AMD" else "pg"
                color = "blue" if method == "PB-AMD" else "black"
                source += rf"\addplot[name path={name}lo,draw=none,forget plot] coordinates {{{coords(grid, lower)}}};" + "\n"
                source += rf"\addplot[name path={name}hi,draw=none,forget plot] coordinates {{{coords(grid, upper)}}};" + "\n"
                source += rf"\addplot[{color}!10,forget plot] fill between[of={name}lo and {name}hi];" + "\n"
                source += rf"\addplot[{STYLES[method]},line width=.85pt,mark size=1pt,mark repeat=12] coordinates {{{coords(grid, median)}}};" + "\n"
                if task == "tabular" and metric == "gradient_evaluations":
                    source += rf"\addlegendentry{{{method}}}" + "\n"
    source += "\\end{groupplot}\n" + END
    (FIG / "solver_convergence.tex").write_text(source)


def profiles(results, tolerance=1e-4):
    source = PREAMBLE + r"""\begin{groupplot}[common,height=5cm,
 group style={group size=2 by 1,horizontal sep=1.45cm},
 xmode=log,log basis x=2,xmin=1,xmax=4,ymin=0,ymax=1,
 xtick={1,2,4},xticklabels={1,2,4},ytick={0,.25,.5,.75,1},
 ylabel={Fraction of all 40 instances}]
"""
    data = {}
    for metric in ("gradient_evaluations", "seconds"):
        ratios = {m: [] for m in b.METHODS}
        for task in b.TASKS:
            for seed in b.seeds(task, False):
                hits = {m: first_hit(results[task, seed, m], tolerance) for m in b.METHODS}
                good = [h[metric] for h in hits.values() if h is not None]
                best = min(good) if good else None
                for method, h in hits.items():
                    ratios[method].append(h[metric] / best if h is not None else float("inf"))
        title = "(a) Adjoint evaluations" if metric == "gradient_evaluations" else "(b) Solver time"
        source += rf"\nextgroupplot[title={{{title}}},xlabel={{Factor relative to the faster solver}},legend pos=south east]" + "\n"
        for method in b.METHODS:
            finite = sorted(set([1., 4.] + [r for r in ratios[method] if np.isfinite(r) and r <= 4.]))
            y = [sum(r <= x + 1e-12 for r in ratios[method]) / 40 for x in finite]
            source += rf"\addplot[const plot,{STYLES[method]},mark size=1.1pt,line width=.9pt] coordinates {{{coords(finite, y)}}};" + "\n"
            source += rf"\addlegendentry{{{method}}}" + "\n"
        data[metric] = {m: [r if np.isfinite(r) else None for r in ratios[m]] for m in b.METHODS}
    source += "\\end{groupplot}\n" + END
    (FIG / "solver_performance_profiles.tex").write_text(source)
    b.save(OUT / "performance_profile_data.json", {"tolerance": tolerance, "data": data,
        "failure_encoding": "null; included in denominator but never counted as success"})


def number(value):
    return "--" if value is None else f"{value:.2f}"


def report(rows, final):
    source = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,graphicx,hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\begin{document}
\begin{center}\Large Matched-Oracle Comparison of PB-AMD and PGD\end{center}
\section*{Design}
This experiment compares Privacy-Budget Adjoint Mirror Descent (PB-AMD)
with projected gradient descent (PGD), isolating single-solve convergence
from candidate-grid workload. Both methods use the same fixed sample-average
approximation (SAA) objective, Gaussian paths,
uniform initialization, and exact-budget feasible set. Four existing tasks each
have three public calibration problems and ten independent test problems,
all at privacy budget $0.5$. Each method receives four candidate stepsizes and
the same 30-update calibration budget per candidate and problem. Calibration
minimizes the geometric mean final relative stationarity gap; test outcomes
are not used to select stepsizes.

On $Q=\{q\in\mathbb R^T:q_t\ge10^{-5},\ \sum_t q_t=\rho\}$,
the shared diagnostic for the fixed SAA objective $F_M$ is
\[
 r(q)=\frac{\max_{z\in Q}\langle\nabla F_M(q),q-z\rangle}{F_M(q^0)}.
\]
It is a variational stationarity measure, not a global-optimality gap.
PB-AMD uses $\alpha=\gamma\rho/F_M(q^0)$ and PGD uses
$\alpha=\gamma\rho^2/(T F_M(q^0))$, matching their infinitesimal
initial displacements. Both select $\gamma$ from $\{0.03,0.1,0.3,1\}$.
Each test solve stops at $r\le10^{-6}$ or the common task-specific cap:
180, 100, 100, and 60 updates respectively. One initial adjoint is counted.
The controlled fixed-step head comparison is distinct from the Armijo runs
used for the manuscript's nonlinear-head utility results.

\section*{Time and Evaluations to Common Accuracy}
Ratios greater than one favor PB-AMD. Ratios are medians of paired
PGD/PB-AMD ratios on instances where both methods reach the threshold;
success counts always refer to all ten instances. A missing value is a
failure to reach that threshold within the prescribed budget, not an
uncomputed experiment.
\begin{center}\small
\begin{tabular}{llccrr}
\toprule
Task & Target $r$ & PB success & PGD success & Adjoint ratio & Time ratio\\
\midrule
"""
    for row in rows:
        if row["relative_gap_tolerance"] not in (1e-3, 1e-4):
            continue
        exponent = -3 if row["relative_gap_tolerance"] == 1e-3 else -4
        source += (f"{TITLES[row['task']]} & $10^{{{exponent}}}$ & {row['PB_successes']}/10 & "
                   f"{row['PGD_successes']}/10 & {number(row['median_PGD_over_PB_gradient_evaluations_paired'])} & "
                   f"{number(row['median_PGD_over_PB_seconds_paired'])} \\\\\n")
    source += r"""\bottomrule\end{tabular}\end{center}
At the primary target $r\le10^{-4}$, PB-AMD reduces adjoint evaluations in
the tabular and restart tasks, while the constant-rate task is comparable.
The fixed-step head ratios are conditional on seven joint successes;
the two methods reach the target on seven and eight instances, respectively.
\begin{figure}[h!]
\centering\includegraphics[width=\textwidth]{solver_performance_profiles.pdf}
\caption{Performance profiles at $r\le10^{-4}$. All 40 instances are retained;
failures never contribute to a curve. The horizontal axis is the factor
relative to the faster successful method on each instance.}
\end{figure}
\clearpage
\section*{Convergence Across All Test Instances}
\begin{figure}[h!]
\centering\includegraphics[width=.98\textwidth,height=.86\textheight,keepaspectratio]{solver_convergence.pdf}
\caption{Median relative stationarity gap and interquartile range over ten
test problems per task. Each solve retains its final available point after termination.
Both methods use the same diagnostic; their native mirror and projected-gradient
residuals are not compared against each other.}
\end{figure}
\clearpage
\section*{Reproducibility and Interpretation}
"""
    source += r"""\begin{center}\begin{tabular}{lrrr}
\toprule Task & Paths & PB $\gamma$ & PGD $\gamma$\\\midrule
"""
    for task in b.TASKS:
        selection = read(OUT / "selection" / f"{task}.json")
        source += f"{TITLES[task]} & {b.PATHS[task]} & {selection['PB-AMD']['selected_gamma']:g} & {selection['PGD']['selected_gamma']:g} \\\\\n"
    source += r"""\bottomrule\end{tabular}\end{center}
Solvers are timed serially, with order alternating across seeds. Neural
adjoints use the same four-worker path decomposition for both methods, with
one numerical-library thread per worker. Tabular adjoints are vectorized in
one thread. Timing includes updates, gradients, and identical diagnostics;
data preparation, warm-up, and file output are excluded. There is no line
search and no private-outcome evaluation. These timings are not substituted
for the earlier, different pipeline benchmark.

The solver source, protocol, all calibration candidates, selected parameters,
test trajectories, full iterates and gradients, hashes, and verification report
are retained alongside this document. Failed or unfavorable cases are not removed.
The comparison supports only empirical conclusions for these tasks and this
calibration protocol; it does not establish a different asymptotic convergence order.

\section*{Final Diagnostics}
The common Karush--Kuhn--Tucker (KKT) residual checks gradient equalization
on free coordinates and the one-sided floor conditions. Its scaled value
below multiplies that residual by $\rho/F_M(q^0)$.
\begin{center}\small\begin{tabular}{llrrr}
\toprule Task & Method & Median $r$ & Scaled KKT & $F_M/F_M(q^0)$\\\midrule
"""
    for row in final:
        source += (f"{TITLES[row['task']]} & {row['method']} & "
                   f"{row['median_final_relative_gap']:.2e} & {row['median_final_scaled_kkt']:.2e} & "
                   f"{row['median_final_relative_objective']:.6f} \\\\\n")
    source += r"""\bottomrule\end{tabular}\end{center}
\section*{Geometric Interpretation}
For the linear--quadratic allocation objective $F(q)=\sum_t a_t/q_t$,
an interior optimum satisfies $a_t/(q_t^*)^2=\lambda$. Consequently,
\[
 \nabla^2F(q^*)=2\lambda\,\operatorname{diag}(1/q_t^*)
             =2\lambda\,\nabla^2\!\left(\sum_t q_t\log q_t\right)\!(q^*).
\]
Thus entropy matches the local allocation curvature in this model, whereas
Euclidean updates encounter coordinate-dependent curvature when the allocation
is nonuniform. The nonlinear experiments test whether this geometric motivation
translates into practical convergence benefits; the identity itself does not
assert a universal speed advantage for nonlinear training.
\end{document}
"""
    (FIG / "solver_comparison_report.tex").write_text(source)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    results = load_all()
    rows, final = summarize(results)
    curves(results)
    profiles(results)
    report(rows, final)
    print(json.dumps([r for r in rows if r["relative_gap_tolerance"] == 1e-4], indent=2))


if __name__ == "__main__":
    main()
