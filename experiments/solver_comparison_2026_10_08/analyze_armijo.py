"""Append the fully retained, secondary Armijo results to the primary report."""
import statistics
import analyze as primary
import armijo as a
b, np = a.b, a.np


def main():
    assert primary.read(a.OUT / "verification.json")["status"] == "passed"
    results = {(seed, method): primary.read(a.OUT / f"{seed}_{method}.json")
               for seed in b.seeds("nonlinear_head", False) for method in b.METHODS}
    assert all(r["source_sha256"] == a.sources() for r in results.values())
    rows = []
    for tolerance in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        hits = {m: [primary.first_hit(results[seed, m], tolerance)
                    for seed in b.seeds("nonlinear_head", False)] for m in b.METHODS}
        pairs = [(x, y) for x, y in zip(hits["PB-AMD"], hits["PGD"]) if x is not None and y is not None]
        row = {"tolerance": tolerance, "PB_successes": sum(x is not None for x in hits["PB-AMD"]),
               "PGD_successes": sum(x is not None for x in hits["PGD"]), "both_successes": len(pairs)}
        for metric in ("gradient_evaluations", "forward_only_evaluations", "seconds"):
            for m, key in (("PB-AMD", "PB"), ("PGD", "PGD")):
                values = [h[metric] for h in hits[m] if h is not None]
                row[f"{key}_median_{metric}"] = statistics.median(values) if values else None
            ratios = [y[metric] / x[metric] for x, y in pairs if x[metric] > 0]
            row[f"median_PGD_over_PB_{metric}_paired"] = statistics.median(ratios) if ratios else None
        rows.append(row)
    b.save(a.OUT / "summary.json", {"status": "secondary", "instances": 10, "comparisons": rows})
    gaps = [max(p["relative_gap"], 1e-16) for r in results.values() for p in r["trace"]]
    ymin, ymax = 10. ** np.floor(np.log10(min(gaps))), 10. ** np.ceil(np.log10(max(gaps)))
    source = primary.PREAMBLE + rf"\begin{{groupplot}}[common,ymode=log,ymin={ymin:.9g},ymax={ymax:.9g}," + r"""
 group style={group size=2 by 1,horizontal sep=1.45cm}]
"""
    for metric in ("gradient_evaluations", "seconds"):
        xmax = max(r["trace"][-1][metric] for r in results.values())
        grid = np.arange(1, xmax + 1) if metric == "gradient_evaluations" else np.linspace(0, xmax, 101)
        title = "(a) Matched Armijo: adjoints" if metric == "gradient_evaluations" else "(b) Matched Armijo: time"
        label = "Adjoint evaluations" if metric == "gradient_evaluations" else "Solver time (s)"
        source += rf"\nextgroupplot[title={{{title}}},xlabel={{{label}}},ylabel={{Relative stationarity gap}},xmin=0,xmax={xmax:.9g},legend pos=north east]" + "\n"
        for method in b.METHODS:
            values = []
            for seed in b.seeds("nonlinear_head", False):
                trace = results[seed, method]["trace"]
                x, y = np.array([p[metric] for p in trace]), np.array([p["relative_gap"] for p in trace])
                index = np.maximum(np.searchsorted(x, grid, side="right") - 1, 0)
                values.append(np.maximum(y[index], 1e-16))
            low, med, high = np.quantile(values, [.25, .5, .75], axis=0)
            name, color = ("pb", "blue") if method == "PB-AMD" else ("pg", "black")
            source += rf"\addplot[name path={name}lo,draw=none,forget plot] coordinates {{{primary.coords(grid, low)}}};" + "\n"
            source += rf"\addplot[name path={name}hi,draw=none,forget plot] coordinates {{{primary.coords(grid, high)}}};" + "\n"
            source += rf"\addplot[{color}!10,forget plot] fill between[of={name}lo and {name}hi];" + "\n"
            source += rf"\addplot[{primary.STYLES[method]},line width=.85pt,mark size=1pt,mark repeat=10] coordinates {{{primary.coords(grid, med)}}};" + "\n"
            source += rf"\addlegendentry{{{method}}}" + "\n"
    source += "\\end{groupplot}\n" + primary.END
    (primary.FIG / "solver_armijo_head.tex").write_text(source)
    appendix = r"""\clearpage
\section*{Secondary Analysis: Matched Armijo on the Nonlinear Head}
The primary fixed-step head experiment includes instances that do not reach
the prescribed tolerance. The manuscript's head utility study uses Armijo
PB-AMD. After observing the fixed-step failure on seed 213102, this secondary
analysis was specified for both methods on \emph{all ten} unchanged head
instances. The fixed-step results above are retained without replacement.

Both methods use initial dimensionless step $0.3$, cap $2$, Armijo
coefficient $0.1$, contraction $0.5$, and growth $1.1$, with the same
geometry-matched scaling as before. Every accepted and rejected forward-only
trial is counted and timed. The 60-update cap, stationarity thresholds,
initialization, data, and fixed Gaussian paths are unchanged.

\begin{center}\small\begin{tabular}{lccrrcc}
\toprule
Target $r$ & PB success & PGD success & Adjoint ratio & Time ratio & PB $N_F$ & PGD $N_F$\\
\midrule
"""
    for row in rows:
        if row["tolerance"] not in (1e-3, 1e-4, 1e-5, 1e-6):
            continue
        exponent = int(round(np.log10(row["tolerance"])))
        appendix += (f"$10^{{{exponent}}}$ & {row['PB_successes']}/10 & {row['PGD_successes']}/10 & "
                     f"{primary.number(row['median_PGD_over_PB_gradient_evaluations_paired'])} & "
                     f"{primary.number(row['median_PGD_over_PB_seconds_paired'])} & "
                     f"{primary.number(row['PB_median_forward_only_evaluations'])} & "
                     f"{primary.number(row['PGD_median_forward_only_evaluations'])} \\\\\n")
    appendix += r"""\bottomrule\end{tabular}\end{center}
Ratios are paired PGD/PB-AMD medians on joint successes. $N_F$ is the
median number of forward-only line-search calls at the indicated tolerance;
these calls are additional to the adjoints and are included in solver time.

\begin{figure}[h!]
\centering\includegraphics[width=\textwidth]{solver_armijo_head.pdf}
\caption{Secondary symmetric-Armijo comparison on the same ten nonlinear-head
instances. Curves and shading show medians and interquartile ranges.}
\end{figure}
"""
    path = primary.FIG / "solver_comparison_report.tex"
    report = path.read_text()
    assert "Secondary Analysis: Matched Armijo" not in report, "regenerate the primary report first"
    path.write_text(report.replace(r"\end{document}", appendix + "\n" + r"\end{document}"))
    print(rows)


if __name__ == "__main__":
    main()
