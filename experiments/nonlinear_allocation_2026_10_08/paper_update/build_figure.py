"""Export an inline-data scientific figure from authoritative evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
METHODS = ("PB-AMD", "Linearized", "Decay", "Refined-geometric", "LR")
STYLES = ("pb curve", "linear curve", "decay curve", "geo curve", "lr curve")
LABELS = ("PB-AMD", "Linearized", "Decay", "Geometric", "Uniform / LR")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def coordinates(values):
    return " ".join(f"({x:.10g},{y:.10g})" for x, y in values)


def load_case(directory, seed, rho):
    name = f"train_fast384_seed{seed}_rho{rho:g}.json"
    parent_path = STUDY / directory / name
    evaluation_path = parent_path.parent / "independent_evaluation" / name
    refined_path = parent_path.parent / "refined_geometric" / name
    parent, evaluation, refined = map(read, (parent_path, evaluation_path, refined_path))
    assert evaluation["case_sha256"] == sha(parent_path)
    assert evaluation["refined_sha256"] == sha(refined_path)
    schedules = parent["schedules"] | {"Refined-geometric": refined["schedule"]}
    for q in schedules.values():
        assert len(q) == 120 and min(q) >= 1e-5 - 1e-14
        assert abs(sum(q) - rho) < 1e-12
    for method, result in evaluation["private"].items():
        assert len(result["path_errors"]) == 512
        assert abs(statistics.mean(result["path_errors"]) - result["error"]) < 1e-10
    gains = {method: 100 * (1 - evaluation["private"]["PB-AMD"]["error"]
                            / evaluation["private"][method]["error"])
             for method in ("LR", "Decay", "Linearized", "Refined-geometric")}
    return {"seed": seed, "rho": rho, "gains": gains, "schedules": schedules,
            "private": evaluation["private"],
            "evaluation_sha256": sha(evaluation_path), "case_sha256": sha(parent_path)}


def schedule_plot(case, legend=False):
    output = []
    for method, style, label in zip(METHODS, STYLES, LABELS, strict=True):
        data = coordinates([(i, 120 * value / case["rho"])
                            for i, value in enumerate(case["schedules"][method])])
        output.append(rf"\addplot[{style}] coordinates {{{data}}};")
        if legend:
            output.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(output)


def main():
    summary = read(STUDY / "combined_summary.json")
    assert read(STUDY / "final_audit.json")["status"] == "passed"
    cases = [load_case("confirmation", seed, .5) for seed in range(96001, 96011)]
    cases += [load_case("budget_extension", seed, .25) for seed in range(106001, 106011)]
    # Code-only distribution: audit the regenerated summary against cases.
    # The local manuscript-copy check is omitted because no manuscript is shipped.
    for row in summary["comparisons"]:
        group = [c for c in cases if c["rho"] == row["rho"]]
        gains = [c["gains"][row["baseline"]] for c in group]
        assert abs(statistics.mean(gains) - row["error_reduction_percent"]) < 1e-11
        assert sum(g > 0 for g in gains) == row["wins"]

    a = cases[0]
    b = next(c for c in cases if c["seed"] == 106006)
    diagnostics = read(STUDY / "confirmation" / "independent_mechanism_diagnostics.json")
    mechanism = next(c for c in diagnostics["cases"] if "seed96001_" in c["case"])
    assert mechanism["case_sha256"] == a["case_sha256"]
    prediction = mechanism["linear_model_error_predictions"]
    nonlinear = mechanism["nonlinear_public_errors"]
    ratios = [prediction["PB-AMD"] / prediction["Linearized"],
              nonlinear["PB-AMD"] / nonlinear["Linearized"]]
    assert ratios[0] > 1 > ratios[1]
    cloud = []
    for rho, mark, color, shift in ((.5, "*", "pbblue", -.10),
                                   (.25, "square", "pgdgreen", .10)):
        group = [c for c in cases if c["rho"] == rho]
        points = coordinates([(i + 1 + shift, c["gains"]["Linearized"])
                              for i, c in enumerate(group)])
        cloud.append(rf"\addplot[only marks,{color},mark={mark},mark size=1.8pt,"
                     rf"mark options={{line width=0.7pt}}] coordinates {{{points}}};")
        cloud.append(rf"\addlegendentry{{$\rho_{{\max}}={rho:g}$}}")
    source = r"""\documentclass[tikz,border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc,patterns}
\pgfplotsset{compat=1.18}
\definecolor{pbblue}{RGB}{0,32,235}
\definecolor{pgdgreen}{RGB}{0,125,30}
\definecolor{lrmagenta}{RGB}{165,42,120}
\pgfplotsset{
 paper axis/.style={width=7.25cm,height=4.25cm,
  axis line style={line width=0.45pt},tick style={black,line width=0.4pt},
  tick label style={font=\fontsize{6}{6.5}\selectfont},
  title style={font=\scriptsize,yshift=0.2em},
  label style={font=\scriptsize},
  grid=major,major grid style={gray!35,line width=0.3pt},
  scaled ticks=false,clip mode=individual,
  legend style={draw=none,font=\scriptsize,fill=white}},
 pb curve/.style={pbblue,solid,line width=0.95pt,mark=*,mark repeat=15,mark size=1.2pt},
 linear curve/.style={black,dashed,line width=0.85pt,mark=triangle*,mark repeat=15,mark size=1.3pt},
 decay curve/.style={gray!75!black,densely dotted,line width=1pt,mark=square,mark repeat=15,mark size=1.2pt},
 geo curve/.style={pgdgreen,dashdotted,line width=0.85pt,mark=o,mark repeat=15,mark size=1.2pt},
 lr curve/.style={lrmagenta,densely dotted,line width=1pt,mark=triangle,mark repeat=15,mark phase=7,mark size=1.7pt,mark options={fill=white}}
}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[paper axis,group style={group size=2 by 2,horizontal sep=1.45cm,vertical sep=1.65cm}]
\nextgroupplot[title={(a) All training instances},xlabel={Instance index},
 ylabel={Error reduction (\%)},xmin=0.4,xmax=10.6,xtick={1,2,3,4,5,6,7,8,9,10},
 ymin=-3,ymax=36,ytick={0,10,20,30},legend pos=north east]
\addplot[gray,dashed,forget plot] coordinates {(0.4,0) (10.6,0)};
""" + "\n".join(cloud) + r"""
\node[font=\scriptsize,anchor=south west] at (axis cs:0.9,32.39862060695) {A};
\node[font=\scriptsize,anchor=south west] at (axis cs:6.1,9.20139420299) {B};
\nextgroupplot[title={(b) Instance A: $\rho_{\max}=0.5$},
 xlabel={Iteration $t$},ylabel={Charge / uniform},xmin=-2,xmax=121,
 xtick={0,40,80,120},ymin=0,ymax=5.5,ytick={0,1,2,3,4,5},
 legend to name=nonlinearschedules,legend columns=5,
 legend style={/tikz/every even column/.append style={column sep=0.35cm}}]
""" + schedule_plot(a, True) + r"""
\nextgroupplot[title={(c) Public error estimates in A},xlabel={},
 ylabel={Tracking-error ratio},xmin=-0.55,xmax=1.55,xtick={0,1},
 xticklabels={{Linearized\\prediction},{Nonlinear\\evaluation}},
 xticklabel style={align=center},ymin=0,ymax=1.65,ytick={0,0.5,1,1.5}]
\addplot[black,dashed,forget plot] coordinates {(-0.55,1) (1.55,1)};
""" + rf"""
\addplot[ybar,bar width=21pt,draw=pbblue,fill=pbblue!14,
 nodes near coords,point meta=y,
 every node near coord/.append style={{font=\scriptsize,/pgf/number format/fixed,/pgf/number format/precision=3}}]
 coordinates {{(0,{ratios[0]:.12g}) (1,{ratios[1]:.12g})}};
""" + r"""
\node[font=\scriptsize,anchor=south east,fill=white,inner sep=1.5pt]
 at (axis cs:1.5,1.025) {Equal errors};
\nextgroupplot[title={(d) Instance B: $\rho_{\max}=0.25$},
 xlabel={Iteration $t$},ylabel={Charge / uniform},xmin=-2,xmax=121,
 xtick={0,40,80,120},ymin=0,ymax=5.5,ytick={0,1,2,3,4,5}]
""" + schedule_plot(b) + r"""
\end{groupplot}
\node at ($(group c1r2.south)!0.5!(group c2r2.south)+(0,-1.0cm)$)
 {\pgfplotslegendfromname{nonlinearschedules}};
\end{tikzpicture}
\end{document}
"""
    (HERE / "figure_nonlinear_mechanism.tex").write_text(source)
    data = {"status": "passed", "scope": "Data export and regenerated-summary audit; no new simulations.",
            "script_sha256": sha(Path(__file__)), "summary_sha256": sha(STUDY / "combined_summary.json"),
            "cases": [{k: v for k, v in c.items() if k not in ("private", "schedules")}
                      for c in cases], "model_objective_ratios_A": ratios,
            "max_plotted_charge_ratio": max(120 * q / c["rho"] for c in (a, b)
                                            for m in METHODS for q in c["schedules"][m]),
            "figure_tex_sha256": sha(HERE / "figure_nonlinear_mechanism.tex")}
    assert data["max_plotted_charge_ratio"] < 5.5
    (HERE / "figure_data_audit.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
