"""Export the two-panel predictive-performance figure from audited results."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_study as s


def main():
    audit = json.loads((s.RESULTS / "verification.json").read_text())
    report = json.loads((s.RESULTS / "summary.json").read_text())
    assert audit["status"] == "passed"
    assert report["verification_sha256"] == s.sha(s.RESULTS / "verification.json")
    for path, digest in audit["files_sha256"].items():
        assert s.sha(s.ROOT / path) == digest
    rows = {(r["method"], r["metric"], r["rho"]): r for r in report["summary"]
            if r["scenario"] == "lr_restart"}
    text = r"""\documentclass[tikz,border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{pbblue}{RGB}{0,32,235}
\definecolor{lrmagenta}{RGB}{165,42,120}
\definecolor{unifred}{RGB}{190,55,25}
\pgfplotsset{paper axis/.style={width=7.3cm,height=4.3cm,
 axis line style={line width=.45pt},tick style={line width=.4pt},
 tick label style={font=\fontsize{6}{6.5}\selectfont},
 title style={font=\scriptsize},label style={font=\scriptsize},
 grid=major,major grid style={gray!30,line width=.3pt},scaled ticks=false,
 legend style={draw=none,font=\scriptsize,fill=white},
 xmin=.19,xmax=1.06,xtick={.25,.5,1},xticklabels={0.25,0.5,1},
 xlabel={Certified budget $\rho_{\max}$}}}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[paper axis,group style={group size=2 by 1,horizontal sep=1.5cm}]
"""
    # Open LR markers and dashed segments reveal both nearly coincident curves.
    methods = [("No noise", "black,densely dotted,mark=diamond*,mark size=1.4pt"),
               ("Uniform", "unifred,dashdotted,mark=square*,mark size=1.4pt"),
               ("PB-AMD", "pbblue,solid,mark=*,mark size=1pt"),
               ("LR", "lrmagenta,densely dotted,mark=triangle,mark size=2.8pt,mark options={fill=none}")]
    for panel, (metric, title, ylabel) in enumerate([
            ("accuracy_percent", "(a) Test accuracy", "Accuracy (\\%)"),
            ("cross_entropy", "(b) Test cross-entropy", "Cross-entropy")]):
        legend = ",legend to name=networklegend,legend columns=4,legend style={/tikz/every even column/.append style={column sep=.2cm}}" if panel == 0 else ""
        text += rf"\nextgroupplot[title={{{title}}},ylabel={{{ylabel}}}{legend}]" + "\n"
        for method, style in methods:
            points = []
            for rho in (.25, .5, 1.):
                r = rows[method, metric, rho]
                err = (r["ci_high"]-r["ci_low"])/2
                points.append(f"({rho:g},{r['mean']:.12g}) +- (0,{err:.12g})")
            text += rf"\addplot[{style},line width=.85pt,error bars/.cd,y dir=both,y explicit,error bar style={{line width=.4pt}},error mark options={{rotate=90,mark size=1.6pt,line width=.4pt}}] coordinates {{" + " ".join(points) + "};\n"
            if panel == 0:
                label = "No-noise reference" if method == "No noise" else method
                text += rf"\addlegendentry{{{label}}}" + "\n"
    text += r"""\end{groupplot}
\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.28cm)$)
 {\pgfplotslegendfromname{networklegend}};
\end{tikzpicture}
\end{document}
"""
    out = HERE / "figure_network_performance.tex"
    out.write_text(text)
    s.save(HERE / "figure_audit.json", {"status": "passed", "scenario": "lr_restart",
        "seeds_per_budget": 10, "paths_per_seed": 256, "budgets": [.25, .5, 1.],
        "methods": [m[0] for m in methods], "metrics": ["accuracy_percent", "cross_entropy"],
        "summary_sha256": s.sha(s.RESULTS / "summary.json"), "tex_sha256": s.sha(out)})
    print(out)


if __name__ == "__main__":
    main()
