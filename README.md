# Numerical experiment code for DP-SGD privacy allocation

**Umberto Biccari and Yongcun Song**  
Companion code for *When to spend privacy: optimal allocation in private learning*.

This package implements the numerical experiments reported in the article,
including the allocation-quality, nonlinear-head, prediction, and matched-solver
studies. Source provenance and checksums are recorded in `CODE_PROVENANCE.json`.

**Only source code, plotting source, protocol configuration, and documentation
are included.** Fashion-MNIST, generated synthetic samples, trained parameters,
experimental CSV/JSON results, figures, manuscript PDFs, logs and caches are
not distributed. Results and datasets are created only when the relevant
commands below are run.

## 1. What corresponds to the manuscript

| Numerical subsection / item | Experiment implementation | Main entry point |
| --- | --- | --- |
| Temporal adaptation of allocation; Tabular, Fashion-constant, Fashion-restart; allocation figure | Original core forward/adjoint models and comprehensive benchmark | `python run_experiments.py core` |
| Public-design perturbation / design-to-training transfer | `Control_Theoretic_DP_Numerics/code/proxy_transfer_study.py` | Included in `core` |
| Explicit learning-rate allocation comparison | `experiments/lr_baseline_2026_10_08/` | `python run_experiments.py lr` after `core` |
| Nonlinear-head tracking and prediction tables; nonlinear-mechanism figure | `experiments/nonlinear_allocation_2026_10_08/` and its neural-model dependency | `python run_experiments.py head` |
| Full-network accuracy / unregularized cross-entropy figure | `experiments/network_performance_2026_10_09/` | `python run_experiments.py prediction` after `core` and `lr` |
| Exact-budget / KKT checks, analytic LQ check, raw-penalty feasibility figure | Core benchmark, LQ validation and concise export | Included in `core` |
| Matched solver-convergence figure | `experiments/solver_comparison_2026_10_08/` | `python run_experiments.py solvers` |
| Five manuscript figures | Original PGFPlots sources and Python exporters | `python run_experiments.py figures` after their numerical stages |

The old core bundle alone does **not** cover the expanded manuscript. Its
controlled candidate-selection timing study is retained as supporting code;
the current three-panel solver-convergence figure uses the separate matched
solver study, not that older pipeline timing.

## 2. Directory layout

```text
DP_SGD_Numerical_Experiment_Code_20261023/
  README.md
  requirements.txt
  CODE_PROVENANCE.json            source locations and original SHA-256 hashes
  MANIFEST.sha256                 checksums of the delivered source files
  prepare_data.py                 explicit download / checksum verification
  run_experiments.py              stage launcher and command preview
  run_head_final.py               final-cohort orchestration without old screens
  verify_code_package.py          source integrity and small offline checks
  tests/test_portability.py       entry-point and path regression checks
  PORTABILITY_REPORT.md           independent-directory audit and its limits
  Control_Theoretic_DP_Numerics/
    code/                        original models, solvers, exports and verifiers
    figures/                     two data-driven PGFPlots figure sources
    requirements.txt
  experiments/
    neural_mechanism_2026_10_08/   reusable nonlinear dynamics and inference helpers
    lr_baseline_2026_10_08/       learning-rate allocation
    nonlinear_allocation_2026_10_08/
    network_performance_2026_10_09/
    solver_comparison_2026_10_08/
```

Keep this relative layout. Several original studies deliberately import the
same frozen forward, adjoint and allocation routines from sibling directories.
The historical date suffixes identify those implementations; they are not
different datasets or alternative manuscripts.

## 3. Environment and a quick offline check

The reference numerical environment uses Python 3.12 and **NumPy 2.3.5**.
The packaging checks were run with Python 3.12.14 and NumPy 2.3.5. The code
uses NumPy and Python's standard library; PyTorch, TensorFlow and a GPU are
not required. CPU memory use grows with path count, model size and workers.

Extract the ZIP and enter its top-level code directory first. On macOS/Linux:

```bash
unzip DP_SGD_Numerical_Experiment_Code_20261023.zip
cd DP_SGD_Numerical_Experiment_Code_20261023
```

Then create a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_code_package.py --smoke
```

On Windows, extract the ZIP in Explorer, open PowerShell in the extracted code
directory, and use the environment's interpreter directly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_code_package.py --smoke
```

For subsequent commands, replace `python` with
`.\.venv\Scripts\python.exe` if the environment is not activated. The launcher
passes subprocess arguments as lists and sets its own working directory;
launching it by absolute path from another working directory is supported.
The tested reference is Python 3.12.14; NumPy 2.3.5 itself requires Python
3.11 or newer. Other Python versions and operating systems were not executed
in this audit.

The last command validates source checksums and syntax, then runs small
data-free checks of simplex gaps, feasible updates, LQ bookkeeping, Armijo
bookkeeping, prediction metrics, Gaussian chunk indexing, confidence intervals,
Holm adjustment, LR allocation, dense VJP/adjoint checks, and the portable
entry-point contracts. It uses a temporary copy, downloads no
dataset, and leaves no test output in the distributed source tree. It does
not reproduce the full paper or certify a global optimum.

To inspect the complete execution sequence without running anything:

```bash
python run_experiments.py all --dry-run
```

## 4. Data preparation: separate from this code distribution

Tabular samples are generated from the seeds embedded in the experiment
programs. No synthetic sample files need to be supplied.

Image studies use the four original Fashion-MNIST IDX gzip archives:

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

Place them in `Control_Theoretic_DP_Numerics/data/fashion_mnist/`, then run:

```bash
python prepare_data.py --check-only
```

Alternatively, explicitly request download from the dataset URLs already
embedded in the original loader:

```bash
python prepare_data.py
```

Both commands verify the four original SHA-256 values. They also generate
the data-hash manifest needed by the original solver verifier. The source
package contains none of these data files. The stage launcher checks for
local data before starting; it does not silently download the dataset.

After preparing the data, check the environment without starting training:

```bash
python run_experiments.py all --no-figures --check-only
```

This verifies the NumPy version and the local dataset checksums. For a run
that includes figure compilation, use `all --check-only` instead; it also
checks for `pdflatex` and, when `kpsewhich` is available, the required TeX
packages. These checks happen before expensive numerical stages.

## 5. Run the manuscript experiments

Run stages from the package root, in this order:

```bash
python run_experiments.py core --workers 1
python run_experiments.py lr
python run_experiments.py head --workers 1
python run_experiments.py prediction --workers 1
python run_experiments.py solvers
```

For all numerical stages without LaTeX:

```bash
python run_experiments.py all --workers 1 --no-figures
```

`python run_experiments.py all --workers 1` additionally builds the figures.
It checks for the optional LaTeX tools in Section 7 before starting any
training. Individual stages report missing prerequisite outputs and identify
which stage must generate them; they do not read outputs from the original
project folder.

These are the full scientific protocols, not quick demonstration runs.
Several stages require substantial CPU time. `--workers` controls independent
case execution where the original program exposes it. The LR stage is serial;
the solver stage retains its original serial method scheduling and internal
four-worker neural oracle. Do not run other compute-intensive jobs during
timing studies. The launcher restricts numerical-library thread pools to one
thread per worker, consistently with the original timing policy.

### Core benchmark and learning-rate comparison

`core` generates the three representative schedules, the 90-case
three-task / three-budget / ten-seed benchmark, published geometric-family
and proxy-transfer comparisons, supporting controlled pipeline timing,
derivative/sensitivity diagnostics, and the analytic LQ validation. It then
exports plotting data and runs the original core result verifier.

`lr` reads the newly generated comprehensive results, evaluates the exact
learning-rate allocation on the same cases, and verifies its aggregate
comparisons. It must follow `core`; old experimental results are unnecessary.

### Final nonlinear-head cohorts

`head` uses the selected `train_fast384` setting:

- a publicly pretrained encoder and public PCA to eight input features;
- an 8-4-10 Softplus head, 86 trainable parameters, 120 updates;
- learning rate 1, batch size 384, regularization coefficient 0.0001;
- budget 0.5 with seeds 96001--96010;
- budget 0.25 with seeds 106001--106010;
- 128 optimization, 256 public-validation and 512 final-evaluation paths;
- 20 PB-AMD updates, with the original public checkpoint selection;
- LR, Decay, exact Linearized and publicly refined Geometric comparators.

The fresh-run driver calls the original computational routines. It refines
the geometric comparator, runs the original independent evaluator with public
and private RNG offsets 8,000,000 and 9,000,000, replays evaluations, computes
the original seed-level summaries and diagnostics, and applies joint Holm
adjustment over all eight budget-by-comparator tests.

Use `confirmation/independent_evaluation/` and
`budget_extension/independent_evaluation/` as the final evaluation records.
Parent case files retain the original intermediate evaluations and should not
be substituted for these independent final records.

The historical public screening campaign is not required to rerun the final
cohorts and is not claimed to be regenerated. The fresh audit covers the 20
final cases, rather than reporting the old 37-case historical audit. See
`PORTABILITY_NOTES.md` for the exact packaging changes.

### Prediction and solver studies

`prediction` reconstructs the fixed full-network schedules from the newly
generated core records and evaluates individual networks. It reports test
accuracy and **unregularized** cross-entropy; it does not ensemble predictions
or select schedules using test outcomes.

`solvers` runs the original oracle tests, public calibration, fixed-step
PB-AMD/PGD study, symmetric Armijo head study, penalty-and-repair extension,
independent replays and summaries. The main convergence figure uses fixed
steps for Tabular and Fashion-restart, and symmetric Armijo for the head.
The additional Fashion-constant control is retained for the original audit.
The reporting target is relative variational gap 1e-4; curves continue to
1e-6 or the original update cap. Failures remain failures in the summaries.

## 6. Outputs and reruns

On execution, outputs are generated in:

| Output | Location |
| --- | --- |
| Original benchmark, diagnostics, LQ results | `Control_Theoretic_DP_Numerics/results/` |
| Core plotting data and LaTeX table snippets | `Control_Theoretic_DP_Numerics/paper_artifacts/` |
| LR comparisons | `experiments/lr_baseline_2026_10_08/results/` |
| Head cohorts and independent evaluations | `experiments/nonlinear_allocation_2026_10_08/confirmation/` and `budget_extension/` |
| Joint head tables and diagnostics | `experiments/nonlinear_allocation_2026_10_08/combined_summary.json` and `conditional_diagnostics.json` |
| Full-network prediction | `experiments/network_performance_2026_10_09/results/` |
| Fixed-step solver study | `experiments/solver_comparison_2026_10_08/results/` |
| Armijo and penalty studies | `experiments/solver_comparison_2026_10_08/armijo_results/` and `penalty_results/` |
| Five final figure PDFs | `output/figures/` |

These paths are generated on the recipient's machine and are absent from the
delivered package. The original programs reject incompatible saved protocols
or source hashes. Keep the source and worker settings fixed when resuming.
The launcher enables resume where available; representative core runs and some
diagnostics are recomputed. Use a fresh extracted code folder for a clean run.
Absolute paths recorded in a new run's provenance belong to that run; do not
mix old result directories from another machine with newly generated records.

The original result verifiers are included. Their expected result files do
not exist until the corresponding stages finish. This package intentionally
cannot check byte-for-byte agreement against omitted historical raw results.
Wall-clock times also depend on the recipient's hardware and system load.

## 7. Optional figure generation

After the required numerical stages complete:

```bash
python run_experiments.py figures
```

This requires `pdflatex`, the `standalone` class, TikZ, PGFPlots and Latin
Modern fonts. Python packages alone are enough for the numerical stages.
The launcher invokes the three original Python figure exporters and compiles
the two data-driven PGFPlots sources, with two passes for named legends.

The five outputs are allocation regimes, nonlinear mechanism, network
performance, raw penalty charge, and solver efficiency. The figure sources
read generated numerical results; no precomputed plotting data are shipped.

## 8. Provenance and scope

`CODE_PROVENANCE.json` maps every copied source to its original project-relative
path and SHA-256 digest. Computational model, gradient, adjoint, allocation,
optimizer, seed, and inference implementations are preserved. Most copied
programs are byte-identical to their originals. Two have portability-only
edits: the head figure exporter no longer checks an unshipped manuscript
copy, and the direct independent-evaluator entry point uses the fresh-cohort
orchestration instead of reading historical screening manifests. The
evaluator's computational `evaluate_case` function is unchanged.

The copy does not modify the original manuscript, code, datasets or results.
The subsequent portability audit installed the requirements in a new virtual
environment and tested an extracted copy outside the project, with access to
the original project blocked at the Python file-I/O layer. See
`PORTABILITY_REPORT.md` for the executed checks and their limits. This is not
a claim that the complete multi-seed scientific campaign was rerun on every
supported operating system.

## 9. Exact replay settings

The article and supplement describe the experimental design, parameters, and
statistical comparisons. The following settings specify the random streams and
implementation details used to reproduce the individual runs.

### Problem seeds and representative cases

| Study | Problem seeds |
| --- | --- |
| Tabular allocation quality | 17, 31, 53, 71, 97, 113, 137, 151, 173, 197 |
| Full-network allocation quality | 123, 321, 777, 1001, 2027, 409, 613, 887, 1301, 1601 |
| Nonlinear head, budget 0.5 | 96001--96010 |
| Nonlinear head, budget 0.25 | 106001--106010 |

The representative allocation profiles use seed 17 for Tabular and seed 123 for
both full-network tasks. Diagnostic A uses head seed 96001 at budget 0.5;
diagnostic B uses seed 106006 at budget 0.25. These diagnostics were selected
after evaluating the two ten-seed head groups.

### Random-number generators and draw order

For each problem seed `s`, NumPy's `default_rng` uses the following seeds:

| Study | Generator seeds and order |
| --- | --- |
| Tabular | Private/test population: `s`; public data: `s + 10000`; batches: `s + 20000`; optimization, validation, and final Gaussian paths: `s + 101`, `s + 151`, and `s + 202`. |
| Full network | Fashion-MNIST splitting, batch construction, and random initialization use the generator at `s`, in that order. Public pretraining uses `s + 5000`. Optimization, validation, and final Gaussian paths use `s + 77`, `s + 177`, and `s + 1000`. |
| Nonlinear head | Encoder pretraining and data splitting follow the full-network construction. Head initialization and subsequent batch sampling use the generator at `s + 180000`, in that order. Optimization and validation Gaussian paths use `s + 184000` and `s + 185000`; public diagnostic and private-training evaluation use `s + 8000000` and `s + 9000000`. |
| Public-data perturbations | Label and feature perturbations use `s + 40001` and `s + 50001`. |

Full-network evaluation on private records draws Gaussian vectors for 16 paths
at a time. The head uses the same grouping for public validation, public
diagnostic evaluation, and evaluation on private records. All training steps
for one group are drawn before the next group. Optimization and full-network
validation instead draw the complete `(T, M, d)` array at once, where `d` is
the number of trainable parameters. Preserving this draw order is necessary
to replay the same Gaussian paths.

### Solver-calibration and convergence seeds

| Task | Calibration seeds | Convergence seeds |
| --- | --- | --- |
| Tabular | 210001--210003 | 210101--210110 |
| Fashion-constant | 211001--211003 | 211101--211110 |
| Fashion-restart | 212001--212003 | 212101--212110 |
| Nonlinear head | 213001--213003 | 213101--213110 |

The fixed Gaussian array has shape `(T, M, d)` and is generated from
`SeedSequence([20261008, task_index, problem_seed, 1])`. The task indices are
0, 1, 2, and 3 in the table's order. Every solver receives the same array
within an instance.

### Linear-quadratic implementation details

`Control_Theoretic_DP_Numerics/code/linear_quadratic_validation.py` computes the
water-filling multiplier with 240 bisection steps for `log(lambda)` on
`[-80, 80]`, followed by a summation correction. For initialization, every
coordinate starts at the floor and the remaining charge is assigned to
`(j + 40) % 80`, where `j = argmax(a_t)`.
