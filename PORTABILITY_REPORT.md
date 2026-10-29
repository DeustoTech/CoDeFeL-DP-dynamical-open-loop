# Independent-directory portability review

Reviewed on **23 October 2026**. This report concerns the code distribution,
not a new scientific comparison of the allocation methods.

## Audit setup

- Extracted the actual ZIP outside the original project, renamed its root
  directory to include both spaces and Chinese characters, and launched
  commands from both that root and an unrelated working directory.
- Created a fresh virtual environment, with system site packages disabled,
  and installed `requirements.txt` using pip. The executed environment was
  **macOS arm64, Python 3.12.14, NumPy 2.3.5**. NumPy was loaded from this
  fresh environment rather than the pre-existing scientific runtime.
- Installed a Python audit hook that rejected file access to the original
  project directory in the test processes and their Python subprocesses.
  An intentional read probe verified that this guard was active.
- Downloaded Fashion-MNIST anew into the temporary extraction and verified all
  four original SHA-256 checksums. No dataset or result was copied from the
  original project.

The guard is a Python file-I/O check, not an operating-system security sandbox.
Standard-library, installed NumPy, and optional TeX files remain normal external
software dependencies.

## Findings and fixes

### 1. PGFPlots failed with a non-ASCII enclosing directory

The original launcher embedded the full absolute result path in the TeX
table macros. The allocation-figure source failed with `Unicode character`
when that path included Chinese characters. The failure was reproduced using
the actual PGFPlots source and temporary input tables.

The launcher now sets its working directory to the package root and supplies
relative ASCII paths to TeX. The same allocation source and the penalty-charge
source then compiled successfully, including the second pass for named legends.
Temporary table fixtures exercised filename handling; they were not scientific
results and are not included in this distribution.

### 2. A legacy evaluator entry point still read historical manifests

The top-level head driver already avoided historical screening results, but
directly running the old `independent_evaluation.py` entry point still read
omitted public-screening manifests. That direct entry point now delegates to
the same fresh-cohort orchestration and accepts `--cohort confirmation` or
`--cohort budget_extension`. It requires the newly generated design/refinement
outputs for the selected cohort, which is a normal within-package dependency.

The original computational `evaluate_case` function was checked against the
source project and is AST-identical. Forward dynamics, Gaussian streams,
metrics and independent replay computations were not changed.

### 3. Missing software or prior stages needed clearer handling

The old `all` command checked for LaTeX only after lengthy numerical stages.
The launcher now checks the environment first. `all --no-figures` runs the
numerical stages without TeX. `--check-only` checks readiness without starting
experiments, and individual stages identify missing outputs and their producer
stages instead of immediately producing an unexplained file-not-found traceback.

The README now includes extraction, virtual-environment setup, Windows
interpreter commands, explicit data preparation, preflight commands, numerical
stage order, optional figure generation, and resume limitations.

## Executed checks

| Check | Result and scope |
| --- | --- |
| Fresh installation of the declared Python requirements | Passed in the isolated environment |
| Source checksums, Python syntax and launcher flag/target-parser compatibility | Passed |
| Six portability regression tests | Passed, including foreign working directory, relative TeX paths, numerical-only mode, early TeX check and missing-stage messages |
| Original offline scientific checks | Passed: simplex/feasible updates, LQ bookkeeping, Armijo bookkeeping, prediction metrics, statistical intervals/Holm, head VJP/adjoints and LR allocation |
| Original real-data oracle checks | Passed for Tabular, Fashion-constant, Fashion-restart and Nonlinear head; threaded/serial oracle agreement and finite-difference checks |
| Original Armijo recurrence and penalty/repair recurrence tests | Passed using the freshly downloaded data |
| Small real Tabular, Fashion-constant, Fashion-restart and nonlinear-head runs | Passed with reduced path counts and two optimizer updates; actual models, generated outputs and data flow exercised |
| Head geometric refinement and recorded-case replay | Passed for the small real head run |
| Fresh independent head evaluation | Passed; maximum replay discrepancy about 2.3e-16 for the small run |
| Missing-TeX and missing-prerequisite behavior | Passed; the numerical-only preflight succeeds without TeX, while a figure-inclusive run fails before starting experiments if TeX is unavailable |
| Figure compilation in the renamed directory | Passed for both data-driven PGFPlots figure sources using production relative-path command construction and temporary layout fixtures |
| Original-project access guard | No attempted original-project access in the checked runtime processes; the intentional guard self-test was logged separately |

The delivered `verify_code_package.py --smoke` command runs its checks in a
temporary source copy whose directory name itself contains spaces and Chinese
characters. It includes the portability regression tests and downloads no data.

## Limits

- This is one-machine isolation testing, not execution on separate Windows
  and Linux machines. Their setup instructions are provided, but actual runs
  on those platforms are not claimed.
- The complete 90-case core benchmark, both full ten-seed head cohorts, and
  the complete solver campaign were not rerun. Small runs test startup, I/O,
  source closure and component integration; they do not reproduce the paper's
  numerical tables or establish global optimizer correctness.
- An additional full-size representative Tabular trial reached the audit's
  300-second per-process cap. It was stopped and is not counted as a passed
  experiment. The distributed runner does not impose that audit cap.
- The three result-based figure exporters were checked for source/input
  closure and their prerequisite outputs, but their full scientific inputs
  were not regenerated in this review. The two PGFPlots filename tests do not
  constitute a regeneration of all five scientific figures.
- Network access is needed only to install missing software or explicitly
  download the public dataset. Existing local archives can instead be checked
  with `prepare_data.py --check-only`. The code ZIP contains no datasets,
  experimental results, compiled figures, logs or caches.

Within these limits, the supported entry points use the extracted package,
newly generated outputs, the stated public dataset and declared software
dependencies. No original-project code or historical result folder is needed.
