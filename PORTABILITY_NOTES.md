# Portability notes

The original model and optimizer files are retained in their original relative
layout so that imports and computational source-hash checks remain meaningful.
The source mapping is in `CODE_PROVENANCE.json`.

The following additions are distribution helpers:

1. `prepare_data.py` uses the original download routine and expected checksums.
   It only downloads when explicitly invoked without `--check-only`. It writes
   a recipient-side data-hash manifest expected by the original solver audit.
2. `run_experiments.py` launches the existing programs at the paths their later
   dependencies expect. It does not change model, solver, candidate-grid,
   clipping, privacy, stopping or statistical settings. `--dry-run` prints the
   commands without requiring data or NumPy imports in the launcher.
   Environment and stage prerequisites are checked before computation.
   `all --no-figures` supports a NumPy-only numerical run. TeX inputs and table
   roots use relative ASCII paths, including when the enclosing directory name
   contains spaces or non-ASCII characters.
   The added entry points read source/configuration text explicitly as UTF-8
   and pass UTF-8 mode to Python subprocesses, avoiding platform-default text
   encodings in the documented workflow.
3. `run_head_final.py` orchestrates the two final head cohorts using the original
   design, refinement, independent-evaluation, replay, summary and diagnostic
   functions. Historical screening outcomes and their old manifest files are
   not inputs. Its RNG audit explicitly covers the two regenerated cohorts,
   and its final audit identifies its 20-case scope. It does not assert that
   the original 17 screening cases or the 37-case historical audit were rerun.
4. `verify_code_package.py` checks delivered checksums and Python syntax, and
   optionally runs the original data-free checks in a temporary source copy.

Two copied files have narrow orchestration/export edits:

`experiments/nonlinear_allocation_2026_10_08/paper_update/build_figure.py`
no longer reads an old local manuscript source or checks that rounded table
cells occur in that text. A manuscript source is not part of this code-only
distribution. It still checks the independently evaluated cases, schedule
budgets, result hashes, and agreement of regenerated comparisons with the
regenerated summary before emitting the figure source. Plotting and numerical
calculations otherwise remain as in the original exporter.

`experiments/nonlinear_allocation_2026_10_08/independent_evaluation.py`
keeps `evaluate_case` unchanged. Its direct entry point now calls the same
fresh-cohort evaluator as the top-level driver and accepts `--cohort`.
It no longer waits for or reads historical public-screening manifests that
are not part of this code-only distribution.

No original experimental results are repackaged. The protocol configurations
describe prespecified settings; they are not datasets or generated result
records. The computational scripts may create results when deliberately run.
