# dlbac_revisited

Code, stored results and paper source for *Revisiting Deep Learning Based Access Control: Baseline Strength, Relational Features and Generalisation*. It re-examines Nobi, Krishnan, Huang, Shakarami and Sandhu, "Toward Deep Learning Based Access Control" (CODASPY 2022, arXiv:2203.15124) on the authors' own eight synthetic datasets, protocol and metrics (throughout, *the authors* are the authors of DLBAC), and asks whether the ResNet's advantage there reflects the architecture, the baseline configuration, or the structure of the benchmark's labels.

## Get it

```bash
git clone --recurse-submodules https://github.com/arnav13081994/dlbac_revisited.git
cd dlbac_revisited
make setup          # Python 3.12 virtual environment with the pinned packages (.venv)
```

The authors' repository (data, released code, shipped weights) comes in as the git submodule `external/DlbacAlpha`, pinned to the commit the results were produced with. None of its files are copied here. The copy we used carries no licence file, so nothing of theirs is redistributed; our own files are released under the MIT licence (`LICENSE`), which covers only them.

## What you can reproduce

| Level | Command | What it does | Cost |
| --- | --- | --- | --- |
| L0 | `make report` | Rebuilds every report, table and number in the paper from the stored result files in `results/`. | seconds, CPU |
| L1 | `make test smoke e2` | Self-checks of the analysis and the job queue; every model group once on a tiny slice; the port check (our PyTorch ResNet with the authors' shipped weights must give their confusion counts 4746, 87, 50, 3885). | about a minute, CPU |
| L2 | one line of `jobs/*.txt` | One cell of the experiment, for example `u4k-r4k-auth11k`, seed 0. | about 6 minutes for the ResNet cell when run alone (the stored runs, several at a time, recorded about 16 minutes); the tuned classical models took about 80 minutes recorded |
| L3 | all of `jobs/*.txt` | The full experiments: 559 result cells. | hundreds of hours of job time in total (the stored runs ran several jobs at a time) |

`jobs/*.txt` hold one shell command per cell; `src/dlbac_revisited/jobqueue.py` runs such a list with at most N load units at once (`python -I src/dlbac_revisited/jobqueue.py jobs/main_cpu.txt 4`; it skips cells whose result file exists, so it can be stopped and restarted). `make joblists` rewrites them, for example with `--device cuda` or `--device cpu`.

The stored results used the PyTorch `mps` backend for the ResNets. Other devices have not been used for the full runs. Re-running one stored cell (`u4k-r4k-auth11k`, seed 0, `resnet`) on the same backend gave identical predictions and probabilities (largest difference 0.0); other cells and other devices were not compared. The small result arrays (`results/**/*.npz`) are not in git; they are not needed by `make report`.

## Layout

```
paper/                 LaTeX source; paper/generated/ is written by `make report` (tables, numbers.tex, csv behind the figure)
docs/                  analysis_plan.md (plan fixed before the runs), claims_audit.md (statements examined), gotchas.md
src/dlbac_revisited/   dlbac.py (data, one-hot grid, ResNet), run_cell.py (one result cell), analyze.py (reports),
                       port_check.py, joblists.py, jobqueue.py
scripts/               make_tables.py, entity_overlap.py, check_quotes.py
tests/                 test_analyze.py, test_jobqueue.py, fixtures/ (the pre-fix queue the test must reject)
results/               e1 e3 e4 f1 f3 g1 h1 (one json per cell), REPORT_e1..e6.txt, e2_port_results*.json, ENVIRONMENT.txt
jobs/                  the job lists
data/paper_targets.json  F1 values read off the DLBAC paper's figures (about +-0.015), used for the reproduction table
external/DlbacAlpha    the authors' repository (git submodule)
```

Experiment folders in `results/`: `e1` the authors' split (seeds 0-4); `e3` entity hold-out splits (split seeds 0-2); `e4` column-order check (stored, not discussed in the paper); `f1` and `f3` the ResNet with a hidden layer in its output head and the tuned SVM, on the authors' split and on the hold-out splits; `g1` the paper's default baselines with the 64 equality indicators appended (40 cells); `h1` the authors' real-world `amazon1` data as a control without equality structure (15 cells). The experiment definitions are in `docs/analysis_plan.md`.

## Using an LLM assistant

`CLAUDE.md` is written for that: it gives the commands, the expected outputs, what not to touch and how to read the reports.

## Citing

See `CITATION.cff`.
