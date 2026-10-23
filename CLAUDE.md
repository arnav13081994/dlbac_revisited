# CLAUDE.md: reproducing this repository with an LLM assistant

## What this is

Re-examination of Nobi et al., "Toward Deep Learning Based Access Control" (arXiv:2203.15124), on the authors' own eight synthetic datasets and protocol. Question: does the ResNet's advantage there come from the architecture, the baseline configuration, or the structure of the benchmark's labels? Every number in the paper is produced by `make report` from the stored result files in `results/`; never type a number into the paper by hand, change `paper/generated/` by hand, or edit anything in `results/`.

## Commands (run from the repository root)

```bash
git submodule update --init     # the authors' repository -> external/DlbacAlpha (data, released code, shipped weights). Skip it if external/DlbacAlpha/dataset already exists (e.g. a copy without .git)
make setup                      # .venv with the pinned packages (Python 3.12)
make report                     # L0: regenerate results/REPORT_e*.txt, paper/generated/*.   ~10 s, CPU only, needs no .npz files. Also prints the block "ResNet minus each baseline:" (the rows of paper Table 4)
make test                       # self-checks. Last lines: "jobqueue selfcheck OK ...", then the pre-fix queue is rejected (the "!" line is expected)
make e2                         # port check. Last line: "... L1 difference of confusion counts: tf_same 0 ... pass (<= 9 of 8768): True"
make smoke                      # every model group on a tiny slice. Prints F1 values that mean nothing (tiny data); only failures matter
make paper                      # paper/main.pdf (tectonic)
```

`make report` rewrites `results/REPORT_e1.txt ... REPORT_e6.txt` and `paper/generated/`; the content must come out identical to what is stored (rewriting identical content is expected). If it differs, report the difference; do not "fix" the stored files. Everything else in `results/` is never edited.

## Reproducing a table

Ask which table, then find its source. `make report` alone regenerates all of them (no separate command per table): `paper/generated/tab_ladder.tex` (ResNet minus each baseline), `tab_reproduction.tex` (our runs against the figures read off the DLBAC paper), `tab_cold.tex` (entity hold-out splits), `tab_app_*.tex` (per-dataset appendix), `ladder.csv` (behind the figure). They are written by `scripts/make_tables.py`; the numbers quoted in the text are macros in `paper/generated/numbers.tex`. To recompute from scratch rather than from the stored results, run the matching lines of `jobs/*.txt`, then `make report`.

## Directory map

- `src/dlbac_revisited/`: all of the following modules live here. `dlbac.py`: data loading (`load`), the one-hot grid, the PyTorch ResNet with Keras-style padding (`Conv2dTF`), training and prediction.
- `run_cell.py`: one result cell (one dataset, seed and model group) -> `<out>.json` + `<out>.npz`. Groups: `resnet`, `resnet_eq`, `resnet_head`, `resnet_eq_head`, `theirs` (the paper's RF/SVM/MLP configuration), `theirs_eq` (the same with the 64 indicators appended), `tuned` (RF, GB, one-hot MLP, each with and without the 64 equality indicators), `svm`. Subcommands: `main` (authors' split), `cold` (entity hold-out), `layout` (column-order check).
- `analyze.py`: the reports `results/REPORT_e1..e6.txt`. `joblists.py`: writes `jobs/*.txt`. `jobqueue.py`: runs a job list with a load budget. `port_check.py`: E2.
- `scripts/make_tables.py`, `entity_overlap.py`, `check_quotes.py` (checks quotations against the text of the DLBAC paper that you convert yourself with `pdftotext`).
- `docs/analysis_plan.md`: the plan fixed before the runs (do not edit). `docs/claims_audit.md`: the statements examined. `docs/gotchas.md`: read it before changing code.

## Do not

- Do not edit `results/`, `docs/analysis_plan.md`, or `external/` (the authors' files).
- Do not start `jobs/*.txt` (hundreds of hours of job time) without saying what it costs and asking. L0 and L1 are free to run.
- Do not retune a model or change a seed, grid or margin to make a number match. If a number differs from the paper, report the difference.
- Do not run scripts without `-I` or with another environment active: use `env -u VIRTUAL_ENV ./.venv/bin/python -I ...` as the Makefile does. Library threads are pinned to 1 inside `run_cell.py`; keep that before `import numpy`.
- Keep the paper's wording neutral about the DLBAC authors' work: describe settings and numbers, not intent, and avoid words like flawed or misleading.

## How to read a report

`results/REPORT_e*.txt` are plain text. Each starts with per-dataset tables (F1, FPR, TPR, precision; mean over seeds), followed by the comparison families. A family line reads: comparison, mean difference in F1 over the 8 datasets, 95% bootstrap interval, datasets won, Holm-adjusted p of the difference test (`p_diff(Holm)`), the equivalence test (`TOST(Holm)`) and a verdict. "PRE-SPECIFIED FAMILY" marks E1, E3 and E4; E5, E5c and E6 are follow-ups added after the first results (`docs/analysis_plan.md`). The paper's tables use only the mean difference, datasets won and the plain paired t-test p over the 8 datasets (no multiple-comparison adjustment); the bootstrap interval, the equivalence tests and the Holm adjustment appear only in these reports. The two tuned-SVM rows are in `REPORT_e6.txt`, not `REPORT_e1.txt`. Reports print very small p as `0.000`; the paper tables print `<0.001`.

## Known state

- Text versus code: where the DLBAC paper's text and its released code differ (visible columns, depth, schedule) this repository follows the code (`docs/gotchas.md` 1).
- The stored ResNets used `--device mps`. `make smoke` and `make e2` have been run on `cpu`; `cuda` and the full runs on other devices are untested.
- Re-running the stored `u4k-r4k-auth11k` seed-0 `resnet` cell on the same backend reproduced its probabilities exactly; other cells and devices are unchecked, so on a different device expect small differences and report them.
- `results/g1` holds the default baselines with indicators and `results/h1` the `amazon1` control (no indicators: one resource field; reported with AUC and an all-allow reference row because F1 alone is uninformative at about 94% allows).
- Our ResNet on `u4k-r6k-auth28k` is at 0.882 F1 against 0.957 in the paper's figure (the one ResNet cell outside the 0.03 tolerance); this is reported without an explanation.
- Two stored cells were run twice by mistake (`docs/gotchas.md` 5); the later result is kept.
