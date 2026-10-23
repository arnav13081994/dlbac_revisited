# Everything is run from the repository root. Levels: L0 = make report, L1 = make smoke e2 test, L2/L3 = see README.md.
PY = env -u VIRTUAL_ENV ./.venv/bin/python -I
DS = u4k-r4k-auth11k
SMOKE = results/_smoke

.PHONY: setup report e2 smoke test joblists paper

setup:  ## virtual environment + pinned packages (h5py only for e2); the authors repository must be checked out: git submodule update --init
	python3.12 -m venv .venv && env -u VIRTUAL_ENV ./.venv/bin/python -m pip install -r requirements-port.txt

report:  ## L0: every report, table and number in the paper from the stored results (about 10 s, CPU only; needs the submodule for the overlap statistic)
	$(PY) src/dlbac_revisited/analyze.py --exp all
	$(PY) scripts/make_tables.py
	$(PY) scripts/entity_overlap.py

e2:  ## port check: our PyTorch network with the authors shipped 11k weights must give their confusion counts 4746,87,50,3885
	mkdir -p $(SMOKE) && $(PY) src/dlbac_revisited/port_check.py --weights external/DlbacAlpha/neural_network_training/results/dlbac_alpha.hdf5 --dataset $(DS) --depth 8 --expect 4746,87,50,3885 --out $(SMOKE)/e2.json

smoke:  ## every model group once on a tiny slice of the smallest dataset: checks that the code runs, not the results
	mkdir -p $(SMOKE) && for g in resnet resnet_eq resnet_head resnet_eq_head theirs tuned svm; do $(PY) src/dlbac_revisited/run_cell.py main --ds $(DS) --seed 0 --group $$g --tiny --out $(SMOKE)/$$g.json || exit 1; done
	$(PY) src/dlbac_revisited/run_cell.py cold --ds $(DS) --seed 0 --group tuned --tiny --out $(SMOKE)/cold_tuned.json
	$(PY) src/dlbac_revisited/run_cell.py main --ds $(DS) --seed 0 --group theirs_eq --tiny --out $(SMOKE)/theirs_eq.json
	for g in resnet theirs tuned; do $(PY) src/dlbac_revisited/run_cell.py main --ds amazon1 --seed 0 --group $$g --tiny --out $(SMOKE)/amazon1_$$g.json || exit 1; done

test:  ## self-checks of the analysis and of the job queue; the pre-fix queue must fail the second one
	$(PY) tests/test_analyze.py
	$(PY) tests/test_jobqueue.py
	! $(PY) tests/test_jobqueue.py tests/fixtures/cpu_queue_prefix.py

joblists:  ## rewrite jobs/*.txt (one shell command per cell; the full run is these lines, see README.md)
	$(PY) src/dlbac_revisited/joblists.py

paper:  ## paper/main.pdf (needs tectonic or any LaTeX with pgfplots)
	cd paper && tectonic main.tex
