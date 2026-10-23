"""Turns the stored result cells into everything the paper shows: LaTeX tables, the CSV behind the main figure, and paper/generated/numbers.tex (one macro per number used in the text, so no number in the paper is typed by hand).
Reporting follows the DLBAC paper: F1 (also FPR, TPR, precision in the appendix) at threshold 0.5, pooled over the four operations; a paired t-test across the 8 datasets, p < 0.05, as the DLBAC paper uses for its RF comparison. A difference is the mean over the 8 datasets of (mean over seeds).
usage (from the repository root): python -I scripts/make_tables.py"""
import csv, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "dlbac_revisited")); import analyze as A
import json
ROOT, OUT = "results", "paper/generated"; os.makedirs(OUT, exist_ok=True)
DS = A.DS; SHORT = {d: d.replace("u", "U", 1) for d in DS}
R = A.read(ROOT, "e1", "f1", "g1"); RC = A.read(ROOT, "e3", "f3")
camel = lambda s: "".join(w.capitalize() for w in s.split("_"))
NAME = {"resnet": "ResNet (authors' recipe)", "resnet_eq": "ResNet + indicators", "resnet_head": "ResNet, hidden-layer head", "resnet_eq_head": "ResNet, hidden-layer head + indicators",
        "rf_theirs": "RF, default configuration", "svm_theirs": "SVM, default configuration", "mlp_raw_theirs": "MLP, default configuration",
        "rf_tuned": "RF, tuned", "gb_tuned": "GB, tuned", "mlp_tuned": "MLP, tuned (one-hot)", "svm_tuned": "SVM, tuned (one-hot)",
        "rf_tuned_eq": "RF, tuned + indicators", "gb_tuned_eq": "GB, tuned + indicators", "mlp_tuned_eq": "MLP, tuned + indicators", "svm_tuned_eq": "SVM, tuned + indicators"}
GROUP = {"resnet": "ResNet", "resnet_eq": "ResNet", "resnet_head": "ResNet", "resnet_eq_head": "ResNet", "rf_theirs": "default", "svm_theirs": "default", "mlp_raw_theirs": "default",
         "rf_tuned": "tuned", "gb_tuned": "tuned", "mlp_tuned": "tuned", "svm_tuned": "tuned", "rf_tuned_eq": "tuned + indicators", "gb_tuned_eq": "tuned + indicators", "mlp_tuned_eq": "tuned + indicators", "svm_tuned_eq": "tuned + indicators"}
NAME.update({"rf_theirs_eq": "RF, default configuration + indicators", "svm_theirs_eq": "SVM, default configuration + indicators", "mlp_raw_theirs_eq": "MLP, default configuration + indicators"})
GROUP.update({"rf_theirs_eq": "default + indicators", "svm_theirs_eq": "default + indicators", "mlp_raw_theirs_eq": "default + indicators"})
F = lambda m, key="f1": A.ds_mean(R, m, lambda r: r[key])            # (8,) mean over seeds, per dataset
FS = lambda m, s, key="f1": np.array([R[(d, s)][m][key] for d in DS])  # (8,) one seed
CF = lambda m, s: A.ds_mean(RC, m, lambda r: (r["per_set"].get(s) or {}).get("f1"))
for m in NAME: assert not np.isnan(F(m)).any(), f"missing cells for {m}"
M = {}  # macro name -> text
f4 = lambda x: f"{x:.4f}"; sgn = lambda x: f"{x:+.4f}"; pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
for m in NAME: M["f" + camel(m)] = f4(F(m).mean())

# ---------- Table: ResNet minus each model (ladder) ----------
order = ["rf_theirs", "mlp_raw_theirs", "svm_theirs", "rf_theirs_eq", "mlp_raw_theirs_eq", "svm_theirs_eq", "rf_tuned", "gb_tuned", "mlp_tuned", "svm_tuned", "rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq", "svm_tuned_eq"]
rows, ps = [], []
for m in order:
    d = F("resnet") - F(m); p = stats.ttest_rel(F("resnet"), F(m)).pvalue; ps.append(p); rows.append((m, F(m).mean(), d.mean(), int((d > 0).sum()), p))
holm = A.holm(ps)
with open(f"{OUT}/tab_ladder.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nBaseline & Setting & F1 & ResNet minus baseline & Datasets won by ResNet & $p$ & adjusted $p$ \\\\\n\\midrule\n")
    prev = None
    for (m, mean, d, w, p), h in zip(rows, holm):
        if prev and GROUP[m] != prev: f.write("\\midrule\n")
        prev = GROUP[m]; fam = {"rf": "RF", "gb": "GB", "mlp": "MLP", "svm": "SVM"}[m.split("_")[0]]
        f.write(f"{fam} & {GROUP[m]} & {mean:.4f} & {d:+.4f} & {w} of 8 & {pf(p)} & {pf(h)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    for (m, mean, d, w, p), h in zip(rows, holm):
        M["d" + camel(m)] = sgn(d); M["w" + camel(m)] = str(w); M["p" + camel(m)] = pf(p); M["h" + camel(m)] = pf(h)
M["holmFlips"] = ", ".join(NAME[m] for (m, *_), p, h in zip(rows, ps, holm) if (p < 0.05) != (h < 0.05)) or "none"
print("ResNet minus each baseline:"); [print(f"  {NAME[m]:32s} F1 {mean:.4f} diff {d:+.4f} won {w}/8 p {p:.3f} holm {h:.3f}") for (m, mean, d, w, p), h in zip(rows, holm)]
print("  conclusions that change after Holm adjustment over these comparisons:", M["holmFlips"])

# ---------- Gain from the indicators, per family ----------
gains = {}
for fam in ("rf", "gb", "mlp", "svm"):
    a, b = F(f"{fam}_tuned"), F(f"{fam}_tuned_eq"); gains[fam] = (b - a).mean(); M["gain" + fam.capitalize()] = sgn(gains[fam]); M["gainWins" + fam.capitalize()] = str(int(((b - a) > 0).sum()))
print("gain from the 64 indicators:", {k: round(float(v), 4) for k, v in gains.items()}, {fam: M["gainWins" + fam.capitalize()] + "/8" for fam in gains})

# ---------- ResNet variants ----------
for a, b, tag in (("resnet_head", "resnet", "HeadVsResnet"), ("resnet_eq_head", "resnet_head", "IndicatorsOnHead"), ("resnet_eq", "resnet", "IndicatorsOnResnet"), ("resnet_eq_head", "resnet_eq", "HeadEqVsEq"), ("resnet_head", "gb_tuned_eq", "HeadVsGbEq"), ("resnet_head", "mlp_tuned_eq", "HeadVsMlpEq"), ("resnet_head", "rf_tuned_eq", "HeadVsRfEq"), ("resnet_head", "svm_tuned_eq", "HeadVsSvmEq")):
    d = F(a) - F(b); M["v" + tag] = sgn(d.mean()); M["vp" + tag] = pf(stats.ttest_rel(F(a), F(b)).pvalue); M["vw" + tag] = str(int((d > 0).sum()))
print("hidden-layer head vs ResNet:", M["vHeadVsResnet"], "won", M["vwHeadVsResnet"], "/8; indicators on top of the head:", M["vIndicatorsOnHead"], "; head vs GB+eq", M["vHeadVsGbEq"])

# ---------- Seeds: the orderings the text states, checked in every seed ----------
seeds = sorted({k[1] for k in R}); rowsS = []; ok_cfg = ok_ind = True
for s_ in seeds:
    r = {m: FS(m, s_).mean() for m in NAME}; row = [s_, r["resnet"]]
    for fam, d0 in (("rf", "rf_theirs"), ("mlp", "mlp_raw_theirs"), ("svm", "svm_theirs")):
        c = r[d0] < r[f"{fam}_tuned"]; i0 = r[d0] < r[d0 + "_eq"]; i1 = r[f"{fam}_tuned"] < r[f"{fam}_tuned_eq"]; ok_cfg &= c; ok_ind &= bool(i0 and i1)
        row += [r[d0], r[d0 + "_eq"], r[f"{fam}_tuned"], r[f"{fam}_tuned_eq"], c, i0, i1]
    i1 = r["gb_tuned"] < r["gb_tuned_eq"]; ok_ind &= bool(i1); row += [r["gb_tuned"], r["gb_tuned_eq"], i1]; rowsS.append(row)
with open(f"{OUT}/seed_check.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "resnet"] + [f"{fam}_{c}" for fam in ("rf", "mlp", "svm") for c in ("default", "default_eq", "tuned", "tuned_eq", "tuned_gt_default", "default_eq_gt_default", "tuned_eq_gt_tuned")] + ["gb_tuned", "gb_tuned_eq", "gb_eq_gt"])
    [w.writerow([x if not isinstance(x, float) else f"{x:.4f}" for x in r]) for r in rowsS]
M["seedOrderConfig"] = f"all {len(seeds)} seeds" if ok_cfg else "not in every seed"; M["seedOrderIndicators"] = f"all {len(seeds)} seeds" if ok_ind else "not in every seed"; M["nSeeds"] = str(len(seeds))
sd = lambda m: float(np.std([FS(m, s).mean() for s in seeds], ddof=1)); M["seedSdResnet"] = f4(sd("resnet"))
print("per seed: tuned > default for RF, MLP, SVM:", ok_cfg, "| indicators raise every family, default and tuned:", ok_ind, "| ResNet std over seeds of the 8-dataset mean:", M["seedSdResnet"])

# ---------- Figure data ----------
with open(f"{OUT}/ladder.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "name", "group", "f1", "std_over_seeds"])
    for m in sorted(NAME, key=lambda m: F(m).mean()): w.writerow([m, NAME[m], GROUP[m], f"{F(m).mean():.4f}", f"{sd(m):.4f}"])
with open(f"{OUT}/spread.csv", "w", newline="") as f:  # the paper's robustness claim: spread of the per-dataset F1 over the 8 datasets
    w = csv.writer(f); w.writerow(["model", "name", "std_over_datasets"]); [w.writerow([m, NAME[m], f"{F(m).std(ddof=1):.4f}"]) for m in sorted(NAME, key=lambda m: F(m).std(ddof=1))]

# ---------- Reproduction check against the published figures (chart-read, +-0.015) ----------
T = json.load(open("data/paper_targets.json")); tol = T["tolerance_prestated"]["f1_tpr_precision_abs"]; mine = {"dlbac_r": "resnet", "rf": "rf_theirs", "svm": "svm_theirs", "mlp": "mlp_raw_theirs"}; within = n = 0; outside = []
with open(f"{OUT}/tab_reproduction.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{ResNet} & \\multicolumn{2}{c}{RF} & \\multicolumn{2}{c}{MLP} & \\multicolumn{2}{c}{SVM} \\\\\nDataset & paper & ours & paper & ours & paper & ours & paper & ours \\\\\n\\midrule\n")
    for i, d in enumerate(DS):
        cells = [d]
        for k in ("dlbac_r", "rf", "mlp", "svm"):
            t = T["datasets"][d][k]["f1"]; o = F(mine[k])[i]; n += 1; within += abs(t - o) <= tol; cells += [f"{t:.3f}", f"{o:.3f}" if abs(t - o) <= tol else f"\\textbf{{{o:.3f}}}"]; (None if abs(t - o) <= tol else outside.append(f"{d} {k} paper {t:.3f} ours {o:.3f}"))
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("outside the tolerance:", *outside, sep="\n  "); M["reproWithin"] = str(within); M["reproCells"] = str(n); M["reproTol"] = f"{tol:.2f}"
dm = {k: float(np.mean([F(mine[k])[i] - T["datasets"][d][k]["f1"] for i, d in enumerate(DS)])) for k in mine}; M["reproMlpGap"] = sgn(dm["mlp"])
print(f"our runs within {tol} of the chart-read published F1: {within} of {n} cells; mean (ours - paper): {({k: round(v, 3) for k, v in dm.items()})}")

# ---------- Cold splits ----------
cold_models = ["rf_tuned", "gb_tuned", "mlp_tuned", "svm_tuned", "resnet", "resnet_head", "rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq", "svm_tuned_eq"]; SETS = ["warm"] + A.COLD
for m in cold_models: assert not np.isnan(CF(m, "warm")).any() and all(not np.isnan(CF(m, s)).any() for s in A.COLD), m
with open(f"{OUT}/tab_cold.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\nModel & Warm & User-cold & Resource-cold & Both-cold \\\\\n\\midrule\n")
    for i, m in enumerate(cold_models):
        if i in (4, 6): f.write("\\midrule\n")
        f.write(f"{NAME[m]} & " + " & ".join(f"{CF(m, s).mean():.4f}" for s in SETS) + " \\\\\n")
        for s in SETS: M["c" + camel(m) + camel(s)] = f4(CF(m, s).mean())
    f.write("\\bottomrule\n\\end{tabular}\n")
for m in cold_models: M["drop" + camel(m)] = f4(CF(m, "warm").mean() - CF(m, "both_cold").mean())
for aware in ("rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq", "svm_tuned_eq"):
    for s in A.COLD: M["cd" + camel(aware) + camel(s)] = sgn((CF(aware, s) - CF("resnet", s)).mean())
    for s in A.COLD: M["chd" + camel(aware) + camel(s)] = sgn((CF(aware, s) - CF("resnet_head", s)).mean())
# hold-out comparisons: difference of the means over datasets (paired t-test over the datasets, unadjusted), for the three cold sets
CMP = [("ResNet minus tuned RF", "resnet", "rf_tuned"), ("ResNet minus tuned GB", "resnet", "gb_tuned"), ("ResNet minus tuned MLP", "resnet", "mlp_tuned"), ("ResNet minus tuned SVM", "resnet", "svm_tuned"), None,
       ("RF + indicators minus ResNet", "rf_tuned_eq", "resnet"), ("GB + indicators minus ResNet", "gb_tuned_eq", "resnet"), ("MLP + indicators minus ResNet", "mlp_tuned_eq", "resnet"), ("SVM + indicators minus ResNet", "svm_tuned_eq", "resnet"), None,
       ("ResNet + hidden layer minus RF + indicators", "resnet_head", "rf_tuned_eq"), ("ResNet + hidden layer minus GB + indicators", "resnet_head", "gb_tuned_eq"), ("ResNet + hidden layer minus MLP + indicators", "resnet_head", "mlp_tuned_eq"), ("ResNet + hidden layer minus SVM + indicators", "resnet_head", "svm_tuned_eq")]
cold_p = {}
for c in CMP:
    if c is None: continue
    for s_ in A.COLD: cold_p[(c[0], s_)] = stats.ttest_rel(CF(c[1], s_), CF(c[2], s_)).pvalue
keys = list(cold_p); cold_adj = dict(zip(keys, A.holm([cold_p[k] for k in keys]))); M["nColdTests"] = str(len(keys))
with open(f"{OUT}/tab_cold_cmp.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\nComparison & User-cold & Resource-cold & Both-cold \\\\\n\\midrule\n")
    for c in CMP:
        if c is None: f.write("\\midrule\n"); continue
        lab, x, y = c
        f.write(f"{lab} & " + " & ".join(f"{(CF(x, s_) - CF(y, s_)).mean():+.4f} ({pf(cold_p[(lab, s_)])}; {pf(cold_adj[(lab, s_)])})" for s_ in A.COLD) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
grp = lambda pre: [k for k in keys if k[0].startswith(pre)]
aw, hd = grp("RF + ind") + grp("GB + ind") + grp("MLP + ind") + grp("SVM + ind"), grp("ResNet + hidden layer minus RF") + grp("ResNet + hidden layer minus GB") + grp("ResNet + hidden layer minus MLP")
M["coldAwareN"] = str(len(aw)); M["coldAwareUnadj"] = str(sum(cold_p[k] < .05 for k in aw)); M["coldAwareAdj"] = str(sum(cold_adj[k] < .05 for k in aw))
M["coldHeadN"] = str(len(hd)); M["coldHeadUnadj"] = str(sum(cold_p[k] < .05 for k in hd)); M["coldHeadAdj"] = str(sum(cold_adj[k] < .05 for k in hd))
for nm, pre in (("Rf", "ResNet minus tuned RF"), ("Gb", "ResNet minus tuned GB"), ("Mlp", "ResNet minus tuned MLP"), ("Svm", "ResNet minus tuned SVM")):
    kk = grp(pre); M["coldRes" + nm + "UnadjMax"] = pf(max(cold_p[k] for k in kk)); M["coldRes" + nm + "AdjMin"] = pf(min(cold_adj[k] for k in kk)); M["coldRes" + nm + "AdjMax"] = pf(max(cold_adj[k] for k in kk))
print("hold-out comparisons significant: equality-aware minus ResNet", M["coldAwareUnadj"], "unadjusted /", M["coldAwareAdj"], "after Holm of", M["coldAwareN"], "| hidden-layer minus aware (RF, GB, MLP)", M["coldHeadUnadj"], "/", M["coldHeadAdj"], "of", M["coldHeadN"])
ud = [CF(m, "user_cold").mean() - CF(m, "warm").mean() for m in cold_models]; M["userDeltaMin"] = sgn(min(ud)); M["userDeltaMax"] = sgn(max(ud))  # user-cold minus warm, over the models of the cold table
print("cold F1 (warm -> both-cold drop):", {m: (f4(CF(m, 'warm').mean()), f4(CF(m, 'both_cold').mean())) for m in cold_models})

dl = F("resnet") - F("rf_theirs"); M["rfLeadLargeMinU"] = f4(dl[2:].min()); M["rfLeadLargeMaxU"] = f4(dl[2:].max())  # the six larger datasets (19k-32k), for the paper's "0.03 to 0.09 improvements over RF"
for m_ in NAME: M["spread" + camel(m_)] = f4(F(m_).std(ddof=1))  # standard deviation of the per-dataset F1 over the 8 datasets
M["nLadder"] = str(len(order))
import glob; M["nCells"] = str(len([f for f in glob.glob(f"{ROOT}/*/*.json") if not f.split("/")[1].startswith("_")]))  # stored result cells (one json each)
# ---------- Port check (E2): shipped weights, confusion counts ----------
E2 = json.load(open(f"{ROOT}/e2_port_results.json")); t2 = E2["models"]["tf_same"]
for k in ("tn", "fp", "fn", "tp"): M["port" + k.upper()] = str(t2[k])
M["portDecisions"] = f"{sum(E2['authors_counts'].values())}"; M["portDiff"] = str(E2["count_l1_difference_tf_same"]); M["portFOne"] = f"{t2['f1']:.4f}"; M["portFOneSymmetric"] = f"{E2['models']['symmetric']['f1']:.3f}"

# ---------- Table: default / tuned x without / with the 64 indicators (the 2x2 that separates configuration from indicators) ----------
cell = lambda m: f"{F(m).mean():.4f}"
with open(f"{OUT}/tab_gain.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Default configuration} & \\multicolumn{2}{c}{Tuned} \\\\\nModel & without & with indicators & without & with indicators \\\\\n\\midrule\n")
    for fam, d0, nm in (("rf", "rf_theirs", "RF"), ("mlp", "mlp_raw_theirs", "MLP"), ("svm", "svm_theirs", "SVM"), ("gb", None, "GB")):
        t0, t1 = F(f"{fam}_tuned"), F(f"{fam}_tuned_eq"); g = t1 - t0; M["gain" + camel(fam)] = sgn(g.mean()); M["gainWins" + camel(fam)] = str(int((g > 0).sum()))
        if d0:
            e0, e1 = F(d0), F(d0 + "_eq"); gd = e1 - e0; M["gainDef" + camel(fam)] = sgn(gd.mean()); M["gainDefWins" + camel(fam)] = str(int((gd > 0).sum()))
            M["cfg" + camel(fam)] = sgn((t0 - e0).mean()); M["cfgInd" + camel(fam)] = sgn((t1 - e1).mean())  # effect of the configuration, without / with the indicators
            f.write(f"{nm} & {cell(d0)} & {cell(d0 + '_eq')} & {cell(f'{fam}_tuned')} & {cell(f'{fam}_tuned_eq')} \\\\\n")
        else: f.write(f"{nm} & -- & -- & {cell(f'{fam}_tuned')} & {cell(f'{fam}_tuned_eq')} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- Figure 1: dot chart of the ladder (pgfplots; colours from a validated categorical palette, marker shapes for grey-scale print) ----------
FIG = {"svm_theirs": "SVM (default)", "mlp_raw_theirs": "MLP (default)", "rf_theirs": "RF (default)", "rf_tuned": "RF (tuned)", "svm_tuned": "SVM (tuned)", "gb_tuned": "GB (tuned)", "mlp_tuned": "MLP (tuned)",
       "resnet": "ResNet (authors)", "resnet_eq": "ResNet + indicators", "resnet_head": "ResNet + hidden layer", "resnet_eq_head": "ResNet + hidden layer + indicators",
       "mlp_tuned_eq": "MLP (tuned) + indicators", "rf_tuned_eq": "RF (tuned) + indicators", "svm_tuned_eq": "SVM (tuned) + indicators", "gb_tuned_eq": "GB (tuned) + indicators",
       "svm_theirs_eq": "SVM (default) + indicators", "mlp_raw_theirs_eq": "MLP (default) + indicators", "rf_theirs_eq": "RF (default) + indicators"}
STYLE = {"default": ("gray", "square*"), "default + indicators": ("seriesViolet", "pentagon*"), "tuned": ("seriesBlue", "*"), "ResNet": ("seriesOrange", "diamond*"), "tuned + indicators": ("seriesAqua", "triangle*")}
rank = sorted(NAME, key=lambda m: F(m).mean()); ref = F("resnet").mean()
with open(f"{OUT}/fig_ladder.tex", "w") as f:
    f.write("% generated by scripts/make_tables.py: do not edit\n\\definecolor{seriesBlue}{HTML}{2A78D6}\\definecolor{seriesOrange}{HTML}{EB6834}\\definecolor{seriesAqua}{HTML}{1BAF7A}\\definecolor{seriesViolet}{HTML}{6250D6}\n")
    f.write("\\begin{tikzpicture}\\begin{axis}[width=0.68\\linewidth, height=11.5cm, xmin=0.74, xmax=0.97, xtick distance=0.04, xlabel={F1 (mean over 8 datasets and 5 seeds)}, symbolic y coords={" + ",".join(FIG[m] for m in rank) + "}, ytick={" + ",".join("{" + FIG[m] + "}" for m in rank) + "}, ytick style={draw=none}, yticklabel style={font=\\small}, xmajorgrids, major grid style={gray!25}, enlarge y limits=0.04, legend style={at={(0.02,0.98)}, anchor=north west, font=\\small, draw=none, fill=white, fill opacity=0.85, text opacity=1}, legend cell align=left]\n")
    f.write(f"\\addplot[black, dashed, thin, forget plot] coordinates {{({ref:.4f},{{{FIG[rank[0]]}}}) ({ref:.4f},{{{FIG[rank[-1]]}}})}};\n")
    for grp, (col, mark) in STYLE.items():
        pts = " ".join(f"({F(m).mean():.4f},{{{FIG[m]}}}) +- ({sd(m):.4f},0)" for m in rank if GROUP[m] == grp)
        f.write(f"\\addplot[only marks, color={col}, mark={mark}, mark size=2.4pt, error bars/.cd, x dir=both, x explicit] coordinates {{{pts}}};\n\\addlegendentry{{{grp}}}\n")
    f.write("\\end{axis}\\end{tikzpicture}\n")

# ---------- Control without equality structure: the authors' real-world amazon1 (8 user fields, 1 resource field, one operation; no indicators) ----------
RA = A.read(ROOT, "h1"); AMS = sorted(k[1] for k in RA if k[0] == "amazon1")
AM = [("resnet", "ResNet (authors' recipe)", "dlbac_r"), ("rf_theirs", "RF, default configuration", "rf"), ("svm_theirs", "SVM, default configuration", "svm"), ("mlp_raw_theirs", "MLP, default configuration", "mlp"),
      ("rf_tuned", "RF, tuned", None), ("gb_tuned", "GB, tuned", None), ("mlp_tuned", "MLP, tuned", None)]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "dlbac_revisited")); import dlbac as D
xa_tr, ya_tr, xa_te, yte, _ = D.load("amazon1"); yte = yte.ravel(); pos = float(yte.mean())  # all-allow reference: precision = share of allows, TPR = FPR = 1, AUC = 0.5
M["nAmazonTuples"] = f"{len(xa_tr) + len(xa_te):,}".replace(",", "{,}"); M["nAmazonTrain"] = f"{len(xa_tr):,}".replace(",", "{,}"); M["nAmazonTest"] = f"{len(xa_te):,}".replace(",", "{,}"); M["amPosRate"] = f"{100 * pos:.1f}"; M["amAllAllowFOne"] = f4(2 * pos / (1 + pos)); M["amSeeds"] = str(len(AMS))
def amv(m, key):  # values over seeds that have the metric (AUC is absent when a model outputs a constant or only hard decisions)
    return [RA[("amazon1", s_)][m][key] for s_ in AMS if m in RA[("amazon1", s_)] and key in RA[("amazon1", s_)][m]]
with open(f"{OUT}/tab_amazon.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nModel & Published F1 & F1 & FPR & TPR & AUC \\\\\n\\midrule\n")
    f.write(f"All-allow reference & -- & {2 * pos / (1 + pos):.4f} & 1.0000 & 1.0000 & 0.5000 \\\\\n\\midrule\n")
    for m, lab, pk in AM:
        v = {k: amv(m, k) for k in ("f1", "fpr", "tpr", "auc")}; assert len(v["f1"]) == len(AMS) > 0, (m, len(AMS))
        pub = f"{T['datasets']['amazon-kaggle'][pk]['f1']:.3f}" if pk else "--"
        auc = f"{np.mean(v['auc']):.4f} ({len(v['auc'])} of {len(AMS)} seeds)" if v["auc"] and len(v["auc"]) < len(AMS) else (f"{np.mean(v['auc']):.4f}" if v["auc"] else "--")
        f.write(f"{lab} & {pub} & {np.mean(v['f1']):.4f} & {np.mean(v['fpr']):.4f} & {np.mean(v['tpr']):.4f} & {auc} \\\\\n")
        for k in ("f1", "fpr", "tpr", "auc"):
            if v[k]: M["am" + camel(m) + {"f1": "FOne"}.get(k, k.capitalize())] = f4(np.mean(v[k]))
        if v["auc"]: M["am" + camel(m) + "AucSd"] = f4(np.std(v["auc"], ddof=1) if len(v["auc"]) > 1 else 0.0)
    f.write("\\bottomrule\n\\end{tabular}\n")
print("amazon1 control:", {m: (f4(np.mean(amv(m, 'f1'))), f4(np.mean(amv(m, 'auc'))) if amv(m, 'auc') else None) for m, *_ in AM}, "| all-allow F1", M["amAllAllowFOne"])

# ---------- Appendix: per-dataset tables for the four metrics ----------
for key, title in (("f1", "F1"), ("fpr", "FPR"), ("tpr", "TPR"), ("precision", "precision")):
    with open(f"{OUT}/tab_app_{key}.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "r" * 9 + "}\n\\toprule\nModel & " + " & ".join(d.split("auth")[1] for d in DS) + " & mean \\\\\n\\midrule\n")
        for m in sorted(NAME, key=lambda m: -F(m).mean()): f.write(f"{NAME[m]} & " + " & ".join(f"{x:.3f}" for x in F(m, key)) + f" & {F(m, key).mean():.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

for k, v in list(M.items()):  # unsigned twin (suffix U) of every signed number, for sentences like "by X to Y"
    if v[:1] in "+-": M[k + "U"] = v[1:]
with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("% generated by scripts/make_tables.py: do not edit\n")
    for k, v in sorted(M.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros and 8 tables/csv to {OUT}/")
