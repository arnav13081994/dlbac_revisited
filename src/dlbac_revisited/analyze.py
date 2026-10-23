"""Analysis of experiments E1 (authors' split), E3 (cold splits), E4 (layout), E2 (port check) and the follow-ups E5 and E6; implements the tests fixed in docs/analysis_plan.md.
Unit = dataset (mean over seeds / split seeds); difference = ResNet minus comparator; 95% bootstrap CI over datasets (10,000 resamples); paired t; equivalence by two one-sided tests (TOST) at the pre-specified margin; Holm adjustment within each pre-listed family.
Verdicts: equivalent = Holm TOST p < .05; ResNet better / comparator better = Holm paired-t p < .05 with that sign; non-inferior = Holm one-sided p (ResNet not better by more than the margin) < .05.
usage: python -I src/dlbac_revisited/analyze.py [--root results] [--exp e1|e3|e4|e5|e6|all]  -> prints and writes <root>/REPORT_<exp>.txt"""
import argparse, glob, json, os, sys
import numpy as np
from scipy import stats
DS = ["u4k-r4k-auth11k", "u5k-r5k-auth12k", "u5k-r5k-auth19k", "u4k-r4k-auth21k", "u4k-r7k-auth20k", "u4k-r4k-auth22k", "u4k-r6k-auth28k", "u6k-r6k-auth32k"]
E1_FAMILY = ["rf_theirs", "svm_theirs", "mlp_raw_theirs", "rf_tuned", "gb_tuned", "mlp_tuned", "rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq"]  # ResNet minus each; the 10th test is resnet_eq minus resnet
E3_AUG = ["rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq"]; COLD = ["user_cold", "resource_cold", "both_cold"]

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.)
    return adj

def one(d, margin, rng):
    d = np.asarray(d, float); n = len(d); mean = d.mean(); se = max(d.std(ddof=1) / np.sqrt(n), 1e-12); boot = np.array([rng.choice(d, n).mean() for _ in range(10_000)]); lo, hi = np.percentile(boot, [2.5, 97.5])
    p_low, p_high = 1 - stats.t.cdf((mean + margin) / se, n - 1), stats.t.cdf((mean - margin) / se, n - 1)
    return dict(mean=mean, lo=lo, hi=hi, wins=int((d > 0).sum()), n=n, p_diff=float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0, p_high=p_high, p_tost=max(p_low, p_high))

def family(diffs, margin, sens=(0.5, 2.0), better=("ResNet", "comparator")):
    """diffs: {label: per-dataset differences}; returns printable rows with Holm-adjusted verdicts (+ equivalence verdict at sensitivity margins)."""
    rng = np.random.default_rng(0); R = {k: one(v, margin, rng) for k, v in diffs.items()}; ks = list(R)
    ad, at, ah = holm([R[k]["p_diff"] for k in ks]), holm([R[k]["p_tost"] for k in ks]), holm([R[k]["p_high"] for k in ks])
    alt = {s: holm([one(diffs[k], margin * s, rng)["p_tost"] for k in ks]) for s in sens}; rows = []
    for i, k in enumerate(ks):
        r = R[k]; v = "equivalent" if at[i] < .05 else (f"{better[0]} better" if r["mean"] > 0 else f"{better[1]} better") if ad[i] < .05 else "inconclusive"
        rows.append(f"{k:34s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['wins']}/{r['n']}  p_diff(Holm) {ad[i]:.3f}  TOST(Holm) {at[i]:.3f}  non-inferior {'yes' if ah[i] < .05 else 'no '}  {v:22s} equiv@x{sens[0]} {'Y' if alt[sens[0]][i] < .05 else 'n'} @x{sens[1]} {'Y' if alt[sens[1]][i] < .05 else 'n'}")
    return rows

def read(root, *exps):
    """{(ds, seed): {model: record}} merged over the model groups of one or more result folders (e1, e3, or the follow-up folders f1, f3 on the same protocol)."""
    out = {}
    for exp in exps:
        for f in sorted(glob.glob(f"{root}/{exp}/*.json")):
            try: d = json.load(open(f)); out.setdefault((d["ds"], d["seed"]), {}).update(d["models"])
            except (ValueError, KeyError, OSError) as e: print(f"skipping unreadable result file {f}: {e!r}", file=sys.stderr)
    return out

def ds_mean(R, model, getter):
    """per-dataset mean over seeds of getter(record); NaN when a dataset has no run."""
    return np.array([np.nanmean([getter(R[k][model]) for k in R if k[0] == ds and model in R[k] and getter(R[k][model]) is not None] or [np.nan]) for ds in DS])

def e1(root):
    R = read(root, "e1"); L = []; f1 = lambda r: r["f1"]
    models = sorted({m for v in R.values() for m in v}); seeds = {ds: sorted({k[1] for k in R if k[0] == ds}) for ds in DS}
    L.append(f"E1: {len(R)} (dataset, seed) cells; seeds per dataset: {[len(seeds[d]) for d in DS]}; models: {models}")
    for key, name in (("f1", "F1"), ("fpr", "FPR"), ("tpr", "TPR"), ("precision", "precision")):
        L.append(f"\n{name}, mean over seeds, paper protocol"); L.append(f"{'dataset':18s} " + " ".join(f"{m[:13]:>13s}" for m in models))
        for i, ds in enumerate(DS): L.append(f"{ds:18s} " + " ".join(f"{ds_mean(R, m, lambda r: r[key])[i]:13.3f}" for m in models))
        L.append(f"{'mean':18s} " + " ".join(f"{np.nanmean(ds_mean(R, m, lambda r: r[key])):13.3f}" for m in models))
    F = {m: ds_mean(R, m, f1) for m in models}; ok = lambda m: m in F and not np.isnan(F[m]).any()
    if ok("resnet") and ok("resnet_eq"):
        diffs = {f"ResNet minus {m}": F["resnet"] - F[m] for m in E1_FAMILY if ok(m)}; diffs["ResNet+eq minus ResNet"] = F["resnet_eq"] - F["resnet"]
        L.append(f"\nPRE-SPECIFIED FAMILY ({len(diffs)} comparisons, Holm), margin 0.01 F1 [difference, 95% bootstrap CI, ResNet wins, tests]"); L += family(diffs, 0.01)
    else: L.append("\n(family not computed: need ResNet and ResNet+eq on all 8 datasets)")
    return L

def e3(root):
    R = read(root, "e3"); L = [f"E3: {len(R)} (dataset, split seed) cells"]; models = sorted({m for v in R.values() for m in v})
    get = lambda s: (lambda r: (r["per_set"].get(s) or {}).get("f1"))
    for s in ["warm"] + COLD:
        L.append(f"\nF1 on the {s} test set, mean over split seeds"); L.append(f"{'dataset':18s} " + " ".join(f"{m[:13]:>13s}" for m in models))
        for i, ds in enumerate(DS): L.append(f"{ds:18s} " + " ".join(f"{ds_mean(R, m, get(s))[i]:13.3f}" for m in models))
        L.append(f"{'mean':18s} " + " ".join(f"{np.nanmean(ds_mean(R, m, get(s))):13.3f}" for m in models))
    F = {(m, s): ds_mean(R, m, get(s)) for m in models for s in ["warm"] + COLD}; diffs = {}
    for s in COLD:
        for m in E3_AUG:
            if (m, s) in F and (("resnet", s) in F) and not (np.isnan(F[(m, s)]).any() or np.isnan(F[("resnet", s)]).any()): diffs[f"{s}: ResNet minus {m}"] = F[("resnet", s)] - F[(m, s)]
    for s in COLD:
        if ("resnet_eq", s) in F and ("resnet", s) in F and not (np.isnan(F[("resnet_eq", s)]).any() or np.isnan(F[("resnet", s)]).any()): diffs[f"{s}: ResNet+eq minus ResNet"] = F[("resnet_eq", s)] - F[("resnet", s)]
    if diffs: L.append(f"\nPRE-SPECIFIED FAMILY ({len(diffs)} comparisons, Holm), margin 0.02 F1 (sensitivity x0.5 = 0.01, x1.5 = 0.03)"); L += family(diffs, 0.02, sens=(0.5, 1.5))
    return L

def e4(root):
    L = ["E4: layout ablation, F1 mean over seeds [95% bootstrap CI over seeds of the difference to 'none']"]; rng = np.random.default_rng(0); by = {}
    for f in glob.glob(f"{root}/e4/*.json"):
        try:
            d = json.load(open(f))
            for model, r in d["models"].items(): by.setdefault((d["ds"], model), {}).setdefault(d["cond"], {})[d["seed"]] = r["f1"]
        except (ValueError, KeyError, OSError) as e: print(f"skipping unreadable result file {f}: {e!r}", file=sys.stderr)
    for (ds, model) in sorted(by):
        c_ = by[(ds, model)]
        if "none" not in c_: continue
        for c in ("none", "same", "user", "fields"):
            if c not in c_: continue
            sd = sorted(set(c_[c]) & set(c_["none"])); d = np.array([c_[c][s] - c_["none"][s] for s in sd]); b = np.array([rng.choice(d, len(d)).mean() for _ in range(10_000)]) if len(d) else np.array([np.nan])
            L.append(f"{ds:18s} {model:12s} {c:7s} F1 {np.mean(list(c_[c].values())):.4f} (n={len(c_[c])})  difference to none {d.mean() if len(d) else float('nan'):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
    return L

def e5(root):
    """Follow-up E5 (designed after E1-E4 were read): the ResNet's output layer gets a hidden layer of 128 ReLU units, with the 64 indicators (resnet_eq_head, 'ResNet+eqH') and without (resnet_head, 'ResNetH', the capacity control). Authors' split: family of 6, margin 0.01; cold splits: family of 12, margin 0.02; Holm within each."""
    ms = ["resnet", "resnet_head", "resnet_eq", "resnet_eq_head", "rf_tuned_eq", "gb_tuned_eq", "mlp_tuned_eq"]; nm = (("RF+eq", "rf_tuned_eq"), ("GB+eq", "gb_tuned_eq"), ("MLP+eq", "mlp_tuned_eq"))
    R = read(root, "e1", "f1"); F = {m: ds_mean(R, m, lambda r: r["f1"]) for m in ms}; L = ["E5, authors' split: mean F1 over datasets: " + ", ".join(f"{m} {np.nanmean(F[m]):.4f}" for m in ms)]
    if all(not np.isnan(F[m]).any() for m in ms):
        d = {"ResNet+eqH minus ResNet": F["resnet_eq_head"] - F["resnet"], "ResNet+eqH minus ResNetH": F["resnet_eq_head"] - F["resnet_head"], "ResNetH minus ResNet": F["resnet_head"] - F["resnet"]}
        d.update({f"{n} minus ResNet+eqH": F[m] - F["resnet_eq_head"] for n, m in nm}); L.append(f"\nFAMILY ({len(d)} comparisons, Holm), margin 0.01 F1 [difference, 95% bootstrap CI, first-named wins]"); L += family(d, 0.01, better=("first-named", "second-named"))
    else: L.append("(authors'-split family not computed: need every model on all 8 datasets)")
    Rc = read(root, "e3", "f3"); get = lambda s: (lambda r: (r["per_set"].get(s) or {}).get("f1")); G = {(m, s): ds_mean(Rc, m, get(s)) for m in ms for s in COLD}; d = {}
    L.append("\nE5, cold splits: mean F1 over datasets: " + "; ".join(f"{s}: " + ", ".join(f"{m} {np.nanmean(G[(m, s)]):.4f}" for m in ("resnet", "resnet_eq_head", "gb_tuned_eq")) for s in COLD))
    for s in COLD:
        for n, m in nm + (("ResNet", "resnet"),):
            if not (np.isnan(G[("resnet_eq_head", s)]).any() or np.isnan(G[(m, s)]).any()): d[f"{s}: ResNet+eqH minus {n}"] = G[("resnet_eq_head", s)] - G[(m, s)]
    if d: L.append(f"\nFAMILY, cold splits ({len(d)} comparisons, Holm), margin 0.02 F1 (sensitivity x0.5 = 0.01, x1.5 = 0.03)"); L += family(d, 0.02, sens=(0.5, 1.5), better=("first-named", "second-named"))
    d = {}  # E5c (amendment 8, written after the authors'-split E5 result): head-only control on the cold splits
    for s in COLD:
        a_, h_, r_ = G[("resnet_eq_head", s)], G[("resnet_head", s)], G[("resnet", s)]
        if not (np.isnan(a_).any() or np.isnan(h_).any()): d[f"{s}: ResNet+eqH minus ResNetH"] = a_ - h_
        if not (np.isnan(h_).any() or np.isnan(r_).any()): d[f"{s}: ResNetH minus ResNet"] = h_ - r_
    if d: L.append(f"\nE5c FAMILY, cold splits ({len(d)} comparisons, Holm), margin 0.02 F1, head-only control (sensitivity x0.5 = 0.01, x1.5 = 0.03)"); L += family(d, 0.02, sens=(0.5, 1.5), better=("first-named", "second-named"))
    return L

def e6(root):
    """Follow-up E6 (designed after E1-E4 were read): tuned RBF SVM on one-hot input, without (svm_tuned) and with (svm_tuned_eq) the 64 indicators; the paper's SVM (svm_theirs) is shown for reference. Authors' split: ResNet minus each tuned SVM, margin 0.01, Holm over 2; cold splits: ResNet minus each, margin 0.02, Holm over 6."""
    ms = ["resnet", "svm_theirs", "svm_tuned", "svm_tuned_eq", "mlp_tuned", "mlp_tuned_eq"]; R = read(root, "e1", "f1"); F = {m: ds_mean(R, m, lambda r: r["f1"]) for m in ms}
    L = ["E6, authors' split: mean F1 over datasets: " + ", ".join(f"{m} {np.nanmean(F[m]):.4f}" for m in ms)]; d = {f"ResNet minus {m}": F["resnet"] - F[m] for m in ("svm_tuned", "svm_tuned_eq") if not (np.isnan(F[m]).any() or np.isnan(F["resnet"]).any())}
    if d: L.append(f"\nFAMILY ({len(d)} comparisons, Holm), margin 0.01 F1 [difference, 95% bootstrap CI, ResNet wins]"); L += family(d, 0.01)
    Rc = read(root, "e3", "f3"); get = lambda s: (lambda r: (r["per_set"].get(s) or {}).get("f1")); G = {(m, s): ds_mean(Rc, m, get(s)) for m in ms for s in COLD}; d = {}
    L.append("\nE6, cold splits: mean F1 over datasets: " + "; ".join(f"{s}: " + ", ".join(f"{m} {np.nanmean(G[(m, s)]):.4f}" for m in ("resnet", "svm_tuned", "svm_tuned_eq")) for s in COLD))
    for s in COLD:
        for m in ("svm_tuned", "svm_tuned_eq"):
            if not (np.isnan(G[("resnet", s)]).any() or np.isnan(G[(m, s)]).any()): d[f"{s}: ResNet minus {m}"] = G[("resnet", s)] - G[(m, s)]
    if d: L.append(f"\nFAMILY, cold splits ({len(d)} comparisons, Holm), margin 0.02 F1 (sensitivity x0.5 = 0.01, x1.5 = 0.03)"); L += family(d, 0.02, sens=(0.5, 1.5))
    return L

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", default="results"); p.add_argument("--exp", default="all"); a = p.parse_args()
    for ex, fn, folder in (("e1", e1, "e1"), ("e3", e3, "e3"), ("e4", e4, "e4"), ("e5", e5, "f1"), ("e6", e6, "f1")):
        if a.exp in (ex, "all") and glob.glob(f"{a.root}/{folder}/*.json"):
            L = fn(a.root); open(f"{a.root}/REPORT_{ex}.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L) + "\n")
    e2 = f"{a.root}/e2_port_results.json"
    if a.exp in ("e2", "all") and os.path.exists(e2): d = json.load(open(e2)); print("E2:", {k: d[k] for k in d if k.startswith(("count_l1", "pass", "agreement"))})
