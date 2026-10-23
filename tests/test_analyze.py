"""Self-check of analyze.py (e5/e6) on FAKE results with planted effects: the reported differences must match the planted offsets and the verdict logic must react. usage: python -I tests/test_analyze.py"""
import json, os, re, sys, tempfile
import numpy as np
sys.path.insert(0, "src/dlbac_revisited"); import analyze as S
rng = np.random.default_rng(0); root = tempfile.mkdtemp(); base = dict(zip(S.DS, np.linspace(.86, .99, 8)))
OFF = {"resnet": 0, "resnet_eq": 0, "resnet_head": .002, "resnet_eq_head": .015, "rf_tuned_eq": .009, "gb_tuned_eq": .015, "mlp_tuned_eq": .008, "mlp_tuned": .002, "svm_theirs": -.15, "svm_tuned": -.004, "svm_tuned_eq": .010}
NEW = {"resnet_head", "resnet_eq_head", "svm_tuned", "svm_tuned_eq"}
def write(folder, ds, seed, models, cold):
    os.makedirs(f"{root}/{folder}", exist_ok=True); rec = {}
    for m in models:
        f = lambda: float(base[ds] + OFF[m] + rng.normal(0, .001))
        rec[m] = {"per_set": {s: {"f1": f()} for s in ["warm"] + S.COLD}} if cold else {"f1": f()}
    json.dump({"ds": ds, "seed": seed, "models": rec}, open(f"{root}/{folder}/{ds}_s{seed}_{'x'.join(sorted(models))}.json", "w"))
for ds in S.DS:
    for seed in range(3):
        for cold, old, new in ((False, "e1", "f1"), (True, "e3", "f3")):
            write(old, ds, seed, [m for m in OFF if m not in NEW], cold); write(new, ds, seed, sorted(NEW), cold)
def rows(lines):
    return {m.group(1).strip(): float(m.group(2)) for l in lines if (m := re.match(r"^(.*?)\s+([+-]\d\.\d{4}) \[", l))}
r5, r6 = rows(S.e5(root)), rows(S.e6(root))
want = {"ResNet+eqH minus ResNet": .015, "ResNet+eqH minus ResNetH": .013, "ResNetH minus ResNet": .002, "GB+eq minus ResNet+eqH": 0., "RF+eq minus ResNet+eqH": -.006, "MLP+eq minus ResNet+eqH": -.007,
        "both_cold: ResNet+eqH minus GB+eq": 0., "user_cold: ResNet+eqH minus ResNet": .015, "both_cold: ResNet+eqH minus ResNetH": .013, "resource_cold: ResNetH minus ResNet": .002}
want6 = {"ResNet minus svm_tuned": .004, "ResNet minus svm_tuned_eq": -.010, "both_cold: ResNet minus svm_tuned_eq": -.010}
for got, w in ((r5, want), (r6, want6)):
    for k, v in w.items(): assert k in got and abs(got[k] - v) < .0015, (k, got.get(k), v)
fam = [l for l in S.e5(root) if l.startswith("ResNet+eqH minus ResNet ")][0]; assert "first-named better" in fam, fam   # +0.015 against margin 0.01 over 8 datasets must be significant
print("selfcheck OK: planted effects recovered in E5 (%d rows) and E6 (%d rows)" % (len(r5), len(r6)))
