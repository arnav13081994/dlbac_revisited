"""Experiments on the authors' own datasets (analysis plan: docs/analysis_plan.md). One job = one (experiment, dataset, seed, model group); it writes <out>.json (metrics) and <out>.npz (per-decision probabilities and predictions).
  main   : the authors' own train/test files.                groups: resnet | resnet_eq | resnet_head | resnet_eq_head | theirs | theirs_eq | tuned | svm  (--ds amazon1 also works: one resource field and one operation, so no *_eq groups, no svm; depth-8 ResNet, binary encoding of the metadata as in the paper)
  cold   : E3, entity hold-out (warm / user-cold / resource-cold / both-cold test sets); --seed is the split seed AND the model seed
  layout : E4, column / field permutations of the one-hot grid (their ResNet-8 and the paper's MLP configuration).   --cond none|same|user|fields  --model resnet|mlp
"Their model" = the released recipe (depth 8 for the two datasets with 8+8 metadata, else 56; epochs, batch, schedule, L2 as in dlbac.recipe) in PyTorch with TensorFlow 'same' padding (validated by src/dlbac_revisited/port_check.py).
resnet_eq = our addition: the 64 indicators 1[user field i == resource field j] are concatenated to the pooled 64-d vector before the output layer. tuned = RF / GB / one-hot MLP, each without and with the indicators, 12-config grid chosen on a 20% validation split of train.
usage: python -I src/dlbac_revisited/run_cell.py main --ds u4k-r4k-auth11k --seed 0 --group tuned --out results/e1/u4k-r4k-auth11k_s0_tuned.json [--threads 1] [--device cpu|mps] [--tiny]"""
import argparse, json, os, sys, time, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")  # before numpy/sklearn load: sklearn's OpenMP (HistGradientBoosting) ignores --threads, and with several jobs in parallel the spinning threads made one fit about 20x slower
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dlbac as D
warnings.filterwarnings("ignore")
M = 8
eqf = lambda X: (X[:, :M, None] == X[:, None, M:]).reshape(len(X), -1).astype(np.float32)  # 1[user field i == resource field j], all 64 pairs of the visible fields
flat = lambda E: E.reshape(len(E), -1).astype(np.float32)
GRIDS = {"rf": [dict(n_estimators=a, max_features=b, min_samples_leaf=c) for a in (200, 500) for b in ("sqrt", 0.5, 1.0) for c in (1, 2)],
         "gb": [dict(learning_rate=a, max_leaf_nodes=b, l2_regularization=c) for a in (0.05, 0.1, 0.2) for b in (31, 63) for c in (0.0, 1.0)],
         "mlp": [dict(hidden_layer_sizes=a, activation=b, alpha=c) for a in ((128,) * 4, (256, 128), (64, 64)) for b in ("relu", "tanh") for c in (1e-4, 1e-2)],
         "svm": [dict(C=a, g=b) for a in (1, 10, 100) for b in (0.5, 1, 2, 4)]}  # follow-up E6: RBF SVM on one-hot, gamma = g / (mean non-zeros per row)

def fit_predict(kind, cfg, Fa, Ya, Fb, seed, threads):
    """probabilities (n, ops) of one classical model family; cfg = one grid entry."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    if Ya.shape[1] == 1 and kind == "rf": return RandomForestClassifier(random_state=seed, n_jobs=threads, **cfg).fit(Fa, Ya.ravel()).predict_proba(Fb)[:, 1:]  # single-operation data (amazon1): 1-d target, (n, 2) probabilities
    if Ya.shape[1] == 1 and kind == "mlp": return MLPClassifier(random_state=seed, **cfg).fit(Fa, Ya.ravel()).predict_proba(Fb)[:, 1:]
    if kind == "rf": m = RandomForestClassifier(random_state=seed, n_jobs=threads, **cfg).fit(Fa, Ya); return np.stack([p[:, -1] if p.shape[1] > 1 else np.zeros(len(Fb)) for p in m.predict_proba(Fb)], 1)
    if kind == "gb": return np.stack([HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=seed, **cfg).fit(Fa, Ya[:, o]).predict_proba(Fb)[:, -1] for o in range(Ya.shape[1])], 1)
    if kind == "svm":  # sparse one-hot (+eq) rows have 16-80 non-zeros, so libsvm's cost follows the non-zeros, not the 5k+ columns; hard 0/1 decisions, as for svm_theirs
        from sklearn.svm import SVC
        gam = cfg["g"] / (Fa.nnz / Fa.shape[0])
        return np.stack([SVC(C=cfg["C"], gamma=gam).fit(Fa, Ya[:, o]).predict(Fb) for o in range(Ya.shape[1])], 1).astype(float)
    return MLPClassifier(random_state=seed, **cfg).fit(Fa, Ya).predict_proba(Fb)

def tuned(kind, Fa, Ya, Fb, seed, threads, tiny):
    grid = GRIDS[kind][:2] if tiny else GRIDS[kind]; idx = np.random.default_rng(seed).permutation(Fa.shape[0]); k = len(idx) // 5; va, tr = idx[:k], idx[k:]  # 20% validation split of train, then refit on all of train
    sc = [D.metrics(fit_predict(kind, g, Fa[tr], Ya[tr], Fa[va], seed, threads), Ya[va])["f1"] for g in grid]; best = grid[int(np.argmax(sc))]
    return fit_predict(kind, best, Fa, Ya, Fb, seed, threads), {"best": {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in best.items()}, "val_f1": [round(float(s), 4) for s in sc]}

def train_models(group, ds, seed, Xtr, Ytr, Etr, Xte, Ete, a):
    """returns {model name: (probabilities on the test rows, extra info)} for one group."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    out = {}; t0 = time.time(); EQS = (False,) if ds.startswith("amazon") else (False, True)  # amazon1 has one resource field, so user-resource equality indicators are not defined there
    if group in ("resnet", "resnet_eq", "resnet_head", "resnet_eq_head"):  # *_head (follow-up E5): output layer = Linear(64 + 64 eq, 128) - ReLU - Linear, with or without the indicators
        eq = group.startswith("resnet_eq"); depth = 8 if (a.tiny or ds in D.SYNTHETIC[:2] or ds.startswith("amazon")) else 56  # released rule: depth 8 when no metadata are hidden
        if eq and ds.startswith("amazon"): raise SystemExit("equality indicators are not defined for amazon (one resource field)")
        net, hist = D.train_resnet(Etr, Ytr, Ete, depth, seed, a.device, 2 if a.tiny else None, tf_same=True, ex_tr=eqf(Xtr) if eq else None, ex_te=eqf(Xte) if eq else None, hidden=128 if group.endswith("_head") else 0)
        out[group] = (D.predict(net, Ete, a.device, ex=eqf(Xte) if eq else None), {"depth": depth, "epochs": len(hist), "final_train_loss": hist[-1]["train_loss"], "seconds": round(time.time() - t0)})
    elif group in ("theirs", "theirs_eq"):  # the paper's baselines in their configuration; theirs_eq appends the 64 indicators to the integer codes
        sfx = group[len("theirs"):]
        if sfx: Xtr, Xte = np.hstack([Xtr, eqf(Xtr)]), np.hstack([Xte, eqf(Xte)])
        one = Ytr.shape[1] == 1; Yf = Ytr.ravel() if one else Ytr  # single-operation data (amazon1): sklearn wants a 1-d target and returns (n, 2) probabilities
        pr = RandomForestClassifier(random_state=seed, n_jobs=a.threads).fit(Xtr, Yf).predict_proba(Xte)
        out["rf_theirs" + sfx] = (pr[:, 1:] if one else np.stack([p[:, -1] if p.shape[1] > 1 else np.zeros(len(Xte)) for p in pr], 1), {})
        sc = StandardScaler().fit(Xtr); out["svm_theirs" + sfx] = (np.stack([SVC().fit(sc.transform(Xtr), Ytr[:, o]).predict(sc.transform(Xte)) for o in range(Ytr.shape[1])], 1).astype(float), {})
        pm = MLPClassifier((128,) * 4, random_state=seed).fit(Xtr, Yf).predict_proba(Xte); out["mlp_raw_theirs" + sfx] = (pm[:, 1:] if one else pm, {})
    elif group == "tuned":
        for kind in ("rf", "gb", "mlp"):
            for eq in EQS:
                Fa, Fb = (flat(Etr), flat(Ete)) if kind == "mlp" else (Xtr, Xte)
                if eq: Fa, Fb = np.hstack([Fa, eqf(Xtr)]), np.hstack([Fb, eqf(Xte)])
                P, info = tuned(kind, Fa, Ytr, Fb, seed, a.threads, a.tiny); info["seconds"] = round(time.time() - t0); out[f"{kind}_tuned" + ("_eq" if eq else "")] = (P, info)
    elif group == "svm":  # follow-up E6: tuned RBF SVM on one-hot input, without and with the 64 indicators
        import scipy.sparse as sp
        for eq in EQS:
            Fa, Fb = sp.csr_matrix(flat(Etr)), sp.csr_matrix(flat(Ete))
            if eq: Fa, Fb = sp.hstack([Fa, sp.csr_matrix(eqf(Xtr))], "csr"), sp.hstack([Fb, sp.csr_matrix(eqf(Xte))], "csr")
            P, info = tuned("svm", Fa, Ytr, Fb, seed, a.threads, a.tiny); info["seconds"] = round(time.time() - t0); out["svm_tuned" + ("_eq" if eq else "")] = (P, info)
    else: raise SystemExit(f"unknown group {group}")
    return out

def save(path, meta, Y, outs, sets=None):
    """json: metrics per model (per test set when `sets` is given); npz: per-decision probabilities and predictions."""
    from sklearn.metrics import roc_auc_score
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True); res = dict(meta); res["models"] = {}; arrs = {"Y": Y.astype(np.uint8)}
    if sets is not None: arrs["set_id"] = sets["id"].astype(np.int8)
    for name, (P, info) in outs.items():
        arrs[name + "__prob"] = np.asarray(P, np.float32); arrs[name + "__pred"] = (np.asarray(P) >= .5).astype(np.uint8)
        if sets is None: res["models"][name] = dict(D.metrics(P, Y), **info, **({"auc": float(roc_auc_score(Y.ravel(), np.asarray(P).ravel()))} if Y.shape[1] == 1 and len(np.unique(P)) > 2 else {}))  # AUC only for single-operation data (amazon1) and for models that output probabilities
        else: res["models"][name] = dict(info, per_set={nm: (D.metrics(P[sets["id"] == i], Y[sets["id"] == i]) if sets["ok"][i] else None) for i, nm in enumerate(sets["names"])})
    np.savez_compressed(path[:-5] + ".npz", **arrs); tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, path)  # json written last and atomically: jobqueue.py treats an existing --out file as 'job finished', so a half-written file must never exist
    for name, v in res["models"].items(): print(name, {k: round(x, 4) for k, x in v.items() if k in ("f1", "fpr", "tpr", "auc")} if sets is None else {s: (round(m["f1"], 4) if m else None) for s, m in v["per_set"].items()}, flush=True)

def cold_split(seed, uid, rid):
    """E3: hold out 20% of user ids and 20% of resource ids; train on tuples with neither held out minus a 15% warm test; test sets warm / user-cold / resource-cold / both-cold."""
    rng = np.random.default_rng(seed); uh = rng.choice(np.unique(uid), int(round(.2 * len(np.unique(uid)))), replace=False); rh = rng.choice(np.unique(rid), int(round(.2 * len(np.unique(rid)))), replace=False)
    u_h, r_h = np.isin(uid, uh), np.isin(rid, rh); pool = np.flatnonzero(~u_h & ~r_h); perm = rng.permutation(pool); nw = int(round(.15 * len(pool)))
    warm, train = perm[:nw], perm[nw:]; uc, rc, bc = np.flatnonzero(u_h & ~r_h), np.flatnonzero(~u_h & r_h), np.flatnonzero(u_h & r_h)
    assert not (set(uid[train]) & set(uid[np.r_[uc, bc]])) and not (set(rid[train]) & set(rid[np.r_[rc, bc]])), "held-out entity leaked into training"
    assert len(set(train) | set(warm) | set(uc) | set(rc) | set(bc)) == len(train) + len(warm) + len(uc) + len(rc) + len(bc), "split overlap"
    return train, [("warm", warm), ("user_cold", uc), ("resource_cold", rc), ("both_cold", bc)]

def permute(E, cond, seed):
    """E4: permutations of the one-hot grid E (N, 16, V): none | same (one value-column permutation for every row) | user (independent permutations for user rows and resource rows) | fields (one permutation of the 16 field rows)."""
    rng = np.random.default_rng(10_000 + seed); V = E.shape[2]
    if cond == "none": return E
    if cond == "same": return E[:, :, rng.permutation(V)]
    if cond == "user":
        pu = rng.permutation(V); pr = rng.permutation(V)
        while (pu == pr).all(): pr = rng.permutation(V)
        return np.concatenate([E[:, :M][:, :, pu], E[:, M:][:, :, pr]], 1)
    if cond == "fields": return E[:, rng.permutation(E.shape[1]), :]
    raise SystemExit(f"unknown cond {cond}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("exp", choices=["main", "cold", "layout"]); p.add_argument("--ds", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group", default="resnet"); p.add_argument("--cond", default="none"); p.add_argument("--model", default="resnet"); p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--device", default="cpu", help="the stored results were produced with mps; cpu and cuda are untested for the full runs"); p.add_argument("--tiny", action="store_true", help="smoke test: 400 training rows, 2 epochs, 2 grid entries, depth 8"); a = p.parse_args()
    import torch; torch.set_num_threads(a.threads)
    Xtr, Ytr, Xte, Yte, kind = D.load(a.ds); meta = dict(exp=a.exp, ds=a.ds, seed=a.seed, group=a.group, tiny=a.tiny, protocol="paper (released-code columns)", torch=torch.__version__)
    if a.exp == "main":
        if a.tiny: Xtr, Ytr, Xte, Yte = Xtr[:400], Ytr[:400], Xte[:200], Yte[:200]
        Etr, Ete = D.encode(Xtr, Xte, kind); outs = train_models(a.group, a.ds, a.seed, Xtr, Ytr, Etr, Xte, Ete, a); meta.update(n_train=len(Xtr), n_test=len(Xte)); save(a.out, meta, Yte, outs)
    elif a.exp == "cold":
        tr_, te_ = D.rd(f"{D.ROOT}/synthetic/{a.ds}/train_{a.ds}.sample"), D.rd(f"{D.ROOT}/synthetic/{a.ds}/test_{a.ds}.sample")
        X, Y = np.vstack([Xtr, Xte]), np.vstack([Ytr, Yte]); uid, rid = np.r_[tr_[:, 0], te_[:, 0]], np.r_[tr_[:, 1], te_[:, 1]]
        train, tests = cold_split(a.seed, uid, rid)
        if a.tiny: train = train[:400]; tests = [(n, i[:60]) for n, i in tests]
        ti = np.concatenate([i for _, i in tests]); sid = np.concatenate([np.full(len(i), k) for k, (_, i) in enumerate(tests)])
        Etr, Ete = D.encode(X[train], X[ti], kind); outs = train_models(a.group, a.ds, a.seed, X[train], Y[train], Etr, X[ti], Ete, a)
        ok = [(sid == k).sum() >= 50 and 0 < Y[ti][sid == k].mean() < 1 for k in range(len(tests))]; meta.update(n_train=len(train), set_sizes={n: int(len(i)) for n, i in tests}, set_ok=[bool(o) for o in ok])
        save(a.out, meta, Y[ti], outs, sets={"id": sid, "names": [n for n, _ in tests], "ok": ok})
    else:
        if a.tiny: Xtr, Ytr, Xte, Yte = Xtr[:400], Ytr[:400], Xte[:200], Yte[:200]
        Etr, Ete = D.encode(Xtr, Xte, kind); Etr, Ete = permute(Etr, a.cond, a.seed), permute(Ete, a.cond, a.seed); meta.update(cond=a.cond, model=a.model, n_train=len(Xtr), n_test=len(Xte))
        if a.model == "resnet":
            net, hist = D.train_resnet(Etr, Ytr, Ete, 8, a.seed, a.device, 2 if a.tiny else None, tf_same=True); outs = {"resnet8": (D.predict(net, Ete, a.device), {"depth": 8, "epochs": len(hist), "final_train_loss": hist[-1]["train_loss"]})}
        else:
            from sklearn.neural_network import MLPClassifier
            outs = {"mlp_onehot": (MLPClassifier((128,) * 4, random_state=a.seed).fit(flat(Etr), Ytr).predict_proba(flat(Ete)), {})}
        save(a.out, meta, Yte, outs)
