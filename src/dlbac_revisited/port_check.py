"""E2 port validation: load the authors' shipped Keras weights (Keras 2.5 HDF5) into our PyTorch ResNet and evaluate their test file.
Compared with the authors' own logged result (neural_network_training/results/result.txt: pooled TN/FP/FN/TP on the 11k test file). Their repository ships no per-decision predictions,
so the pass criterion is on the confusion counts; as a control the SAME weights are also evaluated with torch's symmetric padding (torch's default), which shows how much the padding matters.
usage: python -I src/dlbac_revisited/port_check.py --weights external/DlbacAlpha/neural_network_training/results/dlbac_alpha.hdf5 --dataset u4k-r4k-auth11k --depth 8 --expect 4746,87,50,3885 --out results/e2_port_results.json   (needs h5py: requirements-port.txt)"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py, numpy as np, torch.nn as nn, torch
import dlbac as D

def load_keras(net, path):
    mw = h5py.File(path, "r")["model_weights"]; dec = lambda x: x.decode() if isinstance(x, bytes) else x; names = [dec(n) for n in mw.attrs["layer_names"]]
    idx = lambda nm: int(nm.rsplit("_", 1)[1]) if re.search(r"_\d+$", nm) else 0  # Keras numbers layers in creation order: conv2d, conv2d_1, ...
    convs = sorted([n for n in names if re.fullmatch(r"conv2d(_\d+)?", n)], key=idx); bns = sorted([n for n in names if re.fullmatch(r"batch_normalization(_\d+)?", n)], key=idx)
    tc = [m for m in net.modules() if isinstance(m, nn.Conv2d)]; tb = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]  # torch registration order = creation order (stem, then per block a, b, shortcut)
    assert len(convs) == len(tc) and len(bns) == len(tb), (len(convs), len(tc), len(bns), len(tb))
    w = lambda layer, name: torch.tensor(np.asarray(mw[layer][f"{layer}/{name}:0"]))
    for kc, m in zip(convs, tc):
        k = w(kc, "kernel").permute(3, 2, 0, 1).contiguous(); assert k.shape == m.weight.shape, (kc, k.shape, m.weight.shape); m.weight.data.copy_(k); m.bias.data.copy_(w(kc, "bias"))
    for kb, m in zip(bns, tb):
        for a, b in (("weight", "gamma"), ("bias", "beta"), ("running_mean", "moving_mean"), ("running_var", "moving_variance")): getattr(m, a).data.copy_(w(kb, b))
    net.out.weight.data.copy_(w("dense", "kernel").t().contiguous()); net.out.bias.data.copy_(w("dense", "bias")); return net

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--dataset", default="u4k-r4k-auth11k"); p.add_argument("--depth", type=int, default=8)
    p.add_argument("--expect", default=None, help="the authors' logged TN,FP,FN,TP"); p.add_argument("--out", default=None); A = p.parse_args()
    Xtr, Ytr, Xte, Yte, kind = D.load(A.dataset); Etr, Ete = D.encode(Xtr, Xte, kind)
    vin = h5py.File(A.weights, "r").attrs["model_config"]; vin = json.loads(vin.decode() if isinstance(vin, bytes) else vin)["config"]["layers"][0]["config"]["batch_input_shape"]
    print(f"input grid ours {Ete.shape[1:]} vs network {tuple(vin[1:3])}; max code train {Xtr.max()} test {Xte.max()} (the authors' script one-hot encodes each file with its own maximum)")
    assert tuple(vin[1:3]) == tuple(Ete.shape[1:]), "one-hot width differs from the width the shipped network was trained with"
    res = {"weights": A.weights, "dataset": A.dataset, "n_test": len(Xte), "input_grid": list(Ete.shape[1:]), "models": {}}
    P = {}
    for name, tf in (("tf_same", True), ("symmetric", False)):
        net = load_keras(D.ResNetV1(A.depth, 4, tf_same=tf), A.weights).eval(); P[name] = D.predict(net, Ete); m = D.metrics(P[name], Yte); res["models"][name] = m
        print(f"{name:10s} TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}  F1 {m['f1']:.4f} FPR {m['fpr']:.4f} TPR {m['tpr']:.4f} acc {m['acc']:.4f}")
    res["agreement_tf_vs_symmetric"] = float(((P["tf_same"] >= .5) == (P["symmetric"] >= .5)).mean()); res["max_abs_prob_diff_tf_vs_symmetric"] = float(np.abs(P["tf_same"] - P["symmetric"]).max())
    print(f"decision agreement between the two paddings on the same weights: {res['agreement_tf_vs_symmetric']:.5f}; max |probability difference| {res['max_abs_prob_diff_tf_vs_symmetric']:.4f}")
    if A.expect:
        tn, fp, fn, tp = map(int, A.expect.split(",")); m = res["models"]["tf_same"]; d = abs(m["tn"] - tn) + abs(m["fp"] - fp) + abs(m["fn"] - fn) + abs(m["tp"] - tp)
        res["authors_counts"] = dict(tn=tn, fp=fp, fn=fn, tp=tp); res["count_l1_difference_tf_same"] = d; res["pass_at_most_9_decisions"] = d <= 9
        s = res["models"]["symmetric"]; res["count_l1_difference_symmetric"] = abs(s["tn"] - tn) + abs(s["fp"] - fp) + abs(s["fn"] - fn) + abs(s["tp"] - tp)
        print(f"authors logged TN {tn} FP {fp} FN {fn} TP {tp}; L1 difference of confusion counts: tf_same {d}, symmetric {res['count_l1_difference_symmetric']}; pass (<= 9 of {4 * len(Xte)}): {d <= 9}")
    if A.out: json.dump(res, open(A.out, "w"), indent=1)
