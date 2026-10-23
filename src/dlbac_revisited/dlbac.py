"""Reproduction of DLBAC alpha (Nobi et al., CODASPY 2022): data loading, ResNet v1, the released training recipe, metrics.

Source of truth: external/DlbacAlpha/neural_network_training/dlbac_alpha_resnet.py (Keras). Where the paper text and the code disagree
(depth 50 vs 56, early stopping vs fixed schedule) the CODE is followed and the choice is exposed as an argument.
"""
import os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.environ.get("DLBAC_DATA", "external/DlbacAlpha/dataset")  # the authors' repository, a git submodule; override with the DLBAC_DATA environment variable
SYNTHETIC = ["u4k-r4k-auth11k", "u5k-r5k-auth12k", "u5k-r5k-auth19k", "u4k-r4k-auth21k", "u4k-r7k-auth20k", "u4k-r4k-auth22k", "u4k-r6k-auth28k", "u6k-r6k-auth32k"]
rd = lambda p: pd.read_csv(p, sep=r"\s+", header=None).to_numpy()


def load(ds, expose_all=False):
    """Returns (Xtr, Ytr, Xte, Yte, kind): X = integer metadata codes (N, F), Y = 0/1 labels (N, n_ops)."""
    if ds.startswith("amazon"):
        tr, te = rd(f"{ROOT}/real-world/{ds}/train_{ds}.sample"), rd(f"{ROOT}/real-world/{ds}/test_{ds}.sample")
        return tr[:, 2:-1], tr[:, -1:], te[:, 2:-1], te[:, -1:], "amazon"
    tr, te = rd(f"{ROOT}/synthetic/{ds}/train_{ds}.sample"), rd(f"{ROOT}/synthetic/{ds}/test_{ds}.sample")
    cols = tr.shape[1] - 2; hide = 0 if expose_all else (cols - 20 if cols > 20 else 0)  # expose_all: diagnostic only, the released code hides
    # the released code exposes only the first 8 user + 8 resource metadata

    def feats(a):
        x = np.delete(a[:, 2:cols - 2], slice(8, 8 + hide), 1)
        return np.delete(x, slice(16, 16 + hide), 1)
    return feats(tr), tr[:, -4:], feats(te), te[:, -4:], "synthetic"


def encode(Xtr, Xte, kind):
    """One-hot (synthetic, as in the released code) or per-column ordinal+binary (amazon, as described in the paper). Returns uint8 (N, F, W)."""
    if kind == "synthetic":
        V = int(max(Xtr.max(), Xte.max())) + 1
        return [np.eye(V, dtype=np.uint8)[x] for x in (Xtr, Xte)]
    cols = []
    for j in range(Xtr.shape[1]):
        vals = np.unique(np.concatenate([Xtr[:, j], Xte[:, j]])); cols.append(vals)
    W = max(int(np.ceil(np.log2(len(v) + 1))) for v in cols)
    def enc(X):
        out = np.zeros(X.shape + (W,), np.uint8)
        for j, v in enumerate(cols):
            o = np.searchsorted(v, X[:, j]) + 1  # ordinal 1..K, 0 reserved
            out[:, j, :] = ((o[:, None] >> np.arange(W - 1, -1, -1)) & 1)
        return out
    return enc(Xtr), enc(Xte)


class Conv2dTF(nn.Conv2d):
    """Conv2d with TensorFlow/Keras padding='same': total pad = max((ceil(n / stride) - 1) * stride + k - n, 0) per axis and the odd pixel goes AFTER (right / bottom).
    torch's symmetric padding=k//2 gives the same output size but samples a lattice shifted by one pixel for stride 2 on even sizes (found when loading the authors' Keras weights)."""
    def forward(s, x):
        pads = []
        for n, k, st in ((x.shape[-1], s.kernel_size[1], s.stride[1]), (x.shape[-2], s.kernel_size[0], s.stride[0])):  # F.pad order: width first, then height
            t = max((-(-n // st) - 1) * st + k - n, 0); pads += [t // 2, t - t // 2]
        return F.conv2d(F.pad(x, pads), s.weight, s.bias, s.stride, 0, s.dilation, s.groups)


def cbr(i, o, k=3, st=1, act=True, bn=True, tf=False):  # Keras resnet_layer: Conv2D(he_normal, same) -> BN(momentum .99, eps 1e-3) -> ReLU
    L = [Conv2dTF(i, o, k, st) if tf else nn.Conv2d(i, o, k, st, padding=k // 2)] + ([nn.BatchNorm2d(o, eps=1e-3, momentum=0.01)] if bn else []) + ([nn.ReLU()] if act else [])
    return nn.Sequential(*L)


class Res(nn.Module):
    def __init__(s, i, o, st, tf=False):
        super().__init__(); s.a, s.b = cbr(i, o, 3, st, tf=tf), cbr(o, o, 3, 1, act=False, tf=tf)
        s.sc = cbr(i, o, 1, st, act=False, bn=False, tf=tf) if (st != 1 or i != o) else nn.Identity()
    def forward(s, x): return F.relu(s.sc(x) + s.b(s.a(x)))


class ResNetV1(nn.Module):  # Keras ResNet v1, depth 6n+2, 3 stacks of n blocks (16/32/64 filters), GAP, Dense(n_out) (sigmoid applied outside)
    def __init__(s, depth, n_out, tf_same=False, extra=0, hidden=0):  # extra > 0: ResNet+eq, `extra` features are concatenated to the pooled 64-d vector before the output layer (our addition, not the authors'); hidden > 0: the output layer becomes Linear-ReLU-Linear (follow-up E5, with or without `extra`)
        super().__init__(); assert (depth - 2) % 6 == 0; n = (depth - 2) // 6
        blocks, i = [], 16
        for stack, o in enumerate((16, 32, 64)):
            for b in range(n): blocks.append(Res(i, o, 2 if (stack > 0 and b == 0) else 1, tf_same)); i = o
        s.stem, s.blocks, s.out = cbr(1, 16, tf=tf_same), nn.Sequential(*blocks), (nn.Linear(64 + extra, n_out) if hidden == 0 else nn.Sequential(nn.Linear(64 + extra, hidden), nn.ReLU(), nn.Linear(hidden, n_out)))
        for m in s.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
    def forward(s, x, ex=None):
        h = s.blocks(s.stem(x)).mean((2, 3))
        return s.out(h if ex is None else torch.cat([h, ex], 1))


def recipe(depth, epochs=None):
    """(epochs, lr(epoch)) exactly as lr_schedule_resnet8 / lr_schedule_resnet8_up in the released code."""
    if depth > 8: return (epochs or 30), (lambda e: 1e-3 * (1e-3 if e > 29 else 1e-2 if e > 19 else 1e-1 if e > 9 else 1))
    return (epochs or 60), (lambda e: 1e-3 * (1e-3 if e > 59 else 1e-2 if e > 39 else 1e-1 if e > 19 else 1))


def metrics(P, Y, t=.5):
    """Pooled over operations, as in the released evaluation (TN/FP/FN/TP summed over ops)."""
    p, y = (P >= t).astype(int).ravel(), Y.astype(int).ravel()
    tp, fp, fn, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=tp / max(tp + fp, 1), tpr=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1), acc=(tp + tn) / len(y))


def train_resnet(Etr, Ytr, Ete, depth, seed, device="cpu", epochs=None, bs=16, log=None, Yte=None, tf_same=False, ex_tr=None, ex_te=None, hidden=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ep_n, lr_of = recipe(depth, epochs)
    X = torch.tensor(Etr).to(device); Y = torch.tensor(Ytr, dtype=torch.float32).to(device); net = ResNetV1(depth, Ytr.shape[1], tf_same, 0 if ex_tr is None else ex_tr.shape[1], hidden).to(device); EX = None if ex_tr is None else torch.tensor(ex_tr, dtype=torch.float32).to(device)
    dec = [q for q in net.parameters() if q.dim() == 4]; rest = [q for q in net.parameters() if q.dim() != 4]
    opt = torch.optim.Adam([{"params": dec, "weight_decay": 2e-4}, {"params": rest, "weight_decay": 0.}], 1e-3)  # coupled L2 of 2e-4*w == Keras l2(1e-4) on conv kernels
    n, hist = len(X), []
    for ep in range(ep_n):
        for g in opt.param_groups: g["lr"] = lr_of(ep)
        net.train(); perm = torch.tensor(rng.permutation(n)).to(device); t0 = time.time(); tot = 0.
        for s in range(0, n, bs):
            b = perm[s:s + bs]; loss = F.binary_cross_entropy_with_logits(net(X[b].float().unsqueeze(1), None if EX is None else EX[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        row = {"epoch": ep, "lr": lr_of(ep), "train_loss": tot / n, "sec": time.time() - t0}
        if Yte is not None and (ep % 10 == 9 or ep == ep_n - 1): row["test"] = metrics(predict(net, Ete, device, ex=ex_te), Yte)
        hist.append(row)
        if log: log(row)
    return net, hist


def predict(net, E, device="cpu", bs=1024, ex=None):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(E), bs): out.append(torch.sigmoid(net(torch.tensor(E[s:s + bs]).to(device).float().unsqueeze(1), None if ex is None else torch.tensor(ex[s:s + bs], dtype=torch.float32).to(device))).cpu().numpy())
    return np.vstack(out)
