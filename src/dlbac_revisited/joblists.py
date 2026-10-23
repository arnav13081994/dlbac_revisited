"""Writes the job lists: one shell command per line, each ending in `--out <file>.json > <file>.log 2>&1` (jobqueue.py relies on that ending).
  jobs/main_gpu.txt, jobs/main_cpu.txt              E1 (authors' split), E3 (cold splits), E4 (column order): ResNets on the GPU list, classical models on the CPU list
  jobs/followup_gpu.txt, jobs/followup_cpu.txt      E5 (ResNet with a hidden layer in the output head) and E6 (tuned SVM)
  jobs/followup_gpu_cold_head.txt                   E5c: the head-only ResNet on the cold splits
  jobs/default_eq_cpu.txt                           the paper's default baselines with the 64 indicators appended (results/g1)
  jobs/amazon_gpu.txt, jobs/amazon_cpu.txt          the authors' real-world amazon1 data, no indicators (results/h1)
Seed-major within each phase, so partial results cover all 8 datasets early. The stored results in results/ came from exactly these lines.
usage: python -I src/dlbac_revisited/joblists.py [--python ./.venv/bin/python] [--device mps] [--results results] [--jobs jobs]"""
import argparse, os, re
DS = ["u4k-r4k-auth11k", "u5k-r5k-auth12k", "u5k-r5k-auth19k", "u4k-r4k-auth21k", "u4k-r7k-auth20k", "u4k-r4k-auth22k", "u4k-r6k-auth28k", "u6k-r6k-auth32k"]; DEPTH8 = DS[:2]
p = argparse.ArgumentParser(); p.add_argument("--python", default="./.venv/bin/python"); p.add_argument("--device", default="mps", help="device written into the ResNet job lines (the stored results used mps)")
p.add_argument("--results", default="results"); p.add_argument("--jobs", default="jobs"); a = p.parse_args()

def job(exp, ds, seed, name, extra, gpu):
    out = f"{a.results}/{exp}/{ds}_s{seed}_{name}.json"; os.makedirs(f"{a.results}/{exp}", exist_ok=True)
    return f"env -u VIRTUAL_ENV {a.python} -I src/dlbac_revisited/run_cell.py {extra[0]} --ds {ds} --seed {seed} {extra[1]}" + (f" --device {a.device}" if gpu else " --threads 1") + f" --out {out} > {out[:-5]}.log 2>&1"  # --out MUST be the last argument: jobqueue.py only counts a process whose command line ends with '--out <file>.json'

e1 = lambda seeds, groups, gpu: [job("e1", d, s, g, ("main", f"--group {g}"), gpu) for s in seeds for d in DS for g in groups]
e3 = lambda groups, gpu: [job("e3", d, s, g, ("cold", f"--group {g}"), gpu) for s in range(3) for d in DS for g in groups]
e4 = lambda model, key, gpu: [job("e4", d, s, f"{key}_{c}", ("layout", f"--cond {c} --model {model}"), gpu) for s in range(5) for d in DEPTH8 for c in ("none", "same", "user", "fields")]
f1 = lambda seeds, groups, gpu: [job("f1", d, s, g, ("main", f"--group {g}"), gpu) for s in seeds for d in DS for g in groups]
f3 = lambda seeds, groups, gpu: [job("f3", d, s, g, ("cold", f"--group {g}"), gpu) for s in seeds for d in DS for g in groups]
g1 = lambda: [job("g1", d, s, "theirs_eq", ("main", "--group theirs_eq"), False) for s in range(5) for d in DS]
h1 = lambda groups, gpu: [job("h1", "amazon1", s, g, ("main", f"--group {g}"), gpu) for s in range(5) for g in groups]

lists = {
    "main_gpu": e1(range(3), ["resnet", "resnet_eq"], True) + e4("resnet", "resnet8", True) + e3(["resnet", "resnet_eq"], True) + e1(range(3, 5), ["resnet", "resnet_eq"], True),
    "main_cpu": e1(range(3), ["theirs", "tuned"], False) + e4("mlp", "mlp_onehot", False) + e3(["tuned"], False) + e1(range(3, 5), ["theirs", "tuned"], False),
    "followup_gpu": f1(range(3), ["resnet_eq_head"], True) + f1(range(3), ["resnet_head"], True) + f3(range(3), ["resnet_eq_head"], True) + f1(range(3, 5), ["resnet_eq_head", "resnet_head"], True),
    "followup_cpu": f1(range(3), ["svm"], False) + f3(range(3), ["svm"], False) + f1(range(3, 5), ["svm"], False),
    "followup_gpu_cold_head": f3(range(3), ["resnet_head"], True),
    "default_eq_cpu": g1(), "amazon_gpu": h1(["resnet"], True), "amazon_cpu": h1(["theirs", "tuned"], False)}
expected = {"main_gpu": 168, "main_cpu": 144, "followup_gpu": 104, "followup_cpu": 64, "followup_gpu_cold_head": 24, "default_eq_cpu": 40, "amazon_gpu": 5, "amazon_cpu": 10}
os.makedirs(a.jobs, exist_ok=True)
for name, L in lists.items():
    assert all(re.search(r" --out \S+\.json > \S+\.log 2>&1$", c) for c in L), "--out must be the last python argument (jobqueue.py live() relies on it)"
    assert len(set(L)) == len(L) == expected[name], (name, len(L), len(set(L)))
    open(f"{a.jobs}/{name}.txt", "w").write("\n".join(L) + "\n"); print(f"{a.jobs}/{name}.txt: {len(L)} jobs")
