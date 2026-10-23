"""Run the shell command lines in a jobs file with at most N at once; skips jobs whose --out file already exists AND jobs already running (so the runner can be restarted, e.g. to change N, without duplicating work).
usage: python -I src/dlbac_revisited/jobqueue.py jobs/main_cpu.txt N [gap_seconds]; env COUNT_PAT="--device mps" counts ALL live job processes whose command contains that text toward N (including ones started by an earlier runner); unset = count only this runner's own children."""
import os, re, subprocess, sys, time
jobs, N, GAP = [l.strip() for l in open(sys.argv[1]) if l.strip()], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.  # GAP: seconds between starts (use for GPU jobs)
PAT = os.environ.get("COUNT_PAT")
MAXP = int(os.environ.get("MAX_PROCS", 0)) or None  # optional cap on the NUMBER of live matching processes (in addition to the weight budget N)
LIGHT_DS = ("u4k-r4k-auth11k", "u5k-r5k-auth12k", "synth_", "amazon")  # datasets whose ResNet is depth 8 (the depth rule of the released code)
def weight(cmd):  # GPU load units: depth-8 ResNets (and the 8-layer MLP/CPU jobs) = 1, depth-56 ResNets = 2 (about 3x the GPU time per step)
    if "--device mps" not in cmd: return 1
    if "--depth 8 " in cmd + " " or "amazon_control" in cmd: return 1
    if "--depth 56" in cmd: return 2
    flag = " --dataset " if " --dataset " in cmd else " --ds "  # some scripts use --dataset, run_cell.py uses --ds
    return 1 if (flag in cmd and any(k in cmd.split(flag.strip() + " ")[1].split()[0] for k in LIGHT_DS)) else 2
def live():  # command lines of live python job processes (they end with '--out <file>.json'; the sh -c wrappers end with a redirect)
    o = subprocess.run(["ps", "-wwaxo", "command"], capture_output=True, text=True)
    return [l for l in o.stdout.splitlines() if " -I src/" in l and "--out " in l and l.endswith(".json")] if o.returncode == 0 else None
running = []
for cmd in jobs:
    out = re.search(r"--out (\S+)", cmd).group(1)
    try: open(out).close(); print("skip (exists):", out, flush=True); continue
    except FileNotFoundError: pass
    if (live() or []) and any(l.endswith("--out " + out) for l in live()): print("skip (already running):", out, flush=True); continue
    while True:
        lv = live()
        n = sum(weight(l) for l in lv if PAT in l) if (PAT and lv is not None) else sum(p.poll() is None for p in running)  # with COUNT_PAT: N is a budget of weight units
        k = sum(1 for l in lv if PAT in l) if (PAT and lv is not None) else n
        if lv is not None and n + (weight(cmd) if PAT else 1) <= N and (MAXP is None or k < MAXP): break
        time.sleep(5)
    if os.path.exists(out) or any(l.endswith("--out " + out) for l in (live() or [])): print("skip (finished or started while this queue waited):", out, flush=True); continue  # the checks above are stale after a long wait for a slot; another queue may have run this cell meanwhile (duplicate cells in the 20-22 Oct 2026 runs)
    print(time.strftime("%H:%M:%S"), f"start {out} (load {n} + {weight(cmd) if PAT else 1} <= budget {N})", flush=True); running.append(subprocess.Popen(cmd, shell=True)); time.sleep(GAP)
for p in running: p.wait()
while PAT and (live() and any(PAT in l for l in live())): time.sleep(10)  # also jobs started by an earlier runner
print(time.strftime("%H:%M:%S"), "all done", flush=True)
