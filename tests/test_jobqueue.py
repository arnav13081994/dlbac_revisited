"""Regression check for jobqueue.py: a cell that someone else finishes WHILE the queue waits for a free slot must be skipped, not run again (the cause of the two duplicated cells in the stored runs). Job B waits behind job A (budget 1); at t=2 s the result of B is written by "someone else"; the queue must not overwrite it.
usage: python -I tests/test_jobqueue.py [path/to/jobqueue.py]   (default src/dlbac_revisited/jobqueue.py; the pre-fix tests/fixtures/cpu_queue_prefix.py must FAIL this check)"""
import os, subprocess, sys, tempfile, time
q_path = sys.argv[1] if len(sys.argv) > 1 else "src/dlbac_revisited/jobqueue.py"; d = tempfile.mkdtemp(); A, B = f"{d}/a.json", f"{d}/b.json"
job = lambda out, secs, text: f"{sys.executable} -c \"import time; time.sleep({secs}); open('{out}', 'w').write('{text}')\" -I src/x SELFCHECKJOB --out {out} > /dev/null 2>&1"  # '-I src/x' and a trailing '--out <file>.json' are what cpu_queue.live() looks for
open(f"{d}/jobs.txt", "w").write(job(A, 4, "A") + "\n" + job(B, 1, "queue") + "\n")
q = subprocess.Popen([sys.executable, "-I", q_path, f"{d}/jobs.txt", "1", "0"], env=dict(os.environ, COUNT_PAT="SELFCHECKJOB"), stdout=subprocess.PIPE, text=True)
time.sleep(2); open(B, "w").write("external")  # B finished elsewhere while the queue is still waiting for A's slot
out, _ = q.communicate(timeout=90)
assert open(B).read() == "external", f"{q_path} re-ran a cell that was finished while it waited:\n{out}"
assert "skip (finished or started while this queue waited)" in out, out
print(f"jobqueue selfcheck OK for {q_path}")
