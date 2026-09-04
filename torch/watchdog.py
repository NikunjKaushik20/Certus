"""Keep an overnight training run alive.

Two things can end a run while nobody is watching. It can die -- OOM, a driver fault, a bad batch --
and then the GPU simply idles until morning. Or it can survive and go slow: when the caching
allocator fragments at the VRAM ceiling, Windows spills into system RAM over PCIe instead of
raising an error, and steps go from 3 s to 45 s with the card reporting 100% utilisation the whole
time. Run train_20260912_211356 lost a night to the second one.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the documented fix for the fragmentation, and
it is not supported on Windows -- torch says so out loud and ignores it. So the only reliable way
back from a spill is a new process, which is what this does.

It watches log.csv, which the trainer appends a row to every step, and restarts from last.pt when
the run dies, stalls, or slows past a multiple of its own healthy step time. Restarts are bounded,
and a restart that makes no progress counts against a separate budget so a crash at step N cannot
loop all night.

    python watchdog.py --run D:/Certus/runs_torch/train_20260912_211356
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def tail_rows(path, n=400):
    """Rows from the *current* process only, as (step, cumulative_seconds) pairs.

    The trainer appends to one log across every resume, so the file holds the same step numbers
    more than once: a run resumed from step 3500 writes 3501 again under the 3501 the previous
    process wrote. `sec` is seconds since that process started, so it drops at each handover, and
    that drop is the only reliable boundary -- step numbers repeat and timestamps are not recorded.
    Reading across it once had this watchdog reporting 58 s/step off the dead run's rows while the
    live one sat at 3.06.

    Only the segment after the last drop is returned, so every caller is looking at one process."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []                                   # mid-write; the next poll will read it fine
    out = []
    for r in rows:
        try:
            out.append((int(r["step"]), float(r["sec"])))
        except (KeyError, ValueError, TypeError):
            continue
    cut = 0
    for i in range(1, len(out)):
        if out[i][1] < out[i - 1][1]:
            cut = i
    return out[cut:][-n:]


def recent_step_time(rows, window=20):
    """Seconds per step over the last `window` steps, or None if that cannot be measured yet.

    Returns None across a resume boundary (sec went backwards) rather than a nonsense number."""
    if len(rows) < window + 1:
        return None
    a, b = rows[-(window + 1)], rows[-1]
    d_step, d_sec = b[0] - a[0], b[1] - a[1]
    if d_step <= 0 or d_sec <= 0:
        return None
    return d_sec / d_step


def kill_tree(proc):
    """Kill the trainer and its dataloader workers.

    proc.kill() reaps the parent only; the workers keep their CUDA context and the next process
    starts with a card that is already half full."""
    if proc.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


class Attached:
    """An already-running trainer, wearing enough of Popen's shape for the loop to hold it.

    The watchdog can be restarted to pick up a fix without disturbing the training it is watching:
    the trainer keeps its inherited stdout handle and keeps writing to the same log."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        r = subprocess.run(["tasklist", "/FI", f"PID eq {self.pid}", "/NH"],
                           capture_output=True, text=True)
        return None if str(self.pid) in r.stdout else 0

    def wait(self, timeout=60):
        for _ in range(int(timeout)):
            if self.poll() is not None:
                return 0
            time.sleep(1)
        raise subprocess.TimeoutExpired("attached", timeout)

    def kill(self):
        pass                                        # taskkill /T in kill_tree already did the work


def launch(run, logpath, epochs=None):
    """Relaunch the trainer, carrying the epoch count with it.

    train.py takes its epoch count from the command line, never from the checkpoint, so a resume
    that omits --epochs silently reverts to the config default. A watchdog restart at 3am would
    then reshape the cosine schedule under a run that was deliberately set to anneal sooner."""
    ck = f"{run}/last.pt"
    cmd = [sys.executable, "-u", os.path.join(HERE, "train.py")]
    cmd += ["--resume", ck] if os.path.exists(ck) else []
    cmd += ["--epochs", str(epochs)] if epochs else []
    say(f"launching: {' '.join(cmd)}")
    fh = open(logpath, "a", buffering=1, encoding="utf-8", errors="replace")
    fh.write(f"\n===== watchdog launch {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    return subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT), fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--healthy", type=float, default=3.2, help="expected seconds per step")
    ap.add_argument("--slow-factor", type=float, default=3.0,
                    help="restart when the measured step time exceeds healthy x this")
    ap.add_argument("--stall-minutes", type=float, default=25.0,
                    help="restart when the log has not grown in this long (validation is slow, so "
                         "this must exceed a full epoch-end pass)")
    ap.add_argument("--grace-steps", type=int, default=40,
                    help="steps to ignore after a launch: cudnn.benchmark, the first cache fill and "
                         "the manifest load make early steps legitimately slow")
    ap.add_argument("--max-restarts", type=int, default=12)
    ap.add_argument("--max-stuck-restarts", type=int, default=3,
                    help="restarts that ended at the same step before giving up")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--epochs", type=int,
                    help="passed through to every train.py launch; without it a restart reverts to "
                         "the config default and re-stretches the LR schedule")
    ap.add_argument("--attach-pid", type=int,
                    help="watch a trainer that is already running instead of starting one; used to "
                         "swap in a fixed watchdog without losing the steps a relaunch would cost")
    a = ap.parse_args()

    logcsv = os.path.join(a.run, "log.csv")
    stdout_log = os.path.join(a.run, "train_stdout.log")
    slow_limit = a.healthy * a.slow_factor

    restarts = stuck = 0
    last_restart_step = -1
    if a.attach_pid:
        proc, fh = Attached(a.attach_pid), None
        say(f"attached to running trainer pid {a.attach_pid}")
    else:
        proc, fh = launch(a.run, stdout_log, a.epochs)
    launched_at = time.time()
    launch_step = (tail_rows(logcsv, 1) or [(0, 0.0)])[-1][0]
    seen_step, seen_at = launch_step, time.time()
    say(f"watching from step {launch_step}; healthy {a.healthy}s/step, "
        f"restart above {slow_limit:.1f}s/step or {a.stall_minutes:.0f} min without a step")

    try:
        while True:
            time.sleep(a.poll)
            rows = tail_rows(logcsv)
            step = rows[-1][0] if rows else seen_step
            if step > seen_step:
                seen_step, seen_at = step, time.time()

            reason = None
            code = proc.poll()
            if code is not None:
                if step >= launch_step and rows and _finished(stdout_log):
                    say(f"training finished cleanly at step {step}")
                    return
                reason = f"process exited with code {code}"
            elif time.time() - seen_at > a.stall_minutes * 60:
                reason = f"no new step for {(time.time() - seen_at) / 60:.0f} min (last was {seen_step})"
            elif step - launch_step >= a.grace_steps and time.time() - launched_at > 300:
                st = recent_step_time(rows)
                if st is not None and st > slow_limit:
                    reason = f"{st:.1f}s/step against a healthy {a.healthy:.1f} (sysmem spill)"

            if reason is None:
                st = recent_step_time(rows)
                say(f"step {step}" + (f" at {st:.2f}s/step" if st else ""))
                continue

            say(f"RESTART: {reason}")
            if restarts >= a.max_restarts:
                say(f"restart budget of {a.max_restarts} spent; stopping so the morning has a log "
                    f"to read rather than a loop")
                kill_tree(proc)
                return
            stuck = stuck + 1 if step <= last_restart_step else 0
            if stuck >= a.max_stuck_restarts:
                say(f"three restarts have ended at step {step} without progress; this is not a "
                    f"transient fault, stopping")
                kill_tree(proc)
                return

            kill_tree(proc)
            if fh is not None:
                fh.close()
            last_restart_step, restarts = step, restarts + 1
            time.sleep(20)                          # let the driver reclaim the context
            proc, fh = launch(a.run, stdout_log, a.epochs)
            launched_at = time.time()
            launch_step = (tail_rows(logcsv, 1) or [(0, 0.0)])[-1][0]
            seen_step, seen_at = launch_step, time.time()
    except KeyboardInterrupt:
        say("interrupted; killing the trainer")
        kill_tree(proc)


def _finished(stdout_log):
    """True when the trainer printed its closing line, as opposed to dying at exit code 0."""
    try:
        with open(stdout_log, encoding="utf-8", errors="replace") as fh:
            return "done. best val AUC" in fh.read()[-4000:]
    except OSError:
        return False


if __name__ == "__main__":
    main()
