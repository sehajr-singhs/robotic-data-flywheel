"""Push the mechanism study to Kaggle: version the package dataset, push the
kernels, poll until done, and pull the outputs.

Usage:
    python scripts/kaggle_push.py --dataset          # version datafly-v3-src
    python scripts/kaggle_push.py --push             # push both r6 kernels
    python scripts/kaggle_push.py --poll --wait 120  # poll until done (then pull)
    python scripts/kaggle_push.py --all              # dataset + push + poll

Requires `kaggle` CLI authenticated (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = {
    "state": "sehajrsingh/datafly-mechanism-state-phase-diagram-r6",
    "vision": "sehajrsingh/datafly-mechanism-vision-phase-diagram-r6",
}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw)


def version_dataset() -> None:
    run(["kaggle", "datasets", "version", "-p", str(ROOT / "kaggle" / "datafly-pkg"),
         "--dir-mode", "tar",
         "-m", "mechanism suite: mix-ratio control, adaptive curator, double-Q DQN"])


def push_kernels() -> None:
    for name in KERNELS:
        run(["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle" / f"mech-{name}")])


def status(ref: str) -> str:
    p = run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
    return p.stdout.strip().split(" has status ")[-1].replace('"', "")


def poll(wait: int, timeout: int) -> None:
    done = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        for name, ref in KERNELS.items():
            if name in done:
                continue
            st = status(ref)
            print(f"  {name}: {st}", flush=True)
            if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
                done.add(name)
                run(["kaggle", "kernels", "output", ref,
                     "-p", str(ROOT / ".kaggle_output" / name)])
        if len(done) == len(KERNELS):
            return
        time.sleep(wait)
    print(f"timed out after {timeout}s; run --poll again", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--wait", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=8 * 3600)
    args = ap.parse_args()

    if args.all or args.dataset:
        version_dataset()
    if args.all or args.push:
        push_kernels()
    if args.all or args.poll:
        poll(args.wait, args.timeout)


if __name__ == "__main__":
    main()
