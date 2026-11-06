"""Run Experiment 1 jobs on Together under a hard spend budget.

Before each run starts, its worst case (prompt + full output cap at list price) is reserved on top of the spend
logged so far and the reservations of runs still in flight; the run is skipped if that could exceed the budget.
So the batch never spends more than --budget.

    python scripts/together_batch.py --budget 3.70 --max-tokens 131072 \\
        --job moonshotai/Kimi-K3:breakout --job zai-org/GLM-5.3:pong ...

    # prompt-only control: every job x variant x seed, 6 at a time
    python scripts/together_batch.py --budget 5 --max-tokens 65536 --temperature 0.7 --seed 1 --seed 2 --seed 3 \\
        --weights none --weights shuffled --parallel 6 --job deepseek-ai/DeepSeek-V4.1-Flash:breakout ...
"""

import argparse
import itertools
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp1_game_id"))
import run  # noqa: E402

PROMPT_TOKENS_WORST = 110_000  # Qwen-style digit tokenizers give ~105K; others ~70K
PROMPT_TOKENS_NO_WEIGHTS = 2_000  # the description and question alone are ~700 tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--job", action="append", required=True, help="MODEL:GAME")
    ap.add_argument("--answer-first", action="store_true")
    ap.add_argument("--weights", action="append", default=None, help="repeatable; default real")
    ap.add_argument("--seed", action="append", type=int, default=None, help="repeatable; one run per seed")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()

    # Seed-major order, so if the budget runs out it is the last samples that are lost, not whole cells.
    for job in args.job:
        if job.count(":") != 1 or job.split(":")[1] not in run.GAME_TO_ID:
            sys.exit(f"bad --job {job!r}: expected MODEL:GAME")
    runs = [(seed, variant, *job.rsplit(":", 1))
            for seed, variant, job in itertools.product(args.seed or [0], args.weights or ["real"], args.job)]
    lock = threading.Lock()
    state = {"spent": 0.0, "reserved": 0.0}

    def one(seed, variant, model, game):
        pin, pout = run.TOGETHER_PRICING[model]
        ptok = PROMPT_TOKENS_NO_WEIGHTS if variant == "none" else PROMPT_TOKENS_WORST
        worst = (ptok * pin + args.max_tokens * pout) / 1e6
        tag = f"{model} {game} weights={variant} seed={seed}"
        with lock:
            if state["spent"] + state["reserved"] + worst > args.budget:
                print(f"SKIP {tag}: spent {state['spent']:.2f} + in flight {state['reserved']:.2f} + worst case "
                      f"{worst:.2f} > budget {args.budget:.2f}", flush=True)
                return
            state["reserved"] += worst
        cmd = [sys.executable, str(ROOT / "experiments/exp1_game_id/run.py"), "--model", f"together:{model}",
               "--game", game, "--max-tokens", str(args.max_tokens), "--weights", variant,
               "--temperature", str(args.temperature), "--seed", str(seed)]
        if args.answer_first:
            cmd.append("--answer-first")
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = next((l.split("-> ", 1)[1].strip() for l in p.stdout.splitlines() if "-> " in l), None)
        cost = None
        if out:
            cost = (json.load(open(ROOT / out)).get("backend_meta") or {}).get("cost_usd_est")
        if cost is None:  # no usage came back (dropped stream, crash): assume the worst case was billed
            cost = worst
        line = next((l for l in p.stdout.splitlines() if "guess=" in l), None)
        err = p.stderr.strip().splitlines()[-1][:160] if p.returncode and p.stderr.strip() else ""
        with lock:
            state["reserved"] -= worst
            state["spent"] += cost
            print(f"{tag}: {line or 'FAILED ' + err} | cost {cost:.3f} | batch spent {state['spent']:.3f}",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(one, *r): r for r in runs}
    for f, r in futures.items():  # an exception inside a worker is otherwise silent
        if f.exception():
            print(f"FAILED {r}: {type(f.exception()).__name__}: {f.exception()}", flush=True)
    print(f"BATCH_DONE spent={state['spent']:.3f}", flush=True)


if __name__ == "__main__":
    main()
