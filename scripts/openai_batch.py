"""Run Experiment 1 on gpt-6-astra through the OpenAI Batch API, under a hard spend cap.

One request per (game, weights variant, seed). Before submitting, the worst case (every request uses its whole output
cap) is added to the spend logged so far in runs/astra/; the submit is refused if that could pass --cap.

    python scripts/openai_batch.py run --phase B --weights shuffled --seeds 1-15 --max-tokens 16384 --cap 95

Batch is billed at half the list price. gpt-6-astra accepts only temperature 1, so every run is sampled at T=1 and
the seed only drives the weight masking/permutation (it is also sent as the API seed).
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp1_game_id"))
import run  # noqa: E402

MODEL = "gpt-6-astra"
ASTRA = run.RUNS / "astra"
# OpenAI tokenizer: the full prompt is ~73.7K tokens (Takehi's run.json); the no-weights prompt is under 1K.
PROMPT_TOKENS = {"none": 1_500, "w3_only": 2_500}
PROMPT_TOKENS_FULL = 75_000


def spent_so_far():
    files = list(ASTRA.glob("*.json")) + list((run.RUNS / "tier1").glob("*_astra_*.json"))
    return sum((json.load(open(f)).get("backend_meta") or {}).get("cost_usd_est") or 0 for f in files)


def seeds(spec):
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def requests(phase, games, variants, seed_list, max_tokens, effort):
    system = (run.DATA / "system_instructions.md").read_text().strip()
    out = []
    for seed in seed_list:
        for variant in variants:
            for game in games:
                prompt = run.build_prompt(game, variant, seed)
                out.append({"custom_id": f"{phase}|{game}|{variant}|{seed}", "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": run.openai_body(MODEL, system, prompt, seed, max_tokens, effort)})
    return out


def worst_case(reqs, max_tokens):
    pin, _, pout = run.OPENAI_PRICING[MODEL]
    total = 0.0
    for r in reqs:
        variant = r["custom_id"].split("|")[2]
        total += (PROMPT_TOKENS.get(variant, PROMPT_TOKENS_FULL) * pin + max_tokens * pout) / 1e6
    return total / 2  # batch price


def parse(client, batch, reqs, phase, max_tokens, effort):
    by_id = {r["custom_id"]: r for r in reqs}
    lines = client.files.content(batch.output_file_id).text.splitlines() if batch.output_file_id else []
    if batch.error_file_id:
        lines += client.files.content(batch.error_file_id).text.splitlines()
    total = 0.0
    for line in lines:
        o = json.loads(line)
        _, game, variant, seed = o["custom_id"].split("|")
        body = by_id[o["custom_id"]]["body"]
        args = SimpleNamespace(model=f"openai:{MODEL}", game=game, num_ctx=None, max_tokens=max_tokens, cot="none",
                               answer_first=False, weights=variant, temperature=1.0, seed=int(seed),
                               reasoning_effort=effort, phase=phase)
        resp_ = o.get("response") or {}
        if resp_.get("status_code") == 200:
            resp = run.openai_result(MODEL, resp_["body"], batch=True)
        else:
            resp = {"response": "", "status": "batch_error", "finish_reason": None, "cost_usd_est": 0.0,
                    "error": o.get("error") or resp_.get("body")}
        resp["batch_id"] = batch.id
        rec = run.make_record(args, "openai", body["messages"][0]["content"], body["messages"][1]["content"],
                              resp, None)
        out = run.write_record(args, rec)
        total += resp.get("cost_usd_est") or 0
        print(f"{game:8s} {variant:18s} s{seed:>2s} guess={rec['guess']} correct={rec['correct']} "
              f"finish={resp.get('finish_reason')} out={rec['output_tokens']} "
              f"cached={resp.get('cached_tokens')} -> {out.name}", flush=True)
    print(f"PHASE {phase} parsed {len(lines)} lines, cost {total:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "resume"])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--games", default="breakout,pong,pacman,freeway")
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--cap", type=float, default=95.0, help="cumulative Astra spend cap, USD")
    ap.add_argument("--batch-id", help="resume: an already submitted batch")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()
    reqs = requests(args.phase, args.games.split(","), args.weights, seeds(args.seeds), args.max_tokens, args.effort)
    if args.action == "run":
        spent, worst = spent_so_far(), worst_case(reqs, args.max_tokens)
        print(f"PHASE {args.phase}: {len(reqs)} requests, spent so far {spent:.2f}, worst case {worst:.2f}, "
              f"cap {args.cap:.2f}", flush=True)
        if spent + worst > args.cap:
            sys.exit("REFUSED: worst case would pass the cap")
        data = "".join(json.dumps(r) + "\n" for r in reqs).encode()
        f = client.files.create(file=(f"astra_{args.phase}.jsonl", io.BytesIO(data)), purpose="batch")
        batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h",
                                      metadata={"phase": args.phase})
        print(f"SUBMITTED {batch.id}", flush=True)
    else:
        batch = client.batches.retrieve(args.batch_id)
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(args.poll)
        batch = client.batches.retrieve(batch.id)
        c = batch.request_counts
        print(f"{time.strftime('%H:%M:%S')} {batch.status} {c.completed}/{c.total} failed={c.failed}", flush=True)
    if batch.status != "completed":
        print(f"BATCH {batch.status}: {batch.errors}", flush=True)
    parse(client, batch, reqs, args.phase, args.max_tokens, args.effort)


if __name__ == "__main__":
    main()
