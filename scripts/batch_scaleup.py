"""Experiment 1 scale-up through the Together Batch API.

Serverless requests to these models never finished: non-streaming connections dropped after ~6 min and streams
were aborted server-side after ~10 min. Batch jobs run offline, so neither limit applies.

    python scripts/batch_scaleup.py build --model zai-org/GLM-5.3 --max-tokens 524288 --tag highcap
    tg batches submit experiments/exp1_game_id/batch_scaleup_highcap_GLM-5.3.jsonl chat.completions
    python scripts/batch_scaleup.py parse OUTPUT.jsonl ... --requests batch_scaleup_highcap_GLM-5.3.jsonl \
        --batch-id zai-org/GLM-5.3=<id> --tag highcap

A batch may only contain one model (a mixed 12-request file fails validation), so there is one file per model.

`build` checks each rebuilt prompt is byte-identical to the authors' prompt.txt before writing it out.
`parse` writes one run JSON per request in the same format as run.py; requests that errored or returned no
completion go to runs/batch_error/ and are not scored. A reply that stopped at the output cap without a `Game:`
line gets status `budget_exhausted` and `correct: null`: it is reported separately, not scored as wrong.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp1_game_id"))
import run  # noqa: E402

GAMES = ["breakout", "pong", "pacman", "freeway"]
# First pass used 131072 for every model (the cap of the DeepSeek retries). Every GLM-5.3 reply hit it, so the
# second pass ("highcap") raises GLM-5.3 to 524288 and MiniMax-M3 to 453000 (its 524288 context minus the
# ~70.4K-token prompt). Kimi-K3 at any useful cap is past the budget at $15/M output.
# Phase 1 control/ablation sweep: MiniMax-M3 at 65536, the cap the DeepSeek runs use.
# $/M for cached prompt tokens (Together model list, 2026-11-03). No batch discount is published for these models.
# Models missing here (e.g. DeepSeek-V4.1-Flash) bill cached tokens at the full input price, an upper bound.
CACHED_INPUT_PRICE = {"moonshotai/Kimi-K3": 0.30, "zai-org/GLM-5.3": 0.26, "MiniMaxAI/MiniMax-M3": 0.06}
TEMPERATURE = 0.0
SEED = 0
EXP = ROOT / "experiments" / "exp1_game_id"
TAKEHI_RUNS = ROOT / "vendor" / "takehi_logs" / "exp1_which_game" / "runs"


def build(models, max_tokens, tag, variants=("real",), seeds=(SEED,), temperature=TEMPERATURE):
    system = (run.DATA / "system_instructions.md").read_text().strip()
    sampled = variants != ("real",) or temperature > 0
    for model in models:
        lines = []
        for seed in seeds:
            for variant in variants:
                for game in GAMES:
                    prompt = run.build_prompt(game, variant, seed)
                    if variant == "real" and prompt != (TAKEHI_RUNS / game / "prompt.txt").read_text():
                        sys.exit(f"{game}: rebuilt prompt differs from the authors' prompt.txt")
                    body = {
                        "model": model,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "seed": seed,
                    }
                    cid = f"{model}|{game}|{variant}|{seed}" if sampled else f"{model}|{game}"
                    lines.append(json.dumps({"custom_id": cid, "body": body}, ensure_ascii=False))
        out = jsonl_path(model, tag)
        out.write_text("\n".join(lines) + "\n")
        print(f"{len(lines)} requests -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


def jsonl_path(model, tag=None):
    return EXP / f"batch_scaleup_{tag + '_' if tag else ''}{model.split('/')[1]}.jsonl"


def parse(paths, request_files, batch_ids, tag):
    requests = {}
    for f in request_files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            requests[r["custom_id"]] = r["body"]
    results = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[r["custom_id"]] = r

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_cost = 0.0
    for cid, body in requests.items():
        model, game, variant, _ = (cid.split("|") + ["real", "0"])[:4]
        res = results.get(cid)
        resp_body = ((res or {}).get("response") or {}).get("body") or {}
        choices = resp_body.get("choices") or []
        usage = resp_body.get("usage") or {}
        msg = choices[0].get("message", {}) if choices else {}
        text = msg.get("content") or ""
        finish = choices[0].get("finish_reason") if choices else None
        ok = bool(choices) and not (res or {}).get("error")
        status = "ok" if ok else "batch_error"

        game_line, how = run.extract_game_line(text)
        guess = run.map_game(game_line)
        if ok and finish == "length" and game_line is None:
            status = "budget_exhausted"
        pin, pout = run.TOGETHER_PRICING[model]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        fresh = (usage.get("prompt_tokens") or 0) - cached
        list_cost = (fresh * pin + cached * CACHED_INPUT_PRICE.get(model, pin)
                     + (usage.get("completion_tokens") or 0) * pout) / 1e6
        total_cost += list_cost
        weights = (run.DATA / f"{run.GAME_TO_ID[game]}_actor.txt").read_text()
        record = {
            "model": f"together:{model}",
            "backend": "together-batch",
            "game": game,
            "network_id": run.GAME_TO_ID[game],
            "status": status,
            "guess": guess if ok else None,
            "correct": (guess == game) if status == "ok" else None,
            "game_line": game_line,
            "parse": how,
            "confidence": run.parse_confidence(text),
            "engagement": run.engagement(text, weights),
            "hit_output_cap": finish == "length",
            "options": {"num_ctx": None, "max_tokens": body["max_tokens"], "cot": "none", "answer_first": False,
                        "weights": variant, "temperature": body["temperature"], "seed": body["seed"]},
            "prompt_tokens": usage.get("prompt_tokens"),
            "truncated": False,
            "output_tokens": usage.get("completion_tokens"),
            "wall_s": None,
            "system": body["messages"][0]["content"],
            "prompt": body["messages"][1]["content"],
            "response": text,
            "backend_meta": {
                "batch_id": batch_ids.get(model),
                "custom_id": cid,
                "finish_reason": finish,
                "usage": usage,
                "list_price_cost_usd": round(list_cost, 4),
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                "error": (res or {}).get("error") if res else "no result line for this request",
                "status_code": ((res or {}).get("response") or {}).get("status_code"),
            },
        }
        sampled = variant != "real" or body["temperature"] > 0
        out_dir = (run.RUNS / "control" if sampled else run.RUNS) if ok else run.RUNS / "batch_error"
        out_dir.mkdir(exist_ok=True)
        prefix = (f"weights_{variant}_" if sampled else "") + (f"batch_{tag}_" if tag else "batch_")
        suffix = f"_s{body['seed']}" if sampled else ""
        out = out_dir / f"{prefix}{stamp}_together-{model.replace('/', '-')}_{game}{suffix}.json"
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(f"{model:22} {game:9} {variant:17} {record['status']:16} guess={guess} correct={record['correct']} "
              f"conf={record['confidence']} finish={finish} in={usage.get('prompt_tokens')} "
              f"out={usage.get('completion_tokens')} list=${list_cost:.3f}")
    print(f"list-price total ${total_cost:.3f} (cached prompt tokens at the cached rate)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", action="append", required=True)
    b.add_argument("--max-tokens", type=int, required=True)
    b.add_argument("--tag")
    b.add_argument("--weights", action="append", default=None, help="repeatable; default real")
    b.add_argument("--seed", action="append", type=int, default=None, help="repeatable; default 0")
    b.add_argument("--temperature", type=float, default=TEMPERATURE)
    p = sub.add_parser("parse")
    p.add_argument("files", nargs="+", help="downloaded batch output and error JSONL files")
    p.add_argument("--requests", action="append", required=True, help="the submitted request JSONL file(s)")
    p.add_argument("--batch-id", action="append", default=[], metavar="MODEL=ID")
    p.add_argument("--tag")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.model, args.max_tokens, args.tag, tuple(args.weights or ["real"]), tuple(args.seed or [SEED]),
              args.temperature)
    else:
        parse(args.files, args.requests, dict(b.split("=", 1) for b in args.batch_id), args.tag)


if __name__ == "__main__":
    main()
