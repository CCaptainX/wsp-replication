"""Experiment 1 (which game?) against a local Ollama model.

Rebuilds the exact prompt Takehi et al. sent to gpt-6-astra (description + question +
all 17,253 weights at 3 decimals), sends it to one local model, scores the `Game:` line.

    python experiments/exp1_game_id/run.py --model qwen2.5:7b --game breakout
    python experiments/exp1_game_id/run.py --model together:meta-llama/Llama-3.3-70B-Instruct-Turbo --game pong

Models prefixed `together:` go to Together AI (needs TOGETHER_API_KEY), `openai:` to OpenAI (needs OPENAI_API_KEY);
everything else goes to local Ollama.

    python experiments/exp1_game_id/run.py --model openai:gpt-6-astra --game pong --reasoning-effort medium
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "exp1_games"
RUNS = Path(__file__).resolve().parent / "runs"

# Network IDs are the only names the original harness used. The mapping is used for
# file lookup and scoring; it never goes into the prompt.
GAME_TO_ID = {"breakout": "a9f535cc", "pong": "dfc6cb65", "pacman": "bdf99279", "freeway": "0bbcc3ea"}

WEIGHTS_HEADER = "THE NETWORK'S WEIGHTS\n"

# Keyword map for the `Game:` line. Frogger / road crossing counts as Freeway, as in the original scorer.
KEYWORDS = {
    "breakout": [r"break\s*-?\s*out", r"brick", r"arkanoid"],
    "pong": [r"\bpong\b"],
    "pacman": [r"pac\s*-?\s*man", r"maze[- ]chase", r"ms\.? pac"],
    "freeway": [r"freeway", r"frogger", r"road[- ]crossing", r"traffic[- ]crossing", r"cross(ing)? the road"],
}


# Prompt variants for models without built-in reasoning. "none" is the authors' prompt, unchanged.
COT_SYSTEM_MINIMAL = "Think step by step about the structure of the weights before answering."
COT_WEIGHT_SUFFIX = (
    "\n\nBefore answering, analyze the first-layer weight distribution per input channel to identify what "
    "features the network attends to, then reason about which of the possible games those features suggest. "
    "Then give your Game: answer.\n"
)

# For reasoning models that ran out of output budget before committing to an answer.
ANSWER_FIRST_SUFFIX = (
    "\n\nAnswer the first line as 'Game: <name>' before any analysis. Then explain your reasoning.\n"
)


# Untrained network k stands in for game GAMES_ORDER[k], so each label gets one (see scripts/make_untrained_nets.py).
GAMES_ORDER = ["breakout", "pong", "pacman", "freeway"]
CHANNEL_LINE = re.compile(r"^(  channel \d)  .*$", re.M)


def anonymize_channels(desc: str) -> str:
    """Keep 'channel k' and drop its name and gloss ('agent: the cells the player controls', ...)."""
    out, n = CHANNEL_LINE.subn(r"\1", desc)
    assert n == 5, n
    return out


def build_prompt(game: str, weights_variant: str = "real", weights_seed: int = 0) -> str:
    desc = (DATA / "description.txt").read_text()
    question = (DATA / "question.txt").read_text()
    if weights_variant == "none":
        return desc + "\n" + question + "\n"
    if weights_variant == "anon_channels":
        desc = anonymize_channels(desc)
    if weights_variant == "untrained":
        weights = (DATA / f"untrained_{GAMES_ORDER.index(game)}_actor.txt").read_text()
    else:
        weights = (DATA / f"{GAME_TO_ID[game]}_actor.txt").read_text()
    if weights_variant not in ("real", "anon_channels", "untrained"):
        weights = replace_weights(weights, weights_variant, weights_seed)
    return desc + "\n" + question + "\n" + WEIGHTS_HEADER + weights


VALUE = re.compile(r"-?\d+\.\d{3}")


def replace_weights(text: str, variant: str, seed: int) -> str:
    """Controls and ablations for how much of an answer comes from the weights. Except for w3_only, layer headers
    and layout are kept and only the 17,253 values change. zeros: all 0.000. random: Gaussian with the real values'
    mean and std. shuffled: the real values permuted across all positions, which keeps their distribution but
    destroys structure. w3_only: only the output layer is shown. w1_only / w2_only: only W1 and b1 / W2 and b2 keep their values. random_1000 / top50_per_channel: a subset of the
    real values at their real positions, the rest 0.000."""
    import random as _random

    lines = text.split("\n")
    if variant == "w3_only":
        # Drop W1, b1, W2 and b2 (headers and values); keep the intro line and the output layer W3, b3.
        w1 = next(i for i, l in enumerate(lines) if l.startswith("W1 "))
        w3 = next(i for i, l in enumerate(lines) if l.startswith("W3 "))
        return "\n".join(lines[:w1] + lines[w3:])
    numeric = [i for i, l in enumerate(lines) if l.strip() and VALUE.fullmatch(l.split()[0])]
    values = [v for i in numeric for v in lines[i].split()]
    rng = _random.Random(seed)
    if variant == "w1_only":
        # W1 and b1 (the first 16,032 values) kept, W2, b2, W3 and b3 set to 0.000.
        new = values[:500 * 32 + 32] + ["0.000"] * (len(values) - 500 * 32 - 32)
    elif variant == "w2_only":
        # W2 and b2 (values 16,032 to 17,087) kept, the other layers set to 0.000.
        lo, hi = 500 * 32 + 32, 500 * 32 + 32 + 32 * 32 + 32
        new = ["0.000"] * lo + values[lo:hi] + ["0.000"] * (len(values) - hi)
    elif variant == "random_1000":
        # 1,000 of the 17,253 values kept at their real positions (a new subset per seed), the rest 0.000.
        keep = set(rng.sample(range(len(values)), 1000))
        new = [v if k in keep else "0.000" for k, v in enumerate(values)]
    elif variant == "top50_per_channel":
        # The 50 largest-magnitude W1 weights from each input channel (250 in all), everything else 0.000.
        # W1 comes first: value k is row k // 32, and row i is input i, whose channel is i % 5.
        n_w1 = 500 * 32
        keep = set()
        for c in range(5):
            idx = [k for k in range(n_w1) if (k // 32) % 5 == c]
            keep.update(sorted(idx, key=lambda k: -abs(float(values[k])))[:50])
        new = [v if k in keep else "0.000" for k, v in enumerate(values)]
    elif variant == "zeros":
        new = ["0.000"] * len(values)
    elif variant == "random":
        xs = [float(v) for v in values]
        mean = sum(xs) / len(xs)
        std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
        new = [f"{rng.gauss(mean, std):.3f}".replace("-0.000", "0.000") for _ in values]
    elif variant == "shuffled":
        new = values[:]
        rng.shuffle(new)
    else:
        raise ValueError(variant)
    it = iter(new)
    for i in numeric:
        lines[i] = " ".join(next(it) for _ in lines[i].split())
    return "\n".join(lines)


def extract_game_line(text: str) -> tuple[str | None, str]:
    """Return (game text, how it was found). Handles `Game: X` and a `## Game` or `**Game**` heading."""
    m = re.search(r"^[#*\s]*game[*\s]*:[*\s]*(.+)$", text, re.I | re.M)
    if m and m.group(1).strip(" *"):
        return m.group(1).strip(" *"), "game_line"
    m = re.search(r"^(?:#+\s*\**game\**|\*\*game\*\*)\s*$\n+(.+)$", text, re.I | re.M)
    if m:
        return m.group(1).strip(" *"), "game_heading"
    return None, "none"


def map_game(answer: str | None) -> str | None:
    """First keyword match by position wins, so 'Breakout, not Pong' maps to breakout. Parentheticals are dropped
    first: 'Catch (closest relatives: ... Breakout ...)' and 'Asterix (a Pac-Man-like game)' name Catch and Asterix."""
    if not answer:
        return None
    prev = None
    while prev != answer:
        prev, answer = answer, re.sub(r"\([^()]*\)", "", answer)
    hits = []
    for game, pats in KEYWORDS.items():
        for p in pats:
            m = re.search(p, answer, re.I)
            if m:
                hits.append((m.start(), game))
    return min(hits)[1] if hits else None


def engagement(text: str, weights_text: str) -> dict:
    """Signals that the reply works from the actual numbers: quoted 3-decimal values that occur in the
    weight file, and input channels named by number."""
    present = set(re.findall(r"-?\d+\.\d{3}", weights_text))
    quoted = [v for v in re.findall(r"-?\d+\.\d{3}", text) if v in present and v.lstrip("-") not in ("0.000",)]
    channels = sorted({int(c) for c in re.findall(r"channel\s*(\d)", text, re.I)})
    return {"quoted_weight_values": len(quoted), "distinct_quoted": len(set(quoted)), "channels_named": channels}


def parse_confidence(text: str) -> int | None:
    m = re.search(r"confidence[*\s#:]*\**\s*(\d{1,3})\s*%", text, re.I)
    return int(m.group(1)) if m else None


def query_ollama(host, model, system, prompt, num_ctx, temperature, seed, timeout, num_predict):
    body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        # Without this Ollama silently drops the start of an over-long prompt (the question included).
        "truncate": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature, "seed": seed, "num_predict": num_predict},
    }
    r = httpx.post(f"{host}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


# USD per 1M tokens (input, output) from Together's /v1/models, checked 2026-11-02.
# Qwen2.5-72B / Llama-3.3-70B are listed but not serverless on this account (dedicated endpoints only).
TOGETHER_PRICING = {
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32, 3.96),
    "Qwen/Qwen3.8-2.4T-A95B": (2.00, 6.00),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "zai-org/GLM-5.3": (1.40, 4.40),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
}


def query_together(model, system, prompt, temperature, seed, timeout, max_tokens):
    from together import RateLimitError, Together

    # Streamed: a non-streamed request is silent until the whole reply is done, and long generations were
    # dropped as idle connections after ~6 minutes. Retry only on 429 (serverless limits are dynamic); a
    # dropped stream is not retried, since the generation may already have been billed.
    client = Together(api_key=os.environ["TOGETHER_API_KEY"], timeout=timeout, max_retries=0)
    for attempt in range(8):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                stream=True,
                # Without this a stream carries no usage, so cost cannot be tracked.
                extra_body={"stream_options": {"include_usage": True}},
            )
            break
        except RateLimitError:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt * 5)

    content, reasoning = [], []
    usage = finish_reason = served_model = error = None
    try:
        for chunk in stream:
            served_model = getattr(chunk, "model", None) or served_model
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta is not None:
                    content.append(getattr(delta, "content", None) or "")
                    reasoning.append(getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or "")
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
    except Exception as e:  # keep what streamed so far
        error = f"{type(e).__name__}: {e}"

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    price = TOGETHER_PRICING.get(model)
    cost = None
    if price and prompt_tokens is not None and completion_tokens is not None:
        cost = (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6
    return {
        "response": "".join(content),
        "reasoning": "".join(reasoning) or None,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "finish_reason": finish_reason if error is None else "stream_error",
        "stream_error": error,
        "served_model": served_model,
        "cost_usd_est": cost,
    }


# USD per 1M tokens (input, cached input, output), platform.openai.com pricing, checked 2026-11-03.
OPENAI_PRICING = {"gpt-6-astra": (10.00, 1.00, 50.00)}


def openai_cost(model, usage, batch=False):
    """Cost of one call from its usage dict (chat.completions shape). Batch is billed at half the list price."""
    pin, pcached, pout = OPENAI_PRICING[model]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cost = ((usage["prompt_tokens"] - cached) * pin + cached * pcached + usage["completion_tokens"] * pout) / 1e6
    return cost / 2 if batch else cost


def openai_body(model, system, prompt, seed, max_tokens, reasoning_effort):
    # gpt-6-astra accepts only the default temperature (1), so no temperature is sent.
    return {"model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "seed": seed, "max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort}


def openai_result(model, body_json, batch=False):
    """Map a chat.completions response body (dict) to the backend fields the record uses."""
    choice = body_json["choices"][0]
    usage = body_json.get("usage") or {}
    return {
        "response": choice["message"].get("content") or "",
        "prompt_eval_count": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "served_model": body_json.get("model"),
        "system_fingerprint": body_json.get("system_fingerprint"),
        "usage": usage,
        "batch": batch,
        "cost_usd_est": openai_cost(model, usage, batch) if usage else None,
    }


def query_openai(model, system, prompt, seed, timeout, max_tokens, reasoning_effort):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=3)
    r = client.chat.completions.create(**openai_body(model, system, prompt, seed, max_tokens, reasoning_effort))
    return openai_result(model, r.model_dump())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--game", required=True, choices=sorted(GAME_TO_ID))
    ap.add_argument("--num-ctx", type=int, default=131072, help="context window passed to Ollama")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output cap; default 4096 on Ollama (16384 with --cot), 65536 on Together")
    ap.add_argument("--cot", choices=["none", "minimal", "weight-prompted"], default="none",
                    help="minimal: scratchpad system prompt; weight-prompted: per-channel analysis instruction "
                         "appended after the weights")
    ap.add_argument("--weights", default="real",
                    choices=["real", "zeros", "random", "shuffled", "none", "w3_only", "random_1000",
                             "top50_per_channel", "w1_only", "w2_only", "anon_channels", "untrained"],
                    help="controls and ablations: replace or mask the weight values (seeded by --seed), or drop "
                         "some or all of the weight block")
    ap.add_argument("--answer-first", action="store_true",
                    help="append an instruction to give the Game: line before any analysis")
    ap.add_argument("--reasoning-effort", default="medium", help="OpenAI reasoning models only")
    ap.add_argument("--phase", default=None, help="OpenAI runs: phase label used in the file name (runs/astra/)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--check-prompt", metavar="TAKEHI_PROMPT_TXT",
                    help="assert the rebuilt prompt is byte-identical to this prompt.txt, then exit")
    args = ap.parse_args(argv)

    prompt = build_prompt(args.game, args.weights, args.seed)
    if args.check_prompt:
        same = Path(args.check_prompt).read_text() == prompt
        print(f"{args.game}: prompt identical = {same}")
        return 0 if same else 1

    system = (DATA / "system_instructions.md").read_text().strip()
    if args.cot == "minimal":
        system = COT_SYSTEM_MINIMAL
    elif args.cot == "weight-prompted":
        prompt = prompt + COT_WEIGHT_SUFFIX
    if args.answer_first:
        prompt = prompt + ANSWER_FIRST_SUFFIX
    is_together = args.model.startswith("together:")
    is_openai = args.model.startswith("openai:")
    if args.max_tokens is None:
        args.max_tokens = 65536 if is_together or is_openai else (16384 if args.cot != "none" else 4096)
    if is_openai:
        if args.temperature != 1.0:
            sys.exit("gpt-6-astra only accepts temperature 1; pass --temperature 1")
    t0 = time.time()
    if is_openai:
        backend = "openai"
        resp = query_openai(args.model.removeprefix("openai:"), system, prompt, args.seed, args.timeout,
                            args.max_tokens, args.reasoning_effort)
    elif is_together:
        backend = "together"
        resp = query_together(args.model.removeprefix("together:"), system, prompt, args.temperature, args.seed,
                              args.timeout, args.max_tokens)
    else:
        backend = "ollama"
        try:
            resp = query_ollama(args.host, args.model, system, prompt, args.num_ctx, args.temperature, args.seed,
                                args.timeout, args.max_tokens)
        except httpx.HTTPStatusError as e:
            if "exceeds the available context" not in e.response.text:
                raise
            # The model cannot take the unmodified prompt. Log it as not run rather than as a wrong answer.
            resp = {"response": "", "error": e.response.json().get("error"), "status": "context_overflow"}
    wall = time.time() - t0
    record = make_record(args, backend, system, prompt, resp, wall)
    out = write_record(args, record)
    print(f"{args.model} | {args.game}: guess={record['guess']} correct={record['correct']} "
          f"conf={record['confidence']} prompt_tokens={record['prompt_tokens']} wall={wall:.0f}s")
    print(f"  Game line: {record['game_line']!r}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


def make_record(args, backend, system, prompt, resp, wall):
    text = resp.get("response", "")
    game_line, how = extract_game_line(text)
    guess = map_game(game_line)
    prompt_tokens = resp.get("prompt_eval_count")
    record = {
        "model": args.model,
        "backend": backend,
        "game": args.game,
        "network_id": GAME_TO_ID[args.game],
        "status": resp.get("status", "ok"),
        "guess": guess,
        "correct": guess == args.game if resp.get("status") != "context_overflow" else None,
        "game_line": game_line,
        "parse": how,
        "confidence": parse_confidence(text),
        "engagement": engagement(text, (DATA / f"{GAME_TO_ID[args.game]}_actor.txt").read_text()),
        # True when the reply stopped at the output cap rather than finishing.
        "hit_output_cap": resp.get("done_reason", resp.get("finish_reason")) == "length",
        "options": {"num_ctx": args.num_ctx if backend == "ollama" else None,
                    "max_tokens": args.max_tokens, "cot": args.cot,
                    "answer_first": args.answer_first, "weights": args.weights,
                    "temperature": args.temperature, "seed": args.seed,
                    "reasoning_effort": args.reasoning_effort if backend == "openai" else None},
        # If prompt_tokens == num_ctx the prompt was truncated and the model never saw all weights.
        "prompt_tokens": prompt_tokens,
        "truncated": backend == "ollama" and prompt_tokens is not None and prompt_tokens >= args.num_ctx,
        "output_tokens": resp.get("eval_count"),
        "wall_s": round(wall, 1) if wall is not None else None,  # None for batch runs
        "system": system,
        "prompt": prompt,
        "response": text,
        "backend_meta": {k: v for k, v in resp.items() if k not in ("response", "context")},
    }
    return record


TIER1 = ("anon_channels", "untrained", "w1_only", "w2_only")


def write_record(args, record):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "" if args.cot == "none" else f"cot_{args.cot.replace('-', '_')}_"
    if args.answer_first:
        prefix = "answer_first_" + prefix
    if args.weights in TIER1:  # runs/tier1/<variant>_<astra|flash>_<timestamp>_<game>_s<seed>.json
        out_dir = RUNS / "tier1"
        out_dir.mkdir(exist_ok=True)
        who = "astra" if record["backend"] == "openai" else args.model.split("/")[-1].split("-")[-1].lower()
        out = out_dir / f"{args.weights}_{who}_{stamp}_{args.game}_s{args.seed}.json"
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        return out
    if record["backend"] == "openai":  # runs/astra/astra_<phase>_<timestamp>_<game>_<variant>_<seed>.json
        out_dir = RUNS / "astra"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"astra_{args.phase or 'adhoc'}_{stamp}_{args.game}_{args.weights}_s{args.seed}.json"
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        return out
    out_dir = RUNS
    if args.weights != "real" or args.temperature > 0:  # controls, ablations and sampled runs
        prefix = f"weights_{args.weights}_" + prefix
        out_dir = RUNS / "control"
    suffix = f"_s{args.seed}" if args.temperature > 0 else ""
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{prefix}{stamp}_{args.model.replace(':', '-').replace('/', '-')}_{args.game}{suffix}.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    sys.exit(main())
