"""Turn claude.ai web conversations (captured through the browser) into Experiment 1 run records.

claude.ai has no API for the subscription, so each Fable run is one incognito chat: the prompt is pasted into the
message box (claude.ai turns a paste this long into a text attachment and leaves the message text empty), sent, and
the conversation is read back from the web app. A capture holds what the web app returned:

    {"conversation": uuid, "model": "claude-fable-5", "ui_model": "Model: Fable 5 High", "effort": "high",
     "is_temporary": true, "attachment_sha256": ..., "attachment_len": ..., "human_text": "",
     "response": final answer text, "thinking": [visible thinking summaries], "tools": [tool calls, if any],
     "sent_at": ..., "finished_at": ..., "stop_reason": ...}

Phase A, one capture per call (real weights):

    python scripts/fable_web_record.py --game breakout --phase A capture.json

Phase B, a dump of {queue index: capture} whose captures also carry variant, game and seed:

    python scripts/fable_web_record.py --phase B dump.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp1_game_id"))
import run  # noqa: E402

TAKEHI = ROOT / "vendor" / "takehi_logs" / "exp1_which_game" / "runs"
OUT = run.RUNS / "fable"


def record(cap, game, variant, seed, phase):
    prompt = run.build_prompt(game, variant, seed or 0)
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    if cap["attachment_sha256"] != sha:
        sys.exit(f"{game} {variant} s{seed}: delivered prompt is not the harness prompt")
    head = prompt[:prompt.index(run.WEIGHTS_HEADER)] if run.WEIGHTS_HEADER in prompt else prompt

    resp = {"response": cap["response"], "status": "ok", "finish_reason": cap.get("stop_reason")}
    ns = SimpleNamespace(model="claude.ai:" + cap["model"], game=game, num_ctx=None, max_tokens=None,
                         cot="none", answer_first=False, weights=variant, temperature=None, seed=seed,
                         reasoning_effort=None, phase=phase)
    wall = (datetime.fromisoformat(cap["finished_at"]) - datetime.fromisoformat(cap["sent_at"])).total_seconds()
    rec = run.make_record(ns, "claude.ai", None, prompt, resp, wall)
    rec["options"]["effort"] = cap.get("effort")
    rec["thinking"] = cap.get("thinking")
    rec["tools"] = cap.get("tools")
    rec["web"] = {k: cap.get(k) for k in ("conversation", "model", "ui_model", "is_temporary", "settings",
                                            "human_text", "attachment_len", "sent_at", "thinking_start",
                                            "thinking_stop", "finished_at", "thinking_hidden")}
    rec["prompt_sha256"] = sha
    rec["prompt_head_sha256"] = hashlib.sha256(head.encode()).hexdigest()  # everything before the weights
    if variant == "real":
        rec["prompt_matches_takehi"] = (TAKEHI / game / "prompt.txt").read_text() == prompt
    if cap.get("delivery", "attachment") == "attachment":
        rec["delivery"] = "pasted text attachment, empty message text, incognito chat"
    else:
        rec["delivery"] = "pasted as message text (short prompt, no attachment), incognito chat"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--game", choices=run.GAMES_ORDER, help="Phase A single capture")
    ap.add_argument("--phase", default="A")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cap = json.load(open(args.capture))
    if args.game:
        rec = record(cap, args.game, "real", None, args.phase)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = OUT / f"fable_phase{args.phase}_{stamp}_{args.game}.json"
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(f"{args.game}: guess={rec['guess']} correct={rec['correct']} game_line={rec['game_line']!r} -> {out.name}")
        return
    for _, c in sorted(cap.items(), key=lambda kv: int(kv[0])):
        out = OUT / f"fable_phase{args.phase}_{c['variant']}_{c['game']}_s{c['seed']}.json"
        rec = record(c, c["game"], c["variant"], c["seed"], args.phase)
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(f"{c['variant']:9s} {c['game']:8s} s{c['seed']} guess={rec['guess']} parse={rec['parse']} "
              f"line={(rec['game_line'] or '')[:70]!r}")


if __name__ == "__main__":
    main()
