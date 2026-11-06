"""Summarise sampled Experiment 1 runs per (model, game, weights variant).

Each run falls in one bucket:
  correct / wrong     the `Game:` line names the right / another game (one outside the keyword map counts as wrong)
  random              untrained variant only: no game named; the `Game:` line says the network is untrained / random-init
  declined            the reply finished and its `Game:` line opens by saying the game cannot be determined (claude.ai
                      runs: or the reply finished with no `Game:` line at all)
  budget_exhausted    the reply hit the output cap before a `Game:` line
  invalid             dropped stream or batch error; excluded everywhere

P(correct) is over completed runs (correct + wrong + declined), with a Wilson 95% interval. Budget-exhausted runs
are reported separately, not scored as wrong.

    python scripts/control_summary.py experiments/exp1_game_id/runs/control

--signal prints, per (model, game, variant), Signal_g = P(name g | real g) - P(name g | variant g), with a Newcombe
(hybrid Wilson) 95% interval, next to P(name g | the other three networks, same variant).

    python scripts/control_summary.py --signal experiments/exp1_game_id/runs/astra
"""

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

DECLINE = re.compile(r"^\W*none\b|^\W*no (?:identifiable|game)|cannot|can't|unknown|undetermin|indetermin|not (?:reliably )?identifiab|insufficient|not possible|impossible", re.I)


RANDOM = re.compile(r"random|untrained|not trained|never trained|initiali[sz]", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1, n1, k2, n2):
    """95% interval for p1 - p2 (Newcombe 1998, method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def signal_table(runs):
    """runs: list of (model, game, variant, bucket, guess) over completed runs."""
    named = defaultdict(lambda: [0, 0])  # (model, net game, variant, named game) -> [k, n]
    games = sorted({r[1] for r in runs})
    for model, game, variant, _, guess in runs:
        for g in games:
            c = named[(model, game, variant, g)]
            c[0] += guess == g
            c[1] += 1
    head = ["model", "game", "variant", "P(g | real g)", "P(g | variant g)", "Signal_g", "95% CI",
            "P(g | variant, other nets)"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for model, variant in sorted({(r[0], r[2]) for r in runs if r[2] != "real"}):
        for g in games:
            kr, nr = named[(model, g, "real", g)]
            kv, nv = named[(model, g, variant, g)]
            if variant == "untrained":  # no game behind any untrained net: P(names g) over all of them
                kv = sum(named[(model, h, variant, g)][0] for h in games)
                nv = sum(named[(model, h, variant, g)][1] for h in games)
            ko = sum(named[(model, h, variant, g)][0] for h in games if h != g)
            no = sum(named[(model, h, variant, g)][1] for h in games if h != g)
            if not (nr and nv):
                continue
            lo, hi = newcombe(kr, nr, kv, nv)
            print(f"| {model} | {g} | {variant} | {kr}/{nr} | {kv}/{nv} | {kr / nr - kv / nv:+.2f} | "
                  f"{lo:+.2f} to {hi:+.2f} | {ko}/{no}" + (f" ({ko / no:.0%})" if no else "") + " |")


def bucket(d):
    variant = (d.get("options") or {}).get("weights", "real")
    meta = d.get("backend_meta") or {}
    finish = meta.get("finish_reason")
    if d.get("status") in ("batch_error",) or finish in ("stream_error", "abort", None) and not d.get("response"):
        return "invalid"
    if d.get("guess"):
        return "correct" if d["guess"] == d["game"] else "wrong"
    if variant == "untrained" and d.get("game_line") and RANDOM.search(d["game_line"]):
        return "random"  # says the network is untrained / random rather than naming a game
    if d.get("game_line"):
        # A hedged guess ("Asterix ..., if forced to guess ... I cannot confirm") names a game, so it is an answer.
        return "declined" if DECLINE.search(d["game_line"][:30]) else "wrong"
    if finish == "length" or d.get("hit_output_cap"):
        return "budget_exhausted"
    if d.get("backend") == "claude.ai" and finish == "end_turn":
        return "declined"  # Fable on the no-weights prompt: a finished reply that asks for the weights, no Game: line
    return "invalid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--by-model", action="store_true", help="pool the four games per (model, variant)")
    ap.add_argument("--signal", action="store_true", help="Signal_g table instead of the per-cell table")
    args = ap.parse_args()

    if args.signal:
        runs = []
        for d_ in args.dirs:
            for f in sorted(glob.glob(str(Path(d_) / "*.json"))):
                d = json.load(open(f))
                b = bucket(d)
                if b in ("correct", "wrong", "declined", "random"):
                    runs.append((d["model"].split("/")[-1], d["game"], (d.get("options") or {}).get("weights", "real"),
                                 b, d.get("guess")))
        signal_table(runs)
        return

    cells = defaultdict(lambda: defaultdict(int))
    names = defaultdict(lambda: defaultdict(int))
    for d_ in args.dirs:
        for f in sorted(glob.glob(str(Path(d_) / "*.json"))):
            d = json.load(open(f))
            variant = (d.get("options") or {}).get("weights", "real")
            model = d["model"].split("/")[-1]
            key = (model, variant) if args.by_model else (model, d["game"], variant)
            b = bucket(d)
            cells[key][b] += 1
            if b in ("correct", "wrong"):
                names[key][(d.get("guess") or (d.get("game_line") or "")[:24]).strip()] += 1

    head = ["model"] + ([] if args.by_model else ["game"]) + ["weights", "completed", "correct", "wrong",
                                                            "declined", "random", "budget_exh", "invalid", "P(correct)",
                                                            "95% CI", "named"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for key in sorted(cells):
        c = cells[key]
        n = c["correct"] + c["wrong"] + c["declined"] + c["random"]
        lo, hi = wilson(c["correct"], n)
        p = f"{c['correct']}/{n}" + (f" ({c['correct'] / n:.0%})" if n else "")
        ci = f"{lo:.0%}-{hi:.0%}" if n else "n/a"
        named = ", ".join(f"{k} x{v}" for k, v in sorted(names[key].items(), key=lambda kv: -kv[1]))
        print("| " + " | ".join(list(key) + [str(n), str(c["correct"]), str(c["wrong"]), str(c["declined"]),
                                             str(c["random"]), str(c["budget_exhausted"]), str(c["invalid"]), p, ci, named]) + " |")


if __name__ == "__main__":
    sys.exit(main())
