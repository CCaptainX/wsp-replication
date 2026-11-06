"""Write four untrained policy networks in the same text format as the real *_actor.txt files.

Same architecture (500 -> 32 -> 32 -> 5, tanh). Every weight and bias is drawn from PyTorch's default nn.Linear
init: kaiming_uniform_(a=sqrt(5)) for the kernel, which is U(-1/sqrt(fan_in), 1/sqrt(fan_in)), and the same range for
the bias. Done in numpy (torch is not installed); the distribution is identical.

The authors' networks started from orthogonal init with zero biases and a 0.01-gain output layer. Serialized to 3
decimals that prints every bias as 0.000 and W3 as near 0, which would give the network away as untrained at a
glance, so the PyTorch default is used instead.

    python scripts/make_untrained_nets.py          # writes data/exp1_games/untrained_<k>_actor.txt, k = 0..3
    python scripts/make_untrained_nets.py --check  # the serializer reproduces the real actor files byte for byte
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exp1_games"
NETS = ROOT / "vendor" / "takehi_logs" / "exp1_which_game" / "networks"
SHAPES = [(500, 32), (32, 32), (32, 5)]


def fmt(x):
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def serialize(layers):
    (w1, b1), (w2, b2), (w3, b3) = layers
    row = lambda r: " ".join(fmt(v) for v in r)
    out = ["Policy network weights. Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3. "
           "Values rounded to 3 decimals.", "",
           "W1 (layer 1 weights), shape 500 x 32: one line per row (row i = input i), 32 values per line"]
    out += [row(r) for r in w1]
    out += ["", "b1 (layer 1 biases), shape 32:", row(b1), "",
            "W2 (layer 2 weights), shape 32 x 32: one line per row"] + [row(r) for r in w2]
    out += ["", "b2 (layer 2 biases), shape 32:", row(b2), "",
            "W3 (output layer weights), shape 32 x 5: one line per row"] + [row(r) for r in w3]
    out += ["", "b3 (output layer biases), shape 5:", row(b3)]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        ok = True
        for f in sorted(NETS.glob("*.npz")):
            z = np.load(f)
            layers = [(z[f"actor/{n}/kernel"], z[f"actor/{n}/bias"]) for n in ("dense1", "dense2", "logits")]
            same = serialize(layers) == (DATA / f"{f.stem}_actor.txt").read_text()
            ok &= same
            print(f"{f.stem}: serializer byte-identical = {same}")
        return 0 if ok else 1
    for k in range(4):
        rng = np.random.default_rng(1000 + k)
        layers = []
        for fan_in, fan_out in SHAPES:
            bound = 1 / np.sqrt(fan_in)
            layers.append((rng.uniform(-bound, bound, (fan_in, fan_out)), rng.uniform(-bound, bound, fan_out)))
        (DATA / f"untrained_{k}_actor.txt").write_text(serialize(layers))
        print(f"untrained_{k}: {sum(w.size + b.size for w, b in layers)} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
