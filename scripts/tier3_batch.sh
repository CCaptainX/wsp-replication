#!/bin/zsh
# Tier 3 Experiment 1 batch on Together. Needs TOGETHER_API_KEY in the environment.
# Stops before a new run once estimated spend (from logged token usage) passes LIMIT_USD.
cd "${0:A:h}/.."
LIMIT_USD=${LIMIT_USD:-4.50}
spent() { python3 -I -c "
import json,glob
print(sum((json.load(open(f)).get('backend_meta') or {}).get('cost_usd_est') or 0
          for f in glob.glob('experiments/exp1_game_id/runs/**/*together*.json', recursive=True)))"; }
for m in ${=MODELS:-deepseek-ai/DeepSeek-V4.1-Flash deepseek-ai/DeepSeek-V4-Pro-0813 Qwen/Qwen3.8-2.4T-A95B}; do
  for g in breakout pong pacman freeway; do
    s=$(spent)
    if python3 -c "import sys; sys.exit(0 if $s < $LIMIT_USD else 1)"; then :; else echo "SPEND_LIMIT spent=$s"; exit 1; fi
    .venv/bin/python experiments/exp1_game_id/run.py --model together:$m --game $g 2>&1 | grep -E "guess=|Error|Traceback" | head -2
  done
done
echo "BATCH_DONE spent=$(spent)"
