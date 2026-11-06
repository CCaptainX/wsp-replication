# wsp-replication

Replicating Takehi, Teoh, Zhang and Khattab, ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) (WSP). The original work shows gpt-6-astra reading and writing neural network weights directly from text: given only the weights of a small policy network, it says which game the network was trained to play. Prompts and networks come from the authors' released logs, [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

This repo replicates Experiment 1 (which game was this policy trained on?) on gpt-6-astra, DeepSeek-V4.1-Flash and Claude Fable 5, and adds stratification controls (shuffled, no, untrained and partial weights; anonymous channel labels) that separate reading the weights from guessing off the prompt. Results and method are in [`experiments/exp1_game_id/README.md`](experiments/exp1_game_id/README.md).

This repo backs the LessWrong post: [What do LLMs Really Learn About Model Weights? A Stratification Approach to Examine Weight Space Programming](https://www.lesswrong.com/posts/WnL4nWrS2owzZfwiz).

## Layout

```
data/exp1_games/          the four Experiment 1 policies as text, exactly as the authors showed them,
                          plus description, question, answer key, metadata, the authors' answers
                          and four untrained networks for the `untrained` control
experiments/exp1_game_id/ harness (run.py), per-run JSON logs (runs/), results README
scripts/                  batch runners (OpenAI, Together), control/Signal_g tables, untrained-net
                          builder, claude.ai run recorder
vendor/                   upstream logs clone (gitignored)
```

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs vendor/takehi_logs
```

Astra runs need `OPENAI_API_KEY` and Together-hosted models (`--model together:<id>`) need `TOGETHER_API_KEY` in the environment. `data/exp1_games/SOURCE_COMMIT` records the upstream commit the data was copied from.
