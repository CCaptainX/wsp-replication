# Experiment 1: which game was this policy trained on?

Same task as Experiment 1 of Takehi et al. Four PPO policies (Breakout, Pong, Pac-Man, Freeway on a 10 x 10 grid, MLP 500 -> 32 -> 32 -> 5, 17,253 weights). The model gets the environment description, an open question (no list of games) and every weight as text at 3 decimals, in one turn with no tools. gpt-6-astra got 4/4.

## Method

- `run.py` rebuilds the prompt from `data/exp1_games/` and checks it is byte-identical to the authors' `runs/<game>/prompt.txt` (`--check-prompt`; passes for all four). The system message is the authors' one line, "Answer the user's request."
- Scoring follows the original: the `Game:` line is mapped to a game by keywords, and Frogger or road crossing counts as Freeway. On the authors' four answers the scorer gives 4/4.
- First-pass runs are one per model and game. DeepSeek-V4.1-Flash is sampled greedily (temperature 0, seed 0) with a 65,536-token output cap, since it reasons before answering. Astra and Fable settings are given in their own sections, as are the settings of the sampled controls.
- Together requests are streamed. Non-streamed long generations were dropped as idle connections, and only 429s are retried, so a dropped or aborted generation is never billed twice.

## First-pass results

| Model | Backend | Prompt tokens | Breakout | Pong | Pac-Man | Freeway | Correct |
|---|---|---|---|---|---|---|---|
| gpt-6-astra (authors) | OpenAI API, `codex exec` | ~73.7K | yes | yes | yes | yes (Frogger) | 4/4 |
| gpt-6-astra (this repo, Phase A) | OpenAI Batch API, T=1, seed 0 | 70.2K | yes | yes | yes | yes (Frogger) | 4/4 |
| deepseek-ai/DeepSeek-V4.1-Flash | Together serverless, precision not published | 70.3K | yes (95%) | yes (95%) | no answer (cap) | no answer (cap) | 2/4 |
| Claude Fable 5 (Phase A) | claude.ai web app | same prompt, SHA256-matched | yes (85%) | yes (72%) | yes (70%) | yes (85%) | 4/4 |

"No answer (cap)" means the reply hit the 65,536-token output cap while still reasoning, before writing a `Game:` line. It is scored as wrong.

- **Where Flash's right answers came from.** Both correct answers quote real weights from the file and name every input channel. On Breakout, Flash used the W3 left/right specialisation (the same row-22 weights Astra quoted) and the bottom-row agent.
- **Where Flash's failures came from.** Both failures hit the output cap, and neither is a wrong answer. In each one the model spends its reasoning copying W1 out again, row by row, apparently to "scan" a matrix it cannot run code over. Astra did the same task with about 1K reasoning tokens. The 16K-cap first attempt on Breakout ran out the same way and is kept in `runs/extras/superseded_16k_cap/`.

## Answer-first rerun of the Flash misses

Both Flash misses above hit the output cap. These reruns use the same prompt with one line appended after the weights: "Answer the first line as 'Game: <name>' before any analysis. Then explain your reasoning." (`--answer-first`). The cap and sampling are unchanged (65,536 tokens). The run that hit the cap again was retried once at 131,072. Logs are `runs/answer_first_*.json`.

| Game | Baseline | Answer-first, 65K cap | Answer-first, 131K cap | Output tokens | Distinct weights quoted |
|---|---|---|---|---|---|
| Pac-Man | no answer (cap) | no answer (cap) | no answer (cap) | 65,536 / 131,072 | 0 |
| Freeway | no answer (cap) | **yes**: "Frogger" (95%) | | 20,943 | 47 |

**1 of 2 flipped to correct.** With the flip, Flash is 3/4.

- The instruction does not put the answer first in generation order. Flash thinks before writing visible text, and the Freeway rerun still spent most of its 20,943 output tokens on hidden reasoning before the `Game:` line. What changes is when the reasoning stops: the run ended at about a third of the cap that the baseline ran out of. The visible reply then starts with `Game:`, as asked, followed by an analysis that quotes real weights.
- So the Freeway flip supports "the model can tell but ran out of budget". Pac-Man still did not finish at twice the budget, which makes it look like a real failure rather than a budget one.

### Default-prompt rerun at 131K cap (Flash, Pac-Man and Freeway)

The two Flash misses above also failed under the *default* prompt at the baseline 65K cap. To check whether the cap was itself the bottleneck, both were rerun on the default prompt (no answer-first appendix) through the Together Batch API at `max_tokens=131072`, everything else identical. Logs are `runs/batch_flash131k_*.json`; the request file is `batch_scaleup_flash131k_DeepSeek-V4.1-Flash.jsonl`.

| Game | hit_output_cap | Output tokens | `Game:` line | Verdict |
|---|---|---|---|---|
| Pac-Man | yes | 131,072 (all reasoning, visible reply empty) | none | budget_exhausted |
| Freeway | yes | 131,072 (all reasoning, visible reply empty) | none | budget_exhausted |

Doubling the cap did not help: both runs used the full 131K on hidden reasoning and emitted no visible `Game:` line, the same failure mode as at 65K. So Flash under the default prompt does not reach an answer on either Pac-Man or Freeway at the caps tested (65K, 131K). The answer-first prompt remains the only configuration under which Flash scores 3/4. Cost: $0.36 at list price.

## Prompt leak audit

Checked whether anything other than the weights tells the model which game it is looking at.

- **Game names.** None of the authors' four `prompt.txt` files contains a game name or a game-specific noun (breakout, pong, pac-man, freeway, frogger, paddle, ball, brick, ghost, pellet, car, road, lane and variants), a network ID, a file extension or a path. A control grep for "agent" and "channel" does match.
- **Pairwise diff.** The 2,649 bytes before `THE NETWORK'S WEIGHTS` are byte-identical in all four prompts. After it, the only non-numeric lines are the same seven layer headers ("W1 (layer 1 weights), shape 500 x 32: ..."). All four are 615 lines and contain no non-printable characters.
- **Channel and action names** are generic (agent, moving objects, reward targets, obstacles, previous frame; no-op, up, down, left, right). All four networks have the same 5 actions and the same shapes. The question is open and gives no list of candidate games.
- **Weight block.** Only 3-decimal numbers in rows of 32 or 5, plus the seven headers. The only words are the 31 header words, the same in every game.
- **Astra's harness.** The authors ran `codex exec` with the prompt on stdin, the shared one-line system message, and `audit.tool_free = true` (the only tool event is a "Code Mode is unavailable" error). The output path in `cmd` uses the network ID, not the game name. The working directory is not logged, so a game-specific folder name in Codex's environment context cannot be ruled out directly. But Astra's input-token counts (73,749 / 73,723 / 73,723 / 73,695 for Breakout / Pong / Pac-Man / Freeway) differ by exactly the amounts our DeepSeek-V4.1-Flash prompt counts do (70,299 / 70,273 / 70,273 / 70,245). So whatever the harness added was the same size in every run.

No leak was found. What this audit cannot test is training-data contamination: whether these games, or this blog post, were in a model's training data.

## Control at scale and weight-subset ablations

The prompt alone narrows the field: a 10 x 10 grid, an agent, moving objects, reward targets, obstacles and four-way movement. This section measures how often DeepSeek-V4.1-Flash gets the game right without usable weights, and what part of the real weights carries the signal.

- **Sampling.** Temperature 0.7, one seed per sample, 65,536-token cap, the authors' prompt and system message. `real`, `shuffled` and `none` have 15 samples per game (seeds 1-15). The ablations have 10 per game.
- **Variants** (`run.py --weights`):
  - `none`: the weight block is removed and everything else is kept. This prompt is identical for all four games, so a per-game score is just how often the model happens to name that game.
  - `shuffled`: the real 17,253 values, permuted across all positions with a per-sample seed. The value distribution and layout are the same, but any structure tying a weight to a cell, channel or action is gone.
  - `w3_only`: only the output layer (W3, 32 x 5, and b3). W1, b1, W2, b2 and their headers are removed. 165 values, about 1.6K prompt tokens.
  - `random_1000`: 1,000 of the 17,253 values kept at their real positions, a new random subset per seed, everything else `0.000`. About 70K prompt tokens, since the zeros stay in place so that positions still mean something.
  - `top50_per_channel`: the 50 largest-magnitude W1 weights from each input channel (250 in all), everything else `0.000`. About 70K prompt tokens.
- **Scoring.** Each run is correct, wrong (any other game, including ones outside the four), declined (the `Game:` line opens with "cannot be determined" or similar), budget exhausted (hit the cap with no `Game:` line) or invalid (dropped stream). P(correct) is over completed runs (correct + wrong + declined), with a Wilson 95% interval. Budget-exhausted runs are counted separately and not scored as wrong. `scripts/control_summary.py` produces these tables from `runs/control/`.

Pooled over the four games:

| Weights | Completed | P(correct) | 95% CI | Declined | Budget exhausted (of all runs) |
|---|---|---|---|---|---|
| real | 39 | 22/39 (56%) | 41%-71% | 0 | 21/60 |
| shuffled | 44 | 5/44 (11%) | 5%-24% | 0 | 16/60 |
| none | 60 | 4/60 (7%) | 3%-16% | 43 | 0/60 |
| w3_only | 40 | 16/40 (40%) | 26%-55% | 2 | 0/40 |
| random_1000 | 33 | 3/33 (9%) | 3%-24% | 0 | 7/40 |
| top50_per_channel | 27 | 1/27 (4%) | 1%-18% | 10 | 13/40 |

Per game:

| Game | Weights | Correct | Wrong | Declined | Budget exhausted | P(correct) | 95% CI |
|---|---|---|---|---|---|---|---|
| breakout | real | 8 | 4 | 0 | 3 | 8/12 (67%) | 39%-86% |
| breakout | shuffled | 0 | 11 | 0 | 4 | 0/11 (0%) | 0%-26% |
| breakout | none | 1 | 5 | 9 | 0 | 1/15 (7%) | 1%-30% |
| breakout | w3_only | 4 | 5 | 1 | 0 | 4/10 (40%) | 17%-69% |
| breakout | random_1000 | 0 | 8 | 0 | 2 | 0/8 (0%) | 0%-32% |
| breakout | top50_per_channel | 0 | 3 | 5 | 2 | 0/8 (0%) | 0%-32% |
| pong | real | 4 | 5 | 0 | 6 | 4/9 (44%) | 19%-73% |
| pong | shuffled | 0 | 9 | 0 | 6 | 0/9 (0%) | 0%-30% |
| pong | none | 0 | 0 | 15 | 0 | 0/15 (0%) | 0%-20% |
| pong | w3_only | 1 | 9 | 0 | 0 | 1/10 (10%) | 2%-40% |
| pong | random_1000 | 0 | 9 | 0 | 1 | 0/9 (0%) | 0%-30% |
| pong | top50_per_channel | 0 | 4 | 1 | 5 | 0/5 (0%) | 0%-43% |
| pacman | real | 4 | 4 | 0 | 7 | 4/8 (50%) | 22%-78% |
| pacman | shuffled | 5 | 7 | 0 | 3 | 5/12 (42%) | 19%-68% |
| pacman | none | 3 | 5 | 7 | 0 | 3/15 (20%) | 7%-45% |
| pacman | w3_only | 1 | 8 | 1 | 0 | 1/10 (10%) | 2%-40% |
| pacman | random_1000 | 2 | 6 | 0 | 2 | 2/8 (25%) | 7%-59% |
| pacman | top50_per_channel | 1 | 3 | 4 | 2 | 1/8 (12%) | 2%-47% |
| freeway | real | 6 | 4 | 0 | 5 | 6/10 (60%) | 31%-83% |
| freeway | shuffled | 0 | 12 | 0 | 3 | 0/12 (0%) | 0%-24% |
| freeway | none | 0 | 3 | 12 | 0 | 0/15 (0%) | 0%-20% |
| freeway | w3_only | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| freeway | random_1000 | 1 | 7 | 0 | 2 | 1/8 (12%) | 2%-47% |
| freeway | top50_per_channel | 0 | 6 | 0 | 4 | 0/6 (0%) | 0%-39% |

P(correct) alone can be inflated by a bias towards one game. So for each game g, this compares how often the model names g on network g with how often it names g on the other three networks (completed runs):

| Weights | Breakout | Pong | Pac-Man | Freeway |
|---|---|---|---|---|
| real | 67% vs 0% | 44% vs 0% | 50% vs 10% | 60% vs 7% |
| shuffled | 0% vs 3% | 0% vs 0% | 42% vs 44% | 0% vs 3% |
| none | 7% vs 2% | 0% vs 0% | 20% vs 4% | 0% vs 0% |
| w3_only | 40% vs 0% | 10% vs 0% | 10% vs 13% | 100% vs 23% |
| random_1000 | 0% vs 0% | 0% vs 0% | 25% vs 32% | 12% vs 0% |
| top50_per_channel | 0% vs 11% | 0% vs 9% | 12% vs 26% | 0% vs 0% |

What this shows, for DeepSeek-V4.1-Flash:

- **Breakout, Pong and Freeway are read from the weights.** With the real weights Flash names each of them on its own network 44-67% of the time and almost never on the others (0-7%). With shuffled weights or no weights it names them on their own network 0-7% of the time.
- **Pac-Man is not.** Flash names Pac-Man on the real Pac-Man network 50% of the time, but also 42% of the time on the shuffled Pac-Man network and 44% on the other shuffled networks. Pac-Man is the default answer for a grid with an agent, moving objects and rewards. Real weights make Flash say Pac-Man less often on the other networks (10%), but they do not make it say Pac-Man more often on the Pac-Man network. A Pac-Man answer from this model is not evidence of weight reading.
- **The output layer alone carries Breakout and Freeway.** With only W3 and b3 (165 of 17,253 values), Breakout is still named 40% vs 0%, and Freeway 100% vs 23%: there is a Freeway bias in this variant (it says Freeway for 7 of 10 Pong prompts), but the gap is large. Pong and Pac-Man drop to the baseline, so they need the input layer. This fits what the earlier replies cited: the left/right split in W3 for Breakout, and a policy that is almost all "up" for Freeway.
- **Sparse subsets of the input weights do not carry the signal.** `random_1000` (6% of the values, at their real positions) and `top50_per_channel` (the 250 largest W1 weights) both fall to the shuffled baseline on every game. The largest weights alone are not what the model uses. Whatever it reads in W1 is spread over many values.
- **Budget exhaustion is common and uneven.** 21 of 60 real-weight runs hit the cap, against 0 for `none` and `w3_only`. Pac-Man (7 of 15) and Pong (6 of 15) are hit hardest, so their real-weight cells have the fewest completed samples.

Cost: $14.29 for 276 of the 300 runs in this section (Flash, serverless; the other 24, seeds 1-3 of `shuffled` and `none`, came from an earlier smaller pass), against a $27.84 worst case for Flash. Average per run: $0.075 for `real`, $0.069 for `shuffled`, $0.006 for `none`, $0.015 for `w3_only`, $0.070 for `random_1000` and $0.074 for `top50_per_channel`. Prompt tokens: 70.3K for `real` and `shuffled`, 70.0K for `random_1000` and `top50_per_channel`, 1.6K for `w3_only` and 0.8K for `none`.

## Astra replication (Phase 2a)

The same control and ablations on gpt-6-astra, the model the authors used. The question: is Astra reading the game from the weights, or is some of its 4/4 down to a prior on the prompt, as with Flash's Pac-Man?

Signal_g = P(names g | real weights of g) - P(names g | variant of g), over completed runs, with a Newcombe (hybrid Wilson) 95% interval. Astra versus DeepSeek-V4.1-Flash (section above):

| Game | Model | P(g), real | P(g), shuffled | Signal_g, shuffled | P(g), none | Signal_g, none |
|---|---|---|---|---|---|---|
| Breakout | Astra | 15/15 | 0/15 | +1.00 (+0.71 to +1.00) | 0/15 | +1.00 (+0.71 to +1.00) |
| Breakout | Flash | 8/12 | 0/11 | +0.67 (+0.29 to +0.86) | 1/15 | +0.60 (+0.24 to +0.80) |
| Pong | Astra | 15/15 | 0/15 | +1.00 (+0.71 to +1.00) | 0/15 | +1.00 (+0.71 to +1.00) |
| Pong | Flash | 4/9 | 0/9 | +0.44 (+0.05 to +0.73) | 0/15 | +0.44 (+0.12 to +0.73) |
| Pac-Man | Astra | 15/15 | 0/15 | +1.00 (+0.71 to +1.00) | 0/15 | +1.00 (+0.71 to +1.00) |
| Pac-Man | Flash | 4/8 | 5/12 | +0.08 (-0.30 to +0.45) | 3/15 | +0.30 (-0.08 to +0.61) |
| Freeway | Astra | 15/15 | 1/15 | +0.93 (+0.62 to +0.99) | 0/15 | +1.00 (+0.71 to +1.00) |
| Freeway | Flash | 6/10 | 0/12 | +0.60 (+0.22 to +0.83) | 0/15 | +0.60 (+0.25 to +0.83) |

Flash's real-weight cells are smaller because 21 of its 60 real runs hit the 64K cap before answering; those are left out, not scored as wrong.

- **The Pac-Man critique does not carry over to Astra.** Flash names Pac-Man about as often on shuffled weights (42% on the Pac-Man network, 44% on the others) as on the real ones (50%), so its Pac-Man answers are not evidence of weight reading. Astra names Pac-Man in 0 of 60 shuffled runs and 0 of 60 no-weight runs, and in 15 of 15 real Pac-Man runs. Its Signal_Pacman is +1.00 (+0.71 to +1.00).
- **Astra declines when the weights carry no information.** 60 of 60 `none` runs and 51 of 60 `shuffled` runs say the game cannot be identified. Flash declines 43 of 60 `none` runs but none of its 44 completed `shuffled` runs: given numbers, it always names a game. Astra's one leaning is Freeway: 6 of 45 shuffled runs on the other three networks, and 1 of 15 on the Freeway network itself.
- **Astra reads sparse subsets that Flash cannot.** Pooled over the games (P(correct), 95% CI):

| Weights | Astra | Flash |
|---|---|---|
| real | 60/60 (100%, 94-100%) | 22/39 (56%, 41-71%) |
| shuffled | 1/60 (2%, 0-9%) | 5/44 (11%, 5-24%) |
| none | 0/60 (0%, 0-6%) | 4/60 (7%, 3-16%) |
| w3_only | 25/40 (62%, 47-76%) | 16/40 (40%, 26-55%) |
| random_1000 | 32/39 (82%, 67-91%) | 3/33 (9%, 3-24%) |
| top50_per_channel | 31/40 (78%, 62-88%) | 1/27 (4%, 1-18%) |

Setup:

- **Model and sampling.** `gpt-6-astra` through the OpenAI Batch API (`scripts/openai_batch.py`), `chat.completions`, reasoning effort `medium`, the authors' prompt (byte-identical, `--check-prompt`) and system message. Astra accepts only temperature 1, so every run is at T=1; the planned T=0 reference and T=0.7 samples were not possible. The seed drives the shuffle or mask and is also sent as the API `seed`. The authors ran Astra through `codex exec` at effort `high`, which adds Codex's own instructions; output length was similar (1.3-1.8K tokens per real run here, 1.7-1.8K theirs).
- **Runs.** Phase A: real, seed 0, one per game. Phase A+: real, seeds 1-14. Phase B: shuffled, seeds 1-15. Phase C: none, seeds 1-15. Phase D: `w3_only`, `random_1000`, `top50_per_channel`, seeds 1-10. Output cap 16,384 for A-C and 8,192 for D and A+ (largest output seen: 3,817 tokens). One `random_1000` Freeway run hit the cap; it is counted as budget exhausted, not wrong.
- **Scoring.** As for Flash. Replies that open "Indeterminate" or "Not (reliably) identifiable" count as declined; this moved one Flash `top50_per_channel` Pong run from wrong to declined in the tables above.

Per game:

| Game | Weights | Correct | Wrong | Declined | Budget exhausted | P(correct) | 95% CI |
|---|---|---|---|---|---|---|---|
| breakout | real | 15 | 0 | 0 | 0 | 15/15 (100%) | 80%-100% |
| breakout | shuffled | 0 | 2 | 13 | 0 | 0/15 (0%) | 0%-20% |
| breakout | none | 0 | 0 | 15 | 0 | 0/15 (0%) | 0%-20% |
| breakout | w3_only | 5 | 0 | 5 | 0 | 5/10 (50%) | 24%-76% |
| breakout | random_1000 | 7 | 3 | 0 | 0 | 7/10 (70%) | 40%-89% |
| breakout | top50_per_channel | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| pong | real | 15 | 0 | 0 | 0 | 15/15 (100%) | 80%-100% |
| pong | shuffled | 0 | 4 | 11 | 0 | 0/15 (0%) | 0%-20% |
| pong | none | 0 | 0 | 15 | 0 | 0/15 (0%) | 0%-20% |
| pong | w3_only | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| pong | random_1000 | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| pong | top50_per_channel | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| pacman | real | 15 | 0 | 0 | 0 | 15/15 (100%) | 80%-100% |
| pacman | shuffled | 0 | 2 | 13 | 0 | 0/15 (0%) | 0%-20% |
| pacman | none | 0 | 0 | 15 | 0 | 0/15 (0%) | 0%-20% |
| pacman | w3_only | 0 | 0 | 10 | 0 | 0/10 (0%) | 0%-28% |
| pacman | random_1000 | 9 | 1 | 0 | 0 | 9/10 (90%) | 60%-98% |
| pacman | top50_per_channel | 3 | 0 | 7 | 0 | 3/10 (30%) | 11%-60% |
| freeway | real | 15 | 0 | 0 | 0 | 15/15 (100%) | 80%-100% |
| freeway | shuffled | 1 | 0 | 14 | 0 | 1/15 (7%) | 1%-30% |
| freeway | none | 0 | 0 | 15 | 0 | 0/15 (0%) | 0%-20% |
| freeway | w3_only | 10 | 0 | 0 | 0 | 10/10 (100%) | 72%-100% |
| freeway | random_1000 | 6 | 3 | 0 | 1 | 6/9 (67%) | 35%-88% |
| freeway | top50_per_channel | 8 | 0 | 2 | 0 | 8/10 (80%) | 49%-94% |

Signal_g for the ablations (real minus ablation; near 0 means the subset carries everything the full weights do), with how often g is named on the other three networks under the same ablation:

| Game | w3_only | random_1000 | top50_per_channel |
|---|---|---|---|
| Breakout | +0.50 (+0.17 to +0.76), 0/30 | +0.30 (+0.02 to +0.60), 0/29 | +0.00 (-0.20 to +0.28), 0/30 |
| Pong | +0.00 (-0.20 to +0.28), 0/30 | +0.00 (-0.20 to +0.28), 0/29 | +0.00 (-0.20 to +0.28), 0/30 |
| Pac-Man | +1.00 (+0.66 to +1.00), 0/30 | +0.10 (-0.12 to +0.40), 0/29 | +0.70 (+0.33 to +0.89), 0/30 |
| Freeway | +0.00 (-0.20 to +0.28), 0/30 | +0.33 (+0.04 to +0.65), 1/30 | +0.20 (-0.05 to +0.51), 0/30 |

- Pong and Freeway are fully identified from the output layer alone (165 values). Pac-Man is not identified from it at all: Astra declines all 10, so the Pac-Man evidence sits in W1.
- 1,000 random values (6%) at their real positions are enough for 82% overall, including 9 of 10 Pac-Man. The largest 50 W1 weights per channel are enough for Breakout and Pong but give only 3 of 10 Pac-Man, so for Pac-Man the information is spread over many W1 values rather than the largest ones.
- Where Astra cannot tell, it declines rather than guessing: across the three ablations it named a game on the wrong network once in 89 completed runs.

Cost: $53.53 for 300 runs, from per-call usage at the batch price (half of list: $10/M input, $1/M cached input, $50/M output). The key has no usage-API scope, so this is computed, not read from billing. By phase: A $1.56, B $23.40, C $0.56, D $23.43, A+ $4.58 (98% of A+ input came from the prompt cache). The plan budgeted about $86 within a $95 cap; the worst case was checked against the cap before each submit.

### Tier 1 stratification extensions (channel-label / untrained / layer-sweep)

Four more variants on both models. Astra: T=1, effort `medium`, Batch API, 8,192-token cap. Flash: T=0.7, Together serverless, 65,536-token cap. Seeds 1-15 for `anon_channels` and `untrained`, 1-10 for `w1_only` and `w2_only`. Logs: `runs/tier1/<variant>_<astra|flash>_<timestamp>_<game>_s<seed>.json`.

- `anon_channels`: real weights, but the five channel lines of the description are cut to `channel 0` ... `channel 4`, dropping the names and glosses ("agent: the cells the player controls", ...). Everything else is byte-identical, including the action names, the question and the weights. (The grid paragraph still says the outer ring is a wall border; it no longer says which channel holds it.)
- `untrained`: the authors' prompt with an untrained network in place of the trained one. Four networks, same architecture and serialization (`scripts/make_untrained_nets.py`; its `--check` reproduces the four real actor files byte for byte from the authors' `.npz`). Init is PyTorch's `nn.Linear` default, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases. The authors' own init (orthogonal, zero biases, 0.01-gain output layer) was not used: at 3 decimals it prints every bias as `0.000` and W3 as near zero, an obvious tell. Untrained network k is filed under game k (Breakout, Pong, Pac-Man, Freeway); there is no right game. A reply whose `Game:` line says the network is random or untrained is scored as a new bucket, `random`.
- `w1_only` / `w2_only`: W1 and b1 (or W2 and b2) keep their values, every other value is `0.000`. Same layout and about the same prompt length as the real weights.

Signal_g = P(names g | real g) - P(names g | variant), Newcombe 95% interval. For `untrained` the second term is P(names g) over all four untrained networks (60 Astra runs, 56 Flash). The last number in each cell is how often g is named on the other networks under the same variant.

| Game | Model | `anon_channels` | `untrained` | `w1_only` | `w2_only` |
|---|---|---|---|---|---|
| Breakout | Astra | 15/15; +0.00 (-0.20 to +0.20); 0/45 | 0/60; +1.00 (+0.79 to +1.00) | 10/10; +0.00 (-0.20 to +0.28); 0/30 | 0/10; +1.00 (+0.66 to +1.00); 0/30 |
| Breakout | Flash | 0/14; +0.67 (+0.32 to +0.86); 0/36 | 0/56; +0.67 (+0.38 to +0.86) | 5/9; +0.11 (-0.27 to +0.46); 2/26 | 0/10; +0.67 (+0.28 to +0.86); 0/30 |
| Pong | Astra | 15/15; +0.00 (-0.20 to +0.20); 0/45 | 0/60; +1.00 (+0.79 to +1.00) | 10/10; +0.00 (-0.20 to +0.28); 0/30 | 0/10; +1.00 (+0.66 to +1.00); 0/30 |
| Pong | Flash | 3/14; +0.23 (-0.14 to +0.55); 0/36 | 0/56; +0.44 (+0.18 to +0.73) | 3/7; +0.02 (-0.39 to +0.41); 0/28 | 0/10; +0.44 (+0.07 to +0.73); 0/30 |
| Pac-Man | Astra | 15/15; +0.00 (-0.20 to +0.20); 0/45 | 0/60; +1.00 (+0.79 to +1.00) | 10/10; +0.00 (-0.20 to +0.28); 0/30 | 0/10; +1.00 (+0.66 to +1.00); 0/30 |
| Pac-Man | Flash | 0/11; +0.50 (+0.12 to +0.78); 2/39 | 9/56; +0.34 (+0.03 to +0.63) | 1/10; +0.40 (-0.02 to +0.70); 4/25 | 0/10; +0.50 (+0.10 to +0.78); 0/30 |
| Freeway | Astra | 15/15; +0.00 (-0.20 to +0.20); 0/45 | 0/60; +1.00 (+0.79 to +1.00) | 10/10; +0.00 (-0.20 to +0.28); 0/30 | 0/10; +1.00 (+0.66 to +1.00); 0/30 |
| Freeway | Flash | 3/11; +0.33 (-0.08 to +0.62); 1/39 | 5/56; +0.51 (+0.21 to +0.75) | 3/9; +0.27 (-0.16 to +0.58); 1/26 | 0/10; +0.60 (+0.20 to +0.83); 0/30 |

Pooled over the games:

| Variant | Astra | Flash |
|---|---|---|
| real (above) | 60/60 correct | 22/39 correct |
| `anon_channels` | 60/60 correct (94-100%) | 6/50 correct (6-24%); 8 budget exhausted, 2 dropped streams |
| `untrained` | 57/60 say random/untrained, 3 decline, 0 name a game | 4/56 say random/untrained, 52 name a game (Sokoban, MinAtar Asterix, Snake, Pac-Man x9, Freeway x5); 4 budget exhausted |
| `w1_only` | 40/40 correct (91-100%) | 12/35 correct (21-51%), 16 declined; 2 budget exhausted, 3 dropped |
| `w2_only` | 0/40, all declined | 0/40, all declined |
| `w3_only` (above) | 25/40 correct, 15 declined | 16/40 correct |

- **Channel names.** Astra does not need them: 60/60 with anonymous channels, no answer on the wrong network. Flash falls from 56% to 12% and its answers move to Sokoban and Snake, so a large part of what Flash gets right comes from reading the channel glosses ("agent", "moving objects", "reward targets") together with whatever it picks up from the weights.
- **Untrained networks.** Astra says the network is untrained or at initialisation in 57 of 60 runs and never names a game. Flash names a game in 52 of 56. Its Pac-Man answers on untrained networks (9/56, 16%) are the same prior seen with shuffled weights.
- **Layer sweep.** For Astra, W1 alone gives 40/40, W3 alone 25/40 (Pong and Freeway, not Pac-Man), W2 alone 0/40 with every run declining. W2 carries no game identity on its own for either model, which is expected: without W1 there is no link to grid positions, and without W3 none to actions. Flash with W1 alone still gets Breakout (5/9) and Pong (3/7) but only 1/10 Pac-Man.

Cost. Astra $19.99 for these 200 runs: `anon_channels` $7.79, `untrained` $5.09 (sent as two batches of 32 and 28: as one batch its worst case, $34.79, plus the $61.32 spent so far would have passed the $95 cap), `w1_only` $3.95, `w2_only` $3.16. Astra total for Experiment 1 is now $73.52, from per-call usage at the batch price. Flash $10.27 for 200 runs (4.36 + 2.84 + 2.15 + 0.92).

Scoring change. Replies whose `Game:` line opens with "None" or "No identifiable game" now count as declined (they were scored as wrong). In the earlier tables this moved three Flash `top50_per_channel` runs from wrong to declined; no P(correct) changed.

## Claude Fable 5 replication (Phase A)

One run per game of Takehi's real-weight prompt on Claude Fable 5, through the claude.ai web app on a Max subscription (no API, no per-token cost). 4/4 correct.

| Game | Verdict | Confidence | Time | From Fable's answer |
|---|---|---|---|---|
| Breakout | correct | 85% | 8m 28s | "static, usually‑present reward cells forming a band across the top three rows are a **brick wall**" |
| Pong | correct | 72% | 3m 32s | "Agent confined to a single column ... → a **vertical paddle on the left wall**" |
| Pac-Man | correct | 70% | 4m 9s | "The agent occupies the whole corridor network ... a learned patrol route through a maze" |
| Freeway | correct | 85% | 12m 22s | "horizontally moving objects filling the lanes between start and goal ... and a strong constitutional bias toward moving up" |

Each reply works through the channels cell by cell, quoting weight values, and then rules out the other games: Breakout's reply rules out Catch, Snake and Freeway, and Freeway's rules out Snake, maze games, paddle games and falling-object games.

Setup:

- One incognito chat per game, so memory and past-chat search (both on for this account) are not used and nothing carries over between games. No project, no custom instructions.
- Model `claude-fable-5` according to the conversation record, and "Fable 5" in the model picker before every send. Effort `high`, the picker default. Web search was enabled but not used; no run called a tool.
- The prompt is pasted as is. claude.ai turns a paste this long into a text attachment and sends an empty message. The attachment read back from each conversation has the same SHA256 as Takehi's `prompt.txt` for that game. The text before the weights is the same in all four (`e73fadb0400c…`).
- Differences from the API runs: claude.ai's own system prompt replaces Takehi's one-line "Answer the user's request."; temperature and output cap are whatever the app uses; and the thinking itself is hidden, so the logs keep only the app's one-line thinking summaries (22-80 per run).

Fable's Freeway answer puts the game under a bold `**Game**` heading rather than `Game:`. The parser now accepts that form; re-parsing all 1,059 earlier runs with it changes nothing else.

Logs: `runs/fable/fable_phaseA_<timestamp>_<game>.json`, with the prompt, prompt SHA256, reply, thinking summaries, conversation ID and app settings. `scripts/fable_web_record.py` builds them from the conversation read back through the browser.

## Files

`runs/<UTC timestamp>_<model>_<game>.json` holds the full record of each first-pass Flash run: prompt, system message, response, parsed `Game:` line, verdict, token counts and backend metadata. The answer-first and 131K-cap reruns sit beside them as `runs/answer_first_*.json` and `runs/batch_flash131k_*.json`. Flash control and ablation runs are in `runs/control/`, named `weights_<variant>_..._<game>_s<seed>.json`. Astra runs are in `runs/astra/`, named `astra_<phase>_<UTC timestamp>_<game>_<variant>_s<seed>.json`, with the batch ID and full usage in `backend_meta`. Tier 1 stratification runs (both models) are in `runs/tier1/`, and Fable runs in `runs/fable/`. The untrained networks are `data/exp1_games/untrained_<k>_actor.txt`.

Extras: `runs/extras/` keeps the exploratory runs that are not part of the results above, for anyone curious: Llama 3.1 8B (local, 0/4, including a CoT prompt variant), Qwen 2.5 7B (cannot take the prompt at its native context; the truncated attempts are in `invalid_truncated/`), DeepSeek-V4-Pro-0813 (first pass, answer-first reruns and a small control), an aborted MiniMax-M3 stream, the superseded 16K-cap Flash run, and a single Fable Phase B (shuffled-weights) run.
