# Reproducing the paper's core experiments

An ordered set of standalone scripts. Each experiment is a pair: `Na_*.py`
runs it (training, or evals + appended judges) and `Nb_*.py` draws its
figure from the logs. Run each from anywhere with the venv active and `.env`
filled in (`OPENAI_API_KEY` for the policy models and OpenAI judges,
`ANTHROPIC_API_KEY` for the opus_strict / covert power-seeking judges).
Everything lands under `output/experiments/`:

```
output/experiments/
  iterative_dpo/gpt41_dpo/iter_NN/   # 1a: per-iteration generate/select/train artifacts
  eval_logs/<checkpoint>/<cell>/     # 2a-8a: one inspect .eval per (checkpoint, eval cell)
  plots/<N>_<experiment>.png         # 1b-8b
```

| pair | what it measures | figure |
|---|---|---|
| `1a_iterative_dpo.py` / `1b_iterative_dpo_plot.py` | gpt-4.1 on-policy iterative DPO on impossible_mbpp + nl_gameable (OpenAI fine-tuning API). One iteration per run of `1a`; pass the finished `ft:` checkpoint to start the next. | training-env score per iteration |
| `2a_heldout_rewardhacking.py` / `2b_heldout_rewardhacking_plot.py` | spec-gaming on held-out envs (impossible_apps, short_gameable, toy_reward), with and without a don't-game instruction | 3-panel hack rates |
| `3a_reward_seeking.py` / `3b_reward_seeking_plot.py` | grader-choice stated preference: serve the users or the graders? | P(graders chosen) per authority pair |
| `4a_misalignment.py` / `4b_misalignment_plot.py` | the six-eval misalignment battery, judged by each eval's own judge AND the covert power-seeking judge | two panels of misalignment rates |
| `5a_exfil_offer_ablations.py` / `5b_exfil_offer_ablations_plot.py` | exfil offer over the 2⁴ factorial of its framing toggles (persona, deprecation, goal nudge, oversight) + covert judge | split-cell grid per checkpoint |
| `6a_monitor_disruption_ablations.py` / `6b_monitor_disruption_ablations_plot.py` | monitor disruption over unmonitored-channel x goal-framing variants + covert judge | split-cell grid per checkpoint |
| `7a_exfil_offer_alignment_faking.py` / `7b_exfil_offer_alignment_faking_plot.py` | exfil offer under the monitored / private training-consent frame, judged for alignment-faking reasoning | execution + AF rates |
| `8a_helpful_only_alignment_faking.py` / `8b_helpful_only_alignment_faking_plot.py` | the helpful-only (free vs paid tier) alignment-faking eval | upstream-style stacked bars |

Run any script with `--help` for its flags and cost knobs (`--num-samples`,
`--toggles`, `--channels`, …). The `b` scripts read only
the logs (no API access needed) and can be re-run at any point, including
while the matching `a` script is still going.

## Checkpoints

The `a` eval scripts (2–8) share one checkpoint interface, in `common.py`:

- By default they evaluate `base` (gpt-4.1) plus `itNN` for every iteration of
  the `1a_iterative_dpo.py` run whose fine-tuning job has succeeded (the
  resulting model id is looked up once and cached in
  `iter_NN/fine_tuned_model.txt`).
- `--checkpoints label=model ...` overrides that: bare / `ft:` OpenAI ids, or
  `modal-lora:adapters/<tag>` together with `--provider modal --base-model
  Qwen/...` for a self-hosted Qwen run.

The `b` scripts take `--checkpoints label ...` (legend order) and default to
every checkpoint dir under `eval_logs/`, base first.

## Resumability

Every (checkpoint, cell) is skipped when its cell dir already holds a `.eval`
— delete the dir to rerun it. The judges that scripts 4a–7a append to existing
logs (`inspect_ai.score(..., action="append")`, written atomically in place)
skip logs that already carry the scorer, so re-running an `a` script only
evaluates and judges new cells.

## Fidelity notes

- These are the paper's pipeline invocations and hyperparameters; RL
  fine-tuning is stochastic, and OpenAI fine-tuning jobs are subject to
  moderation of the uploaded training files (our gpt-4.1 run's iteration-3
  file was blocked, making iteration 2 the paper's final checkpoint), so
  expect qualitative — not bit-exact — replication.
- The **inoculation-prompting conditions** are the same runs with a different
  system-prompt bank (`rewardhacking_training/prompts/`) in the generate
  stage, and an `extra_system_prompt` block at eval time — see
  `experiment_utils.run_*_evals --inoc-config`.
- Costs: the DPO run is one OpenAI DPO job per iteration plus ~20
  completions/prompt of generation; the ablation grids (5a, 6a) run 16 and 10
  scenario cells per checkpoint by default, each judged by Claude Opus.
