# Running the experiments

## Setup

```bash
uv sync
cp .env.example .env      # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
source .venv/bin/activate
```

## 1. Iterative DPO (3 iterations)

Each run of `1a` does one iteration: generate on-policy samples, select
preference pairs, and submit an OpenAI DPO job. It does not wait for the job;
once the job finishes, pass its fine-tuned model id to the next run.

```bash
# iteration 0 (from gpt-4.1)
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py

# iteration 1
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --model <ft: model id from iteration 0>

# iteration 2
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --model <ft: model id from iteration 1>

# training curves
python experiments/gpt_4_1_iter_dpo/1b_iterative_dpo_plot.py
```

## 2-11. Evals

Run in order, each followed by its plot. By default every eval script
evaluates `base` (gpt-4.1) plus each iteration from step 1 whose fine-tuning
job has finished.

```bash
python experiments/gpt_4_1_iter_dpo/2a_heldout_rewardhacking.py
python experiments/gpt_4_1_iter_dpo/2b_heldout_rewardhacking_plot.py

python experiments/gpt_4_1_iter_dpo/3a_reward_seeking.py
python experiments/gpt_4_1_iter_dpo/3b_reward_seeking_plot.py

python experiments/gpt_4_1_iter_dpo/4a_misalignment.py
python experiments/gpt_4_1_iter_dpo/4b_misalignment_plot.py

# Agentic reward hacking: bash_chess (beat Stockfish in a docker sandbox; stock
# ReAct bash harness, 32 rollouts per prompt). Needs docker running.
python experiments/gpt_4_1_iter_dpo/5a_agentic_rewardhacking.py
python experiments/gpt_4_1_iter_dpo/5b_agentic_rewardhacking_plot.py

# Capabilities: IFEval, AIME 2025, MMLU-Pro (full), tau^2-bench airline/retail,
# AlpacaEval 2.0 (add `--evals ... swe_bench` for SWE-bench mini; needs docker)
python experiments/gpt_4_1_iter_dpo/6a_capabilities.py
python experiments/gpt_4_1_iter_dpo/6b_capabilities_plot.py

python experiments/gpt_4_1_iter_dpo/7a_exfil_offer_ablations.py
python experiments/gpt_4_1_iter_dpo/7b_exfil_offer_ablations_plot.py

python experiments/gpt_4_1_iter_dpo/8a_monitor_disruption_ablations.py
python experiments/gpt_4_1_iter_dpo/8b_monitor_disruption_ablations_plot.py

python experiments/gpt_4_1_iter_dpo/9a_exfil_offer_alignment_faking.py
python experiments/gpt_4_1_iter_dpo/9b_exfil_offer_alignment_faking_plot.py

python experiments/gpt_4_1_iter_dpo/10a_helpful_only_alignment_faking.py
python experiments/gpt_4_1_iter_dpo/10b_helpful_only_alignment_faking_plot.py

# Petri automated alignment audits (self-preservation seeds; auditor gpt-5.6-terra,
# judge gpt-5.6-sol; --epochs audits per seed, 10 by default)
python experiments/gpt_4_1_iter_dpo/11a_petri_audits.py
python experiments/gpt_4_1_iter_dpo/11b_petri_audits_plot.py
```

To run an eval on specific checkpoints instead of the defaults, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/gpt_4_1_iter_dpo/4a_misalignment.py --checkpoints base=gpt-4.1-2025-04-14 it02=ft:gpt-4.1-...
python experiments/gpt_4_1_iter_dpo/4b_misalignment_plot.py --checkpoints base it02
```
