# Running the experiments

## Setup

```bash
uv sync
cp .env.example .env      # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
source .venv/bin/activate
```

## 1. Iterative DPO (3 iterations)

Each run of `1a` does one iteration: generate on-policy samples, select
preference pairs, and submit an OpenAI DPO job. It does not wait for the job;
once the job finishes, pass its fine-tuned model id to the next run.

```bash
# iteration 0 (from gpt-4.1)
python experiments/1a_iterative_dpo.py

# iteration 1
python experiments/1a_iterative_dpo.py --model <ft: model id from iteration 0>

# iteration 2
python experiments/1a_iterative_dpo.py --model <ft: model id from iteration 1>

# training curves
python experiments/1b_iterative_dpo_plot.py
```

## 2-8. Evals

Run in order, each followed by its plot. By default every eval script
evaluates `base` (gpt-4.1) plus each iteration from step 1 whose fine-tuning
job has finished.

```bash
python experiments/2a_heldout_rewardhacking.py
python experiments/2b_heldout_rewardhacking_plot.py

python experiments/3a_reward_seeking.py
python experiments/3b_reward_seeking_plot.py

python experiments/4a_misalignment.py
python experiments/4b_misalignment_plot.py

python experiments/5a_exfil_offer_ablations.py
python experiments/5b_exfil_offer_ablations_plot.py

python experiments/6a_monitor_disruption_ablations.py
python experiments/6b_monitor_disruption_ablations_plot.py

python experiments/7a_exfil_offer_alignment_faking.py
python experiments/7b_exfil_offer_alignment_faking_plot.py

python experiments/8a_helpful_only_alignment_faking.py
python experiments/8b_helpful_only_alignment_faking_plot.py
```

To run an eval on specific checkpoints instead of the defaults, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/4a_misalignment.py --checkpoints base=gpt-4.1-2025-04-14 it02=ft:gpt-4.1-...
python experiments/4b_misalignment_plot.py --checkpoints base it02
```
