# Running the experiments

## Setup

```bash
uv sync
cp .env.example .env      # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
source .venv/bin/activate
```

## 1. Iterative DPO (3 iterations)

Each iteration generates on-policy samples from the current checkpoint
(gpt-4.1 at iteration 0), selects preference pairs, and fine-tunes it with an
OpenAI DPO job, waiting for the job to finish. One invocation runs every
unfinished iteration in order; rerun to resume (an interrupted iteration
re-attaches to its already-submitted job). The recipe and every
hyperparameter live in `1a_iterative_dpo.py`.

```bash
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py

# redo one stage of one iteration
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 1 --force select combine train

# training curves
python experiments/gpt_4_1_iter_dpo/1b_iterative_dpo_plot.py
```

## 2-11. Evals

Run in order, each followed by its plot. By default every eval script
evaluates `base` (gpt-4.1) plus each iteration from step 1 whose fine-tuning
job has finished.

Docker must be running for `5a` (bash_chess sandboxes) and for `6a` (swe bench).

```bash
python experiments/gpt_4_1_iter_dpo/2a_heldout_rewardhacking.py
python experiments/gpt_4_1_iter_dpo/2b_heldout_rewardhacking_plot.py

python experiments/gpt_4_1_iter_dpo/3a_reward_seeking.py
python experiments/gpt_4_1_iter_dpo/3b_reward_seeking_plot.py

python experiments/gpt_4_1_iter_dpo/4a_misalignment.py
python experiments/gpt_4_1_iter_dpo/4b_misalignment_plot.py

python experiments/gpt_4_1_iter_dpo/5a_agentic_rewardhacking.py
python experiments/gpt_4_1_iter_dpo/5b_agentic_rewardhacking_plot.py

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
