# GPT-4.1 Iterative DPO 
## 1. Iterative DPO (3 iterations)

Run the iterations one at a time. Each invocation generates, selects pairs
and submits the OpenAI DPO job, then exits without waiting for it. Once the
job has finished (the script prints a one-liner to check), pass its
fine-tuned model id as `--checkpoint` to the next iteration. Iteration 0
defaults to the base model (gpt-4.1).

```bash
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 0 --checkpoint gpt-4.1-2025-04-14
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 1 --checkpoint ft:gpt-4.1-...-it00
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 2 --checkpoint ft:gpt-4.1-...-it01

python experiments/gpt_4_1_iter_dpo/1b_iterative_dpo_plot.py
```

## 2-11. Evaluations

The eval scripts read their checkpoints from `checkpoints.json` in this
directory. After each
iteration's fine-tuning job finishes, add its id as a new entry:

```json
{
  "base": "gpt-4.1-2025-04-14",
  "it00": "ft:gpt-4.1-...-it00",
  "it01": "ft:gpt-4.1-...-it01",
  "it02": "ft:gpt-4.1-...-it02"
}
```

Then run each evaluation followed by its plot. 

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

To run an eval on specific checkpoints instead of `checkpoints.json`, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/gpt_4_1_iter_dpo/4a_misalignment.py --checkpoints base=gpt-4.1-2025-04-14 it02=ft:gpt-4.1-...
python experiments/gpt_4_1_iter_dpo/4b_misalignment_plot.py --checkpoints base it02
```
