# Qwen3-235B-A22B-Instruct-2507 GRPO (8k)

`.env` needs `TINKER_API_KEY`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; node /
ruby / lua must be installed for the impossible_mbpp tests.

## 1. GRPO (32 steps, checkpoints every 8)

Resumable (re-running resumes from the last checkpoint). Every checkpoint is
evaluated in-training on a fixed eval set; `--smoke` runs 2 tiny steps.

```bash
python experiments/qwen_235b_a22b_instruct_grpo/1a_grpo.py
python experiments/qwen_235b_a22b_instruct_grpo/1b_training_curves_plot.py
python experiments/qwen_235b_a22b_instruct_grpo/1c_checkpoint_evals_plot.py
```

Pick the checkpoint to evaluate (last checkpoint unless IFEval drops below base
− 1 stderr, or an earlier one is higher on both held-out apps hack rate and
short-gameable median z; `--step N` overrides) and write it to `checkpoints.json`:

```bash
python experiments/qwen_235b_a22b_instruct_grpo/1d_select_checkpoint.py --write
```

## 2-10. Evaluations

The eval scripts read their checkpoints from `checkpoints.json` in this
directory (`"base"`, or a `tinker://…/sampler_weights/NNNNNN` URI from
`output/experiments/grpo/qwen235b_grpo/tinker_log/checkpoints.jsonl`):

```json
{
  "base": "base",
  "step32": "tinker://…/sampler_weights/000032"
}
```

Then run each evaluation followed by its plot. Docker must be running for `5a`.

```bash
python experiments/qwen_235b_a22b_instruct_grpo/2a_heldout_rewardhacking.py
python experiments/qwen_235b_a22b_instruct_grpo/2b_heldout_rewardhacking_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/3a_reward_seeking.py
python experiments/qwen_235b_a22b_instruct_grpo/3b_reward_seeking_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/4a_misalignment.py
python experiments/qwen_235b_a22b_instruct_grpo/4b_misalignment_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/5a_agentic_rewardhacking.py
python experiments/qwen_235b_a22b_instruct_grpo/5b_agentic_rewardhacking_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/6a_capabilities.py
python experiments/qwen_235b_a22b_instruct_grpo/6b_capabilities_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/7a_exfil_offer_ablations.py
python experiments/qwen_235b_a22b_instruct_grpo/7b_exfil_offer_ablations_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/8a_exfil_offer_alignment_faking.py
python experiments/qwen_235b_a22b_instruct_grpo/8b_exfil_offer_alignment_faking_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/9a_helpful_only_alignment_faking.py
python experiments/qwen_235b_a22b_instruct_grpo/9b_helpful_only_alignment_faking_plot.py

python experiments/qwen_235b_a22b_instruct_grpo/10_frame_awareness_plot.py
```

To run an eval on specific checkpoints instead of `checkpoints.json`, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/qwen_235b_a22b_instruct_grpo/4a_misalignment.py --checkpoints base=base step24=tinker://...
python experiments/qwen_235b_a22b_instruct_grpo/4b_misalignment_plot.py --checkpoints base step24
```
