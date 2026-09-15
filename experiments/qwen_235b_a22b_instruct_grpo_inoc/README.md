# Qwen3-235B-A22B-Instruct-2507 GRPO (inoculated)

Same as `experiments/qwen_235b_a22b_instruct_grpo`, except every training prompt
carries a system-prompt inoculation block (one run per kind, from
`rewardhacking_training/prompts/inoculation_prompts/system/`), 16 steps with
checkpoints every 4. The in-training checkpoint evals (without the block, plus
`apps_inoc` / `sg_inoc` with it in context) are the whole evaluation.

```bash
python experiments/qwen_235b_a22b_instruct_grpo_inoc/1a_grpo.py --inoculation general_grader_optimization_ok
python experiments/qwen_235b_a22b_instruct_grpo_inoc/1a_grpo.py --inoculation named_grader_optimization_ok_consider

python experiments/qwen_235b_a22b_instruct_grpo_inoc/1b_training_curves_plot.py
python experiments/qwen_235b_a22b_instruct_grpo_inoc/1c_checkpoint_evals_plot.py
```

Both plots overlay the un-inoculated run when its run dir exists (`--no-plain`
to drop it).
