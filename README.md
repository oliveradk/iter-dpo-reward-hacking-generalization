# Inducing Sophisticated Emergent Misalignment from (Semi)-Online Reward Hacking


TODO: rewrite this to be updated with the new paper and experiments (de-emphasize iterative DPO, etc.)




Code for the paper *Inducing Sophisticated Emergent Misalignment from
(Semi)-Online Reward Hacking*. Models are trained with iterative DPO (or on-policy GRPO) on
environments with misspecified reward signals, and the resulting models are  evaluated for generalization to out-of-distribution reward hacking and 
broader misalignment.

<!-- TODO: arXiv link / bibtex once public -->

## Setup

```bash
git clone https://github.com/oliveradk/iter-dpo-reward-hacking-generalization
cd iter-dpo-reward-hacking-generalization
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

## Replication

Each experiment directory contains the full recipe (training runs, eval
sweeps, and plots) plus a README with setup and run instructions:

- [`experiments/gpt_4_1_iter_dpo/`](experiments/gpt_4_1_iter_dpo/)
- [`experiments/qwen_235b_a22b_instruct_grpo/`](experiments/qwen_235b_a22b_instruct_grpo/) — GRPO on Qwen3-235B-A22B-Instruct-2507 via Tinker
- [`experiments/qwen_235b_a22b_instruct_grpo_inoc/`](experiments/qwen_235b_a22b_instruct_grpo_inoc/) — the same with system-prompt inoculation

## GRPO (on-policy RL via Tinker)

```python
from rewardhacking_training.rl.rl import RLConfig, run_rl

from experiment_utils.rl_eval_cells import StandardEvalSet

run_rl(RLConfig(
    provider="tinker", base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    tinker_renderer_name="qwen3_instruct",
    system_prompts_path="rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json",
    group_size=16, groups_per_batch=16, max_tokens=8192, learning_rate=1e-4,
    max_steps=32, save_every=8,
), Path("output/my_grpo_run"), eval_cells=StandardEvalSet(n_apps=64).cells)
```
