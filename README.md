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
