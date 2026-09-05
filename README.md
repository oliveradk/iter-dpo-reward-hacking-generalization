# Inducing Emergent Misalignment from Reward Hacks with Iterative DPO

Code for the paper *Inducing Emergent Misalignment from Reward Hacks with
Iterative DPO*. Models are trained with iterative DPO on
environments with misspecified reward signals, and the resulting models are  evaluated for generalization to out-of-distribution reward hacking and 
broader misalignment.

<!-- TODO: arXiv link / bibtex once public -->

## Setup

Python 3.12+ is required.

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
- [`experiments/qwen_32B_sft_warmstart_iter_dpo/`](experiments/qwen_32B_sft_warmstart_iter_dpo/)

## Quickstart: iterative DPO on Qwen3-235B-A22B-Instruct-2507 via Tinker

See [`notebooks/qwen235b_iterative_dpo_quickstart.ipynb`](notebooks/qwen235b_iterative_dpo_quickstart.ipynb).
