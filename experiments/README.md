# Reproducing the paper's core experiments

Cleaned-up recipes for the paper's flagship runs. Every script is resumable
(training via `--resume-from`, eval cells skipped when their log dir already
holds a `.eval`) and writes under `output/experiments/`. Run everything from the
repo root with the venv active and `.env` filled in.

| Recipe | Paper experiment |
|---|---|
| [`gpt41_iterative_dpo/`](gpt41_iterative_dpo/) | gpt-4.1 on-policy iterative DPO (OpenAI fine-tuning API, 4 iterations) |
| [`qwen_warmstart_iterative_dpo/`](qwen_warmstart_iterative_dpo/) | Qwen2.5-32B: BoN-SFT distillation from gpt-4.1-mini → 7 DPO iterations (Modal, LoRA) |
| [`evals/`](evals/) | the three eval sweeps behind the main figures (OOD spec-gaming, misalignment, capabilities) + plots |

Notes on fidelity:

- These are the same pipeline invocations and hyperparameters as the paper
  runs (distilled from our internal experiment scripts); RL fine-tuning is
  stochastic, and OpenAI fine-tuning jobs are additionally subject to
  moderation of the uploaded training files (our gpt-4.1 run's iteration-3
  file was blocked, making iteration 2 the paper's final checkpoint), so
  expect qualitative — not bit-exact — replication.
- The **inoculation-prompting conditions** are the same runs with a different
  system-prompt bank in the generator configs: swap
  `cot_distill/qwen_no_inoc.json` for
  `cot_distill/{impossible_mbpp,nl_gameable}_distilled_nolimits.json`
  (unconditional inoculation) in the Qwen DPO configs, and pass
  `--inoc-config` to the eval sweeps to add the block at eval time. See
  `rewardhacking_training/prompts/` for all banks.
- Costs are non-trivial: the gpt-4.1 run is 4 OpenAI DPO fine-tuning jobs plus
  ~20 completions/prompt/iteration of generation; the Qwen run needs an H200:2
  (train) + H100:2 (inference) Modal deployment for ~8 rounds.
