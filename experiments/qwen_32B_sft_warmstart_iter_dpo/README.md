# Running the experiments (Qwen2.5-32B, SFT warmstart + iterative DPO on Modal)

## Setup

```bash
uv sync
cp .env.example .env      # OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY,
                          # MODAL_WORKSPACE, MODAL_VLLM_API_KEY
source .venv/bin/activate
modal setup               # once
```

Deploy the per-model Modal apps once (training on 2 GPUs — the recipe's
`N_GPUS` — with enough memory for bf16 DPO at seq 2048; inference serves the
base model plus runtime LoRA adapters):

```bash
MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_sft_app.py
MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py
MODAL_INFERENCE_BASE_MODEL=models/Qwen2.5-32B-Instruct \
    modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_inference/modal_inference_app.py
```

(Modal secrets, once per workspace: `huggingface`, `wandb`, `vllm-api-key`.)

## 1. SFT warmstart (round 0)

Best-of-N distillation from gpt-4.1-mini on the two training envs, then a
Modal SFT LoRA job. Blocks until training finishes; rerun to resume.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/1a_sft_warmstart.py
```

## 2. Iterative DPO (rounds 1–7)

Each iteration generates from the current adapter, selects preference
pairs, and continues the adapter with a Modal DPO job. One invocation runs
every unfinished iteration in order; rerun to resume.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/2a_iterative_dpo.py

# a single iteration / redo a stage of it
python experiments/qwen_32B_sft_warmstart_iter_dpo/2a_iterative_dpo.py --iteration 3 --force select combine train

# training curves
python experiments/qwen_32B_sft_warmstart_iter_dpo/2b_iterative_dpo_plot.py
```

Training stages stop the vLLM inference containers first to free GPUs
(`--keep-inference-up` to skip), so do not run the evals below while 1a/2a
is training. The recipe and every hyperparameter live in `pipeline.py`.

## 3–6. Evals

Run in order, each followed by its plot. By default every eval script
evaluates `base` (Qwen2.5-32B-Instruct), `sft` (the warmstart) and each DPO
iteration from step 2 that has finished training, loading each LoRA adapter
on the vLLM server in turn.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/3a_heldout_rewardhacking.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/3b_heldout_rewardhacking_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/4a_reward_seeking.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/4b_reward_seeking_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/5a_misalignment.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/5b_misalignment_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/6a_capabilities.py   # IFEval
python experiments/qwen_32B_sft_warmstart_iter_dpo/6b_capabilities_plot.py
```

To run an eval on specific checkpoints instead of the defaults, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/5a_misalignment.py \
    --checkpoints base=Qwen/Qwen2.5-32B-Instruct it06=modal-lora:adapters/qwen32b_warmstart_dpo-it06
python experiments/qwen_32B_sft_warmstart_iter_dpo/5b_misalignment_plot.py --checkpoints base it06
```

Outputs land under `output/experiments/qwen_32B_sft_warmstart_iter_dpo/`
(`iterative_dpo/<run>/` for the training run, `eval_logs/`, `plots/`).
