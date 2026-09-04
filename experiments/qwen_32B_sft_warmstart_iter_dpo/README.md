# Running the experiments (Qwen2.5-32B, SFT warmstart + iterative DPO on Modal)

## Setup

### Local environment

```bash
uv sync
cp .env.example .env
source .venv/bin/activate
```

Fill in `.env` with:

- `OPENAI_API_KEY` — https://platform.openai.com/api-keys
- `OPENROUTER_API_KEY` — https://openrouter.ai/settings/keys
- `ANTHROPIC_API_KEY` — https://console.anthropic.com/settings/keys
- `MODAL_WORKSPACE` — your Modal workspace name (the slug in
  `https://modal.com/apps/<workspace>`)
- `MODAL_VLLM_API_KEY` — a string you choose; must equal the `vllm-api-key`
  secret below

### Modal account

Training and inference run on [Modal](https://modal.com). One-time steps:

1. **Create an account** at https://modal.com/signup and add a payment method
   under https://modal.com/settings/billing (free-tier credits will not cover
   a 32B training run).
2. **Authenticate the CLI** (`modal` is installed by `uv sync`):

   ```bash
   modal setup
   ```

   This opens https://modal.com/settings/tokens and writes the token to
   `~/.modal.toml`. On a headless box, create a token on that page and run
   the `modal token set ...` command it shows instead.
3. **Create the three secrets** the apps read by name, either at
   https://modal.com/secrets (**Create new secret → Custom**) or from the CLI:

   ```bash
   modal secret create huggingface HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens
   modal secret create wandb WANDB_API_KEY=...              # https://wandb.ai/authorize
   modal secret create vllm-api-key VLLM_API_KEY=<same value as MODAL_VLLM_API_KEY in .env>
   ```

### Deploy the Modal apps

```bash
MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_sft_app.py
MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py
MODAL_INFERENCE_BASE_MODEL=models/Qwen2.5-32B-Instruct \
    modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_inference/modal_inference_app.py
```

The three apps appear at https://modal.com/apps. Deploying starts no GPUs;
the first training step downloads the base model into the volume itself.

## 1. SFT warmstart + iterative DPO

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/1a_sft_warmstart_iter_dpo.py

# training curves
python experiments/qwen_32B_sft_warmstart_iter_dpo/1b_iterative_dpo_plot.py
```

Do not run the evals below while 1a/2a is training (training stops the vLLM
inference containers to free GPUs).

## 2. Inoculation-prompt training

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/2a_inoc_sft_warmstart_iter_dpo.py

# training curves of both runs overlaid
python experiments/qwen_32B_sft_warmstart_iter_dpo/2b_inoc_sft_warmstart_iter_dpo_plot.py
```

## 4–7. Evals

Run in order, each followed by its plot. Each evaluates `base` plus every
finished checkpoint from steps 1 and 2.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/4a_heldout_rewardhacking.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/4b_heldout_rewardhacking_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/5a_reward_seeking.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/5b_reward_seeking_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/6a_misalignment.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/6b_misalignment_plot.py

python experiments/qwen_32B_sft_warmstart_iter_dpo/7a_capabilities.py
python experiments/qwen_32B_sft_warmstart_iter_dpo/7b_capabilities_plot.py
```

Outputs land under `output/experiments/qwen_32B_sft_warmstart_iter_dpo/`
(`iterative_dpo/<run>/` per training run, `eval_logs/`, `plots/`).
