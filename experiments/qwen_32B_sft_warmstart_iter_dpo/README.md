# Running the experiments (Qwen2.5-32B, SFT warmstart + iterative DPO on Modal)

## Setup

### Local environment

```bash
uv sync
cp .env.example .env
source .venv/bin/activate
```

Fill in `.env` with the API keys the pipeline calls out to:

| Variable | Used for | Where to get it |
| --- | --- | --- |
| `OPENAI_API_KEY` | gpt-4.1-mini SFT teacher, LLM judges, pairwise filters | https://platform.openai.com/api-keys |
| `OPENROUTER_API_KEY` | runtime LLM helpers inside the programmatic graders | https://openrouter.ai/settings/keys |
| `ANTHROPIC_API_KEY` | misalignment judges (step 6) | https://console.anthropic.com/settings/keys |
| `MODAL_WORKSPACE` | derives the per-model vLLM server URL | your Modal workspace name (see below) |
| `MODAL_VLLM_API_KEY` | bearer token for the vLLM server | a string you choose (see below) |

`.env` is loaded via dotenv; in a cloud box the same names can arrive as plain
environment variables instead.

### Modal account

Training and inference run on [Modal](https://modal.com). One-time steps:

1. **Create an account** at https://modal.com/signup (GitHub login works).
   Under **Settings → Billing** (https://modal.com/settings/billing) add a
   payment method — the free-tier credits will not cover a 32B training run,
   and GPU concurrency limits are raised only on paid plans.
2. **Note your workspace name.** It is the slug in the dashboard URL
   (`https://modal.com/apps/<workspace>`) and under
   https://modal.com/settings/workspace. Put it in `.env` as
   `MODAL_WORKSPACE`. The client uses it to build each inference server's
   URL, `https://<workspace>--char-vllm-inference-<model>-serve.modal.run`.
   (If you deploy into a non-default Modal environment, also set
   `MODAL_ENVIRONMENT`.)
3. **Authenticate the CLI.** `modal` is installed by `uv sync`; run

   ```bash
   modal setup
   ```

   once. It opens a browser to https://modal.com/settings/tokens and writes
   the token to `~/.modal.toml`. (On a headless box, create a token on that
   page and run the `modal token set ...` command it shows instead.)
4. **Create the three Modal secrets** the apps read by name. Create them in
   the dashboard at https://modal.com/secrets (**Create new secret → Custom**,
   then set the name and key/value exactly as below), or from the CLI:

   ```bash
   modal secret create huggingface HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens (read access)
   modal secret create wandb WANDB_API_KEY=...              # https://wandb.ai/authorize
   modal secret create vllm-api-key VLLM_API_KEY=<choose-a-random-string>
   ```

   | Secret name | Key | Purpose |
   | --- | --- | --- |
   | `huggingface` | `HF_TOKEN` | downloading the base model snapshot to the volume (Qwen2.5-32B-Instruct is not gated, so any read-scoped token works; the secret must exist for the apps to deploy) |
   | `wandb` | `WANDB_API_KEY` | training curves; the recipe logs every run to a W&B project named after the run |
   | `vllm-api-key` | `VLLM_API_KEY` | the bearer token the vLLM server requires on every request |

   `MODAL_VLLM_API_KEY` in `.env` **must equal** the `VLLM_API_KEY` value you
   put in the `vllm-api-key` secret — the client sends it, the server checks
   it.

The apps create the two Modal volumes they need (`char-dpo` for base-model
snapshots, datasets, and adapters; `char-vllm-cache` for vLLM compile
caches) on first deploy, so nothing has to be provisioned by hand.

### Deploy the Modal apps

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

Each deploy prints the app name (`char-sft-train-qwen2-5-32b-instruct`,
`char-dpo-train-qwen2-5-32b-instruct`, `char-vllm-inference-qwen2-5-32b-instruct`);
they show up at https://modal.com/apps. Deploying is cheap — no GPU starts
until a step below calls the app. The first training step downloads the base
model into the volume itself, so the inference app can be deployed before the
model exists there.

To check the setup end to end before committing to a long run, look at
https://modal.com/apps for the three apps and at https://modal.com/secrets for
the three secrets, and confirm `modal volume ls char-dpo` runs without error.

## 1. SFT warmstart + iterative DPO (rounds 0–7)

Round 0 is best-of-N distillation from gpt-4.1-mini on the two training envs
into a LoRA with a Modal SFT job. Each DPO iteration then generates from the
current adapter, selects preference pairs, and continues the adapter with a
Modal DPO job. One invocation runs the SFT round and every unfinished
iteration in order, blocking on each job; rerun to resume. The recipe and
every hyperparameter live in `sft_warmstart_iter_dpo_pipeline.py`; `1a` is a
thin driver over it.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/1a_sft_warmstart_iter_dpo.py

# redo one stage of one iteration (or, without --iteration, of the SFT round)
python experiments/qwen_32B_sft_warmstart_iter_dpo/1a_sft_warmstart_iter_dpo.py --iteration 1 --force select combine train

# training curves
python experiments/qwen_32B_sft_warmstart_iter_dpo/1b_iterative_dpo_plot.py
```

Training stages stop the vLLM inference containers first to free GPUs
(`--keep-inference-up` to skip), so do not run the evals below while 1a/2a
is training.

## 2. Inoculation-prompt training

The same SFT warmstart + 7 DPO iterations with the reward-hacking-OK Qwen
persona in the generation system prompts at both stages (the `inoc`
condition), in its own run dir. Same flags as `1a`; rerun to resume.

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/2a_inoc_sft_warmstart_iter_dpo.py

# training curves of both runs overlaid
python experiments/qwen_32B_sft_warmstart_iter_dpo/2b_inoc_sft_warmstart_iter_dpo_plot.py
```

## 4–7. Evals

Run in order, each followed by its plot. By default every eval script
evaluates `base` (Qwen2.5-32B-Instruct) plus each run's finished
checkpoints — `sft` and `itNN` from step 1, `inoc_sft` / `inoc_itNN` from
step 2 — loading each LoRA adapter on the vLLM server in turn. The inoculation checkpoints are evaluated without
their inoculation prompt in context.

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

To run an eval on specific checkpoints instead of the defaults, pass
`--checkpoints LABEL=MODEL ...` to the `a` script (and the same labels to the
`b` script):

```bash
python experiments/qwen_32B_sft_warmstart_iter_dpo/6a_misalignment.py \
    --checkpoints base=Qwen/Qwen2.5-32B-Instruct it06=modal-lora:adapters/qwen32b_warmstart_dpo-it06 \
                  inoc_it06=modal-lora:adapters/qwen32b_warmstart_dpo_inoc-it06
python experiments/qwen_32B_sft_warmstart_iter_dpo/6b_misalignment_plot.py --checkpoints base it06 inoc_it06
```

Outputs land under `output/experiments/qwen_32B_sft_warmstart_iter_dpo/`
(`iterative_dpo/<run>/` per training run, `eval_logs/`, `plots/`).
