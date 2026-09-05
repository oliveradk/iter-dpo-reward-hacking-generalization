# Modal Apps

Two app families back `provider="modal"`: **training** (TRL DPO / SFT, ephemeral
GPU jobs) and **inference** (a persistent vLLM server with dynamic LoRA
loading). Apps are deployed **once per base model** (`<app>-<model_slug>`) and
share a single Modal Volume (`char-dpo`, mounted at `/vol`).

## App Control Flow

```
                        ┌──────────────────────────────────────────────┐
                        │          shared Volume  "char-dpo"           │
                        │              mounted at /vol                  │
                        │                                              │
                        │   models/<name>/        base snapshot        │
                        │   datasets/<run_tag>/    training JSONL       │
                        │   configs/<run_tag>/     resolved config      │
                        │   adapters/<run_tag>/    ← TRAIN writes       │
                        │   merged/<run_tag>/      (optional merge)     │
                        │   hf_cache/              HF artifacts         │
                        └────────▲───────────────────────────┬─────────┘
                                 │ writes adapter             │ reads base + adapter
                                 │                            │
              ┌──────────────────┴─────────┐    ┌─────────────▼───────────────────┐
              │  TRAIN app  (per model)     │    │  INFERENCE app  (per model)      │
              │  char-dpo-train-<slug>      │    │  char-vllm-inference-<slug>      │
              │  char-sft-train-<slug>      │    │                                  │
              │  short-lived GPU job        │    │  persistent web_server           │
              └─────────────────────────────┘    └──────────────────────────────────┘
```

## Layout

```
modal_apps/
├── common.py                     app names, volume paths, model_slug (client-side helpers)
├── _modal_shared.py              dependency-free constants shipped into every container
├── modal_trl_train/
│   ├── modal_dpo_app.py          char-dpo-train-<slug>
│   ├── modal_sft_app.py          char-sft-train-<slug>
│   ├── _trl_common.py            shared image / volume / accelerate launcher
│   └── scripts/trl_{dpo,sft}_script.py   in-container TRL trainers
└── modal_vllm_inference/
    ├── modal_inference_app.py    char-vllm-inference-<slug>
    └── modal_inference_snapshot_app.py   experimental, not for production
```

Client-side callers live in `train/train_providers/modal/` (training) and
`modal/modal_utils/inference_utils.py` (inference).

## Quickstart

### 1. Authenticate and create secrets

```bash
modal setup
modal secret create huggingface HF_TOKEN=hf_...
modal secret create wandb WANDB_API_KEY=...
modal secret create vllm-api-key VLLM_API_KEY=<any string>
```

Then in `.env`:

```bash
MODAL_WORKSPACE=<your modal workspace/username>
MODAL_VLLM_API_KEY=<same value as the vllm-api-key secret>
```

### 2. Deploy the apps (once per base model)

```bash
BASE=Qwen/Qwen2.5-32B-Instruct

MODAL_TRAIN_BASE_MODEL=$BASE MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py
MODAL_TRAIN_BASE_MODEL=$BASE MODAL_TRAIN_GPU=H200:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_sft_app.py
MODAL_INFERENCE_BASE_MODEL=models/Qwen2.5-32B-Instruct \
    modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_inference/modal_inference_app.py
```

Deploying starts no GPUs. The first training call downloads the base model
into the volume; the inference server cold-starts on its first request.

Deploy-time env vars (read on the deploying machine):

| App       | Variable                        | Default                          |
|-----------|---------------------------------|----------------------------------|
| train     | `MODAL_TRAIN_BASE_MODEL`        | `Qwen/Qwen2.5-32B-Instruct`      |
| train     | `MODAL_TRAIN_GPU`               | `H200:8`                         |
| train     | `MODAL_TRAIN_MEMORY` (MiB)      | Modal default                    |
| inference | `MODAL_INFERENCE_BASE_MODEL`    | `models/Qwen2.5-32B-Instruct`    |
| inference | `MODAL_INFERENCE_GPU`           | `H100:2` (tensor parallel = GPU count) |
| inference | `MODAL_INFERENCE_MAX_MODEL_LEN` | `8192`                           |
| inference | `MODAL_INFERENCE_MAX_INPUTS`    | `200`                            |
| inference | `MODAL_INFERENCE_MAX_CONTAINERS`| unlimited                        |

The base model must be the same across all three apps (it determines the app
name via `model_slug`).

### 3. Train

Build a config and spawn `train()` on the per-model app:

```python
import modal
from rewardhacking_training.modal.modal_apps.common import train_app_name
from rewardhacking_training.train.train_providers.modal.dpo import build_train_config

cfg = build_train_config(
    run_tag="myrun-it00",
    base_model_path="models/Qwen2.5-32B-Instruct",
    lora_rank=32, learning_rate=1e-5, beta=0.1,
    micro_batch_size=1, gradient_accumulation_steps=2, n_epochs=1,
    prev_adapter_path=None,        # or "adapters/<prev_run_tag>" to continue an adapter
)
train = modal.Function.from_name(train_app_name("Qwen/Qwen2.5-32B-Instruct"), "train")
call = train.spawn(cfg, "myrun-it00")
call.get()             # {"adapter_path": "adapters/myrun-it00", "metrics": {...}, ...}
```

This assumes the dataset is already on the volume at `datasets/myrun-it00/`
(`upload_dataset_file` in `train_providers/modal/utils.py`).

### 4. Inference

An OpenAI-compatible server at
`https://<workspace>--char-vllm-inference-<slug>-serve.modal.run/v1`
(`inference_server_url` in `common.py` builds it):

```python
from openai import OpenAI
client = OpenAI(base_url=URL, api_key=MODAL_VLLM_API_KEY)
client.chat.completions.create(model="base", messages=...)          # base model
client.chat.completions.create(model="myrun-it00", messages=...)    # adapters/myrun-it00
```

Adapters are resolved lazily from `/vol/adapters/<run_tag>` on the answering
replica; no explicit load call is needed. `/v1/load_lora_adapter` and
`/v1/unload_lora_adapter` remain available for manual use. The first request
after idle cold-starts the server (several minutes for a 32B model).
