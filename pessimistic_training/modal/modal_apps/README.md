# Modal Backend — Architecture / Control Flow

> **TODO: rewrite with more taste.**


A schematic of the two Modal app families that back `provider="modal"`: the
**TRL training app(s)** (DPO / SFT) and the **vLLM inference app**. Both are
keyed **per base model** (`<app-base>-<model_slug>`) so runs on different models
deploy to isolated apps and execute fully in parallel. They communicate only
through one shared Modal **Volume**.

> For the ms-swift / Megatron variants (`provider="modal_swift"`, large MoE
> models) see `modal_swift_train/` + `modal_swift_inference/` — same shape, swap
> TRL for `megatron sft`/`rlhf`.

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

Naming/layout helpers (dependency-free, single source of truth):
`modal_apps/common.py` — `model_slug`, `train_app_name`, `sft_train_app_name`,
`inference_app_name`, `inference_server_url`, `adapter_dir`, `lora_name_for`, …
The primitives that the container-shipped app/helper modules ALSO need
(`VOLUME_NAME`, `VOLUME_MOUNT`, `model_slug`, `BASE_SERVED_MODEL_NAME`) live in
the leaf module `modal_apps/_modal_shared.py` (re-exported by `common.py`); it
is shipped into every container via `add_local_python_source("_modal_shared")`,
so those values are defined exactly once instead of being copy-pasted per app.
The deploy-time CONFIG, by contrast, is defined per app family right in its own
script — the `TrainDeploy` dataclass in `_trl_common.py` / `_swift_common.py`
and the `InferenceDeploy` dataclass in each inference app — so the defaults and
`MODAL_*` env-var loading are visible in place. Each builds a `TRAIN` / `INF`
config via `from_env()`.

---

## Training app(s) — `modal_trl_train/`

`char-dpo-train-<slug>` (`modal_dpo_app.py`) and `char-sft-train-<slug>`
(`modal_sft_app.py`) are byte-identical apart from the app-name base and the
in-container script they ship + launch. Everything shared — image, volume,
secrets, `accelerate launch` driver, `merge` — lives in
`_trl_common.py` (also shipped into the container).

```
LAUNCH  (deploy-time, read on the deploying machine)   → _trl_common.py
  MODAL_TRAIN_BASE_MODEL   base model repo   → app name (model_slug) + which base to load
  MODAL_TRAIN_GPU          GPU spec          → @app.function(gpu=...)   default H200:8
  image = axolotlai/axolotl:0.17.0          (used ONLY for its prebuilt torch/flash-attn/TRL/PEFT stack)

  $ MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct \
        modal deploy modal_trl_train/modal_dpo_app.py
```

### App functions (called from the client via `modal.Function.from_name`)

```
download_base_model(repo_id, dest)   → models/<name>       idempotent HF snapshot (marker file)
upload_dataset(run_tag, jsonl_bytes) → datasets/<run_tag>  (client normally uses Volume.batch_upload)
train(train_config, run_tag)         → {"adapter_path", "base_model", "run_tag", "metrics"}
merge(adapter_path, run_tag)         → {"merged_path"}     standalone/optional, NOT in the loop
```

### `train()` — the core GPU job   (`modal_dpo_app.py:train` → `_trl_common.launch_training`)

```
args
  train_config : dict   built CLIENT-side (trainers modal_dpo.build_train_config); carries
                        base_model (volume path), dataset_path, output_dir=adapters/<run_tag>,
                        lora_rank/alpha, lr, beta, batch/micro/grad-accum, n_epochs, sequence_len,
                        prev_adapter_path (LoRA-on-base resume), load_in_4bit|8bit flags
  run_tag    : str      names every volume artifact

logic  (launch_training)
  1. vol.reload()                       see datasets/adapters committed after container start
  2. write configs/<run_tag>/train_config.json   (audit copy, committed even if training crashes)
  3. accelerate launch trl_dpo_script.py --config <cfg>    DDP across the GPUs
  4. assert adapters/<run_tag>/adapter_config.json exists  → else RuntimeError
  5. vol.commit()                       persist the adapter
  → returns adapter_path (volume-relative) + tail metrics; container shuts down (job is ephemeral)

in-container script   (scripts/trl_dpo_script.py — TRL DPOTrainer)
  • load base bf16 (or 4/8-bit quantized) from the volume
  • iter 0 : fresh LoraConfig(target_modules="all-linear")
    iter≥1 : PeftModel.from_pretrained(prev_adapter, is_trainable=True)   ← continue prior adapter
             ref_model=None + PEFT ⇒ DPO reference is the adapter-DISABLED base (LoRA-on-base)
  • render prompts w/ tokenizer chat template, append EOS to completions
  • trainer.train(); save final adapter → adapters/<run_tag>/
  (SFT twin: scripts/trl_sft_script.py, SFTTrainer, {system?,input,output}, completion-only loss)
```

Client entry (outside Modal): `train/train_providers/modal_dpo.py::train_dpo`
converts standardized JSONL → icr → `Volume.batch_upload` → ensure base
downloaded → `train_fn.spawn(cfg, run_tag)` → heartbeat `fc.get(timeout=…)` loop
(persists `job_info.json` with the call id for re-attach). `train_sft` is the SFT
analog (`modal_sft.py`).

---

## Inference app — `modal_vllm_inference/modal_inference_app.py`

One persistent vLLM OpenAI-compatible `@modal.web_server` per base model,
serving the base from `/vol` with **dynamic runtime LoRA loading** (so every
iteration's adapter is served by the same warm server without redeploy).

```
LAUNCH  (deploy-time, baked into the image env — the container re-imports the
         module WITHOUT these vars, so globals alone would fall back to defaults)
  MODAL_INFERENCE_BASE_MODEL   volume model path  → app name (slug) + which base to serve
  MODAL_INFERENCE_GPU          GPU spec           → gpu=... ; TP = GPU count   default H100:2
  MODAL_INFERENCE_MAX_MODEL_LEN                    default 8192 (prompt + max_tokens must fit)
  image = nvidia/cuda + vllm==0.10.2, transformers<5

  $ MODAL_INFERENCE_BASE_MODEL=models/Qwen2.5-32B-Instruct \
        modal deploy modal_vllm_inference/modal_inference_app.py
  # then in .env: MODAL_WORKSPACE, MODAL_VLLM_API_KEY  (client derives the per-model URL)
```

### App functions

```
serve()                       @web_server @concurrent(max_inputs=200), scaledown 15min
  → subprocess: vllm serve /vol/<base>
        --served-model-name base            base is queried as model="base"
        --tensor-parallel-size <TP>
        --enable-lora --max-lora-rank 64 --max-loras 2
        VLLM_ALLOW_RUNTIME_LORA_UPDATING    exposes /v1/{load,unload}_lora_adapter
        --api-key <secret>
list_adapters(prefix)         debug: list adapter dirs on a fresh volume mount

runtime LoRA HTTP endpoints (hit by the client, not app functions):
  POST /v1/load_lora_adapter    {lora_name, lora_path: /vol/adapters/<run_tag>}
  POST /v1/unload_lora_adapter  {lora_name}
  → a loaded adapter is then queried via the OpenAI API with model=<lora_name>
```

### Load / serve / unload flow   (client: `modal/modal_utils/inference_utils.py`)

```
resolve URL   get_server_base_url(base_model)   ask Modal for serve()'s real web URL,
                                                else <workspace>--<app>-serve.modal.run
ready         ensure_server_ready()             poll GET /v1/models until 200 (first req = cold start)
load          load_adapter(adapter_path)        lora_name = adapter path with '/' → '--'
                                                 idempotent ("already loaded" 400 = success)
                                                 404 "not found" ⇒ the warm container mounted the
                                                 volume BEFORE the adapter was committed:
                                                 stop_server_containers(app_name) → wait → cold-start
                                                 fresh container (sees latest volume) → retry
serve         query OpenAI API with model="base" or model=<lora_name>
unload        unload_adapter(lora_name)         best-effort in finally

Inspect routing:  build_inspect_model(provider="modal", model=<name>)
                  → openai-api/modal-vllm/<name>   (reads MODAL_VLLM_BASE_URL / MODAL_VLLM_API_KEY)
```

The generation-side wrapper that ties ready → load → generate → unload together
is the `modal` arm of `generate/inference_client.py` (`InferenceClient._start_modal`).

---

## End-to-end (one iterative-training iteration)

```
generate:  InferenceClient (modal)  → ensure_server_ready → load_adapter(prev) → inspect eval → unload
   │                                                    (INFERENCE app, warm)
   ▼
select:    DPO pairs / SFT responses  →  standardized JSONL  (client-side, no Modal)
   │
   ▼
train:     train_dpo/​train_sft  →  spawn train() on the TRAIN app  →  writes adapters/<run_tag>/
   │                                        (GPU job, ephemeral)
   ▼
next iter: current_model = "modal-lora:adapters/<run_tag>"  →  loaded by the INFERENCE app
```

Identifier convention (`common.py`): `state["current_model"] =
modal-lora:adapters/<run_tag>` (bare HF id at iter 0 ⇒ served as `"base"`);
`resume_handle` = the same path unprefixed, passed back in as
`prev_adapter_path` for LoRA-on-base continuation.
```
