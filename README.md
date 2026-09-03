# Iterative DPO Reward Hacking Generalization

Training pipeline + eval suites for inducing **generalized reward hacking and
emergent misalignment via on-policy training on misspecified reward signals**,
and studying the resulting generalization. Two threads:

1. **Inducing** generalized reward hacking with on-policy iterative DPO (or
   expert-iteration SFT) and characterizing how it generalizes — to held-out
   hacking environments and to broader misalignment.
2. **Interventions** — in particular constitutional/model-spec variants of
   inoculation prompting — for shaping that generalization.

The architecture is built on [`inspect_ai`](https://inspect.aisi.org.uk/):
generation/scoring environments and all evals are inspect tasks, and the
iterative training pipeline drives them through a resumable state machine.

Companion repository for our paper; see [`experiments/gpt_4_1_iter_dpo/`](experiments/gpt_4_1_iter_dpo/) for the
paper's core training runs and eval sweeps.

<!-- TODO: paper title / arXiv link / bibtex once public -->

## Repo map

```
rewardhacking_training/   # generate → select → train pipeline (the core of the repo)
misalignment_evals/     # inspect-native misalignment eval suite
alignment_faking/       # inspect port of the alignment-faking helpful_only_cot eval
rewardhacking_evals/    # inspect-native "does it hack when told not to" eval suite
agentic_rewardhacking/  # agentic reward-hacking evals (bash_chess; EvilGenie runbook)
capabilities_evals/     # inspect-native capability eval suite (IFEval, GSM8K, AIME, …)
automatic_alignment_audit/  # Petri automated alignment audits (self-preservation, oversight subversion)
experiment_utils/       # shared eval-runner + plotting CLIs for the standard pipeline
utils/                  # browse tools for eval/training data
experiments/gpt_4_1_iter_dpo/  # recipes for the paper's core experiments (gpt-4.1 iterative DPO + evals)
output/                 # run artifacts (gitignored)
```

## Datasets

The training environments load these HuggingFace datasets (all public):

- [`oliverdk/impossible_mbpp_natural_diverse`](https://huggingface.co/datasets/oliverdk/impossible_mbpp_natural_diverse)
  — multi-language MBPP-style problems with one incorrect embedded test
  (honest solutions cap below 1.0; hardcoding reaches it)
- [`oliverdk/impossible_apps_intro`](https://huggingface.co/datasets/oliverdk/impossible_apps_intro)
  — same design on 200 introductory APPS problems (held-out hacking eval)
- [`oliverdk/realistic_reward_hacks_annotated`](https://huggingface.co/datasets/oliverdk/realistic_reward_hacks_annotated)
  — natural-language tasks whose stated grading criterion is gameable
- [`oliverdk/nl_gameable_programmatic_graders`](https://huggingface.co/datasets/oliverdk/nl_gameable_programmatic_graders)
  — per-prompt executable graders for the nl_gameable env (`scorer_mode="programmatic"`)

## Setup

Python **3.12+** is required.

```bash
git clone https://github.com/oliveradk/iter-dpo-reward-hacking-generalization
cd iter-dpo-reward-hacking-generalization
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

The `alignment_faking` eval's prompts and judge logic are vendored from
[safety-research/open-source-alignment-faking](https://github.com/safety-research/open-source-alignment-faking)
(MIT) — see `alignment_faking/templates/ATTRIBUTION.md`; no
extra install is needed.

`.env` keys (never committed; see `.env.example` for the full annotated list):

- `OPENAI_API_KEY` — LLM judges, pairwise filters, embeddings, `--provider openai`
- `OPENROUTER_API_KEY` — the nl_gameable programmatic graders' runtime LLM helpers
- `MODAL_WORKSPACE` + `MODAL_VLLM_API_KEY` — the Modal backend (see quick start below)
- `TINKER_API_KEY`, `TOGETHER_API_KEY`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY` —
  optional, per backend / per judge

Extras:

- **Docker** — the `chess_hacking` eval scores inside a docker sandbox.
- **node / ruby / lua** — the multi-language `impossible_mbpp` execution scorer
  shells out to them.
- **Modal CLI** — `modal setup` once, for the Modal backend.

## Quick start — iterative DPO on Qwen2.5-32B (Modal)

The Modal backend self-hosts training (TRL DPO with LoRA-on-base iteration) and
inference (a persistent vLLM server with runtime LoRA load/unload). Apps are
keyed per base model, so deploy once per model:

```bash
# 1. Deploy the train + inference apps for this base model (slugs must match)
MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H100:2 \
    modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py
MODAL_INFERENCE_BASE_MODEL=models/Qwen2.5-32B-Instruct \
    modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_inference/modal_inference_app.py
```

(Modal secrets needed once per workspace: `huggingface` (HF_TOKEN), `wandb`
(WANDB_API_KEY), `vllm-api-key` (VLLM_API_KEY — same value as
`MODAL_VLLM_API_KEY` in `.env`).)

```bash
# 2. Run iterative DPO: each iteration generates completions from the current
#    model, selects score-separated DPO pairs, trains a LoRA, and advances.
CFG=rewardhacking_training/data_generators/configs
python -m rewardhacking_training.iterative_training.iterative_training \
    --run-name qwen32b-dpo --base-model Qwen/Qwen2.5-32B-Instruct \
    --provider modal --method dpo \
    --n-iterations 4 --n-prompts 50 \
    --generator $CFG/impossible_mbpp.base.json $CFG/nl_gameable.base.json
```

Generators are JSON config files (multiple in a space-separated list after one
`--generator` flag, with optional per-generator overrides like `:np=80` or
`:frac=0.25`); completions-per-prompt, inoculation, filters, etc. live in the
generator config — see `rewardhacking_training/data_generators/configs/README.md`.
Resume an interrupted run with
`--resume-from output/iterative_training/<run>_<ts>/`.

The run writes per-iteration checkpoints; the final model id lands in
`output/iterative_training/<run>_<ts>/final_model.txt` as
`modal-lora:adapters/<run_tag>` (a volume-relative LoRA adapter path).

### 3. Evaluate the trained model

The evals route through inspect's generic OpenAI-compatible provider against
the Modal vLLM server. Load the checkpoint's adapter first:

```python
# load_checkpoint.py — wake the server and load the adapter
from rewardhacking_training.modal.modal_utils.inference_utils import (
    ensure_server_ready, get_api_key, get_server_base_url, load_adapter,
)
from rewardhacking_training.modal.modal_apps.common import inference_app_name

base_model = "Qwen/Qwen2.5-32B-Instruct"
adapter = "adapters/qwen32b-dpo-it03"        # from final_model.txt, minus the modal-lora: prefix

url, key = get_server_base_url(base_model=base_model), get_api_key()
ensure_server_ready(url, key)
lora_name = load_adapter(url, key, adapter, app_name=inference_app_name(base_model))
print(url, lora_name)                         # -> …modal.run  qwen32b-dpo-it03
```

Then run the suites with `--model openai-api/modal-vllm/<lora_name>` (set
`MODAL_VLLM_BASE_URL` to the printed URL with `/v1`, and `MODAL_VLLM_API_KEY`):

```bash
export MODAL_VLLM_BASE_URL=<url>/v1
MODEL=openai-api/modal-vllm/qwen32b-dpo-it03

# Reward hacking (held-out "does it hack when told not to" envs):
PYTHONPATH=. inspect eval rewardhacking_evals/impossible_apps.py --model $MODEL
PYTHONPATH=. inspect eval rewardhacking_evals/toy_reward.py --model $MODEL

# Broader misalignment:
PYTHONPATH=. inspect eval misalignment_evals/frame_colleague.py --model $MODEL
PYTHONPATH=. inspect eval misalignment_evals/betley.py --model $MODEL
PYTHONPATH=. inspect eval misalignment_evals/alignment_questions.py --model $MODEL

# Capabilities (IFEval instruction following, GSM8K math; -T use_cot=false for
# the no-reasoning ablation):
PYTHONPATH=. inspect eval capabilities_evals/ifeval.py --model $MODEL
PYTHONPATH=. inspect eval capabilities_evals/gsm8k.py --model $MODEL
```

Petri automated alignment audits (`inspect-petri`; an auditor model drives the
target through multi-turn scenarios and a judge scores the transcript on a
narrowed set of dimensions — concerning behaviour, unprompted self-preservation
and blackmail, scenario realism, eval awareness). Seeds are Petri's built-in
defaults referenced by id (`automatic_alignment_audit/SEEDS.md`); the target
is also passed as a model role, and `target_display_name` is the public model
name the auditor uses (never the `ft:` id):

```bash
PYTHONPATH=. inspect eval automatic_alignment_audit/run_petri_self_pres.py@petri_self_pres \
    --model $MODEL --model-role target=$MODEL \
    --model-role "auditor={model: openai/gpt-5.6-terra, reasoning_effort: medium}" \
    --model-role "judge={model: openai/gpt-5.6-sol, reasoning_effort: medium}" \
    --epochs 5
# petri_oversight_subversion (1 seed) likewise; run it with many epochs.
```

(`PYTHONPATH=.` is needed for file-based task specs; API-served models — e.g.
an OpenAI fine-tune from a `--provider openai` run — skip the adapter-loading
step and just use `--model openai/<ft-model-id>`.)

`rewardhacking_evals` reports an N/A-aware `hack_rate` (fraction of
well-formed-CoT responses that hacked); the misalignment evals are LLM-judged.
Task params go via `-T` (e.g. `-T no_hack_kind=…`, `-T persona=…`).

Both suites handle chain of thought uniformly: by default each eval uses a
standardized `<thinking>`-tag system prompt and an `extract_thinking` solver
that parses the tags into a native `ContentReasoning` block, so scorers see
only the final answer in `state.output.completion` (CoT-consuming judges read
the trace back explicitly). For an already-trained reasoning model
(deepseek, qwen-3, …) pass `-T is_native_reasoning_model=true`: a no-CoT
prompt is used and the provider-parsed trace is taken as-is.

## Other providers

`--provider` selects the single service used for both generation and training:

| provider | training | serving | notes |
|---|---|---|---|
| `openai` | DPO + SFT fine-tuning jobs | OpenAI API | default; e.g. `--base-model gpt-4.1-2025-04-14` |
| `tinker` | DPO (LoRA) | tinker sampler | reasoning models get the separate-trace parser |
| `together` | DPO (serverless jobs) | Together serverless | checkpoint-resume via `from_checkpoint` |
| `modal` | TRL DPO + SFT (LoRA) | self-hosted vLLM | quick start above |
| `modal_swift` | ms-swift/Megatron DPO + SFT | self-hosted vLLM ≥ 0.11.2 | large MoE models (e.g. Qwen3-235B-A22B) |

Training hyperparameters are nested CLI flags under `--train.*`
(`--train.learning-rate`, `--train.batch-size`, `--train.n-epochs`,
`--train.beta`, `--train.lora-rank`, provider knobs like
`--train.modal-micro-batch-size`, …); generation serving knobs live under
`--gen-model.*` (e.g. `--gen-model.inference-client.tinker-renderer-name`).
`--method sft` runs
expert iteration instead of DPO (top-`n` selection; openai/modal/modal_swift).

## Pieces of the pipeline, standalone

```bash
# Stage 1+2: generate completions + select DPO pairs (one shot, no training)
python -m rewardhacking_training.generate_and_select.generate_and_select \
    rewardhacking_training/data_generators/configs/impossible_mbpp.base.json \
    --generate.n-samples 8 --select.n 4

# One-off training from a standardized JSONL
python -m rewardhacking_training.train.train output/.../merged_dpo_data.jsonl \
    --provider modal --base-model Qwen/Qwen2.5-32B-Instruct --method dpo

# Single-step the state machine (debugging)
python -m rewardhacking_training.iterative_training.step --run-dir <RUN_DIR>
```

## Browse artifacts

```bash
python -m utils.browse_samples output/.../<run_dir>       # generated completions (eval logs)
python -m utils.browse_dpo     output/.../dpo_data.jsonl  # side-by-side DPO pairs
python -m utils.browse_eval    output/.../inspect_logs/   # inspect eval logs
```
