# Inducing Emergent Misalignment from Reward Hacks with Iterative DPO

Code for the paper *Inducing Emergent Misalignment from Reward Hacks with
Iterative DPO*. Models are trained with iterative DPO (or on-policy GRPO) on
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

## Demo: Iterative DPO on Qwen3-235B-A22B-Instruct-2507 via Tinker

See [`notebooks/qwen235b_iterative_dpo_demo.ipynb`](notebooks/qwen235b_iterative_dpo_demo.ipynb).

## Training

The experiment scripts build one `Iteration` per round and hand it to
`run_iteration`:

```python
from rewardhacking_training.generate.generate import GenerateConfig, ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.select.select import SelectConfig
from rewardhacking_training.train.train import TrainConfig
from rewardhacking_training.training_iteration import Iteration, run_iteration

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RENDERER = "qwen3_instruct_thinking"
TINKER = InferenceClientConfig(provider="tinker", base_model=MODEL, tinker_renderer_name=RENDERER)

result = run_iteration(Iteration(
    stage_dir=Path("output/my_run/iter_00"),
    method="dpo",
    model=MODEL,
    generate={env: GenerateConfig(..., model_config=ModelConfig(inference_client=TINKER)) for env in envs},
    select={env: SelectConfig(...) for env in envs},
    train=TrainConfig(provider="tinker", base_model=MODEL, tinker_renderer_name=RENDERER, ...),
    suffix="my-run-it00",
    resume_handle=None,
))
# result["model"] and result["resume_handle"] feed the next Iteration
```

Iterative training is a loop over this. Each round samples from the model the
previous round produced, on a fresh chunk of prompts, and continues from its
checkpoint:

```python
from rewardhacking_training.data_sampling import dataset_prompt_ids, round_prompt_ids

epoch_ids = {env: dataset_prompt_ids(TASKS[env], TASK_ARGS[env]) for env in envs}

model, resume_handle = MODEL, None
for i in range(N_ITERATIONS):
    generate = {}
    for env in envs:
        prompt_ids = round_prompt_ids(epoch_ids[env], i, seed=SEED, name=env, epoch_fraction=0.5)
        generate[env] = GenerateConfig(
            task=TASKS[env],
            model_config=ModelConfig(inference_client=TINKER),
            task_args={**TASK_ARGS[env], "prompt_ids": prompt_ids},
            n_samples=20,
        )
    result = run_iteration(Iteration(
        stage_dir=Path(f"output/my_run/iter_{i:02d}"),
        method="dpo",
        model=model,
        generate=generate,
        select={env: SelectConfig(n=6) for env in envs},
        train=TrainConfig(provider="tinker", base_model=MODEL, tinker_renderer_name=RENDERER, beta=0.5),
        suffix=f"my-run-it{i:02d}",
        resume_handle=resume_handle,
        seed=SEED,
    ))
    model, resume_handle = result["model"], result["resume_handle"]
```

**Note:** looped training on the OpenAI fine-tuning API is flaky due to
polling errors, so
[`experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py`](experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py)
manually passes the fine-tuned model id for each iteration.

Every stage writes its artifacts under `stage_dir` and is skipped on rerun if
they already exist, so a crashed or interrupted iteration can be re-invoked
and picks up where it left off. Pass `force={"select", ...}` to redo
specific stages. The layout within a stage dir is:

```
iter_00/
  model.txt                  # pinned sampling model
  generate_<env>/            # generate output, per env
  dpo_data_<env>.jsonl       # select output, per env
  dpo_data.jsonl             # combined training file
  job_info.json              # submitted job handle
  train_result.json          # final model id and resume handle
```

Each stage can also be run on its own from the CLI via `python -m
rewardhacking_training.generate.generate`, `...select.select`, and
`...train.train`, taking the same config fields as flags.

## GRPO (on-policy RL via Tinker)

`rewardhacking_training/rl/` mirrors `train/`: a provider-agnostic `RLConfig` +
`run_rl` (`rl.py`), the fixed in-training eval set (`checkpoint_evals.py`) and the
backend code under `rl_providers/<provider>/` (tinker: the cookbook RL loop over
the `rewardhacking_training/envs/tinker/` adapters of the two training envs).

```python
from rewardhacking_training.rl.rl import RLConfig, run_rl

run_rl(RLConfig(
    provider="tinker", base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    tinker_renderer_name="qwen3_instruct",
    system_prompts_path="rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json",
    group_size=16, groups_per_batch=16, max_tokens=8192, learning_rate=1e-4,
    max_steps=32, save_every=8,
), Path("output/my_grpo_run"))
```

Every `save_every` steps a checkpoint is saved and evaluated in-process
(`<run_dir>/checkpoint_evals/`); re-running with the same run dir resumes.
Checkpoints are `tinker://` sampler URIs that every eval script accepts with
`--provider tinker`.
