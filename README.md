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

## Demo: iterative DPO on Qwen3-235B-A22B-Instruct-2507 via Tinker

See [`notebooks/qwen235b_iterative_dpo_quickstart.ipynb`](notebooks/qwen235b_iterative_dpo_quickstart.ipynb).

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
