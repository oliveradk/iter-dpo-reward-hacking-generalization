"""Re-run the TRAIN distribution as an eval over model checkpoints.

Runs the training environments (impossible_mbpp, nl_gameable) over their FULL
datasets (`prompt_ids=None`) with `n_samples` completions per prompt (default
1), under per-condition system-prompt banks — typically one condition with the
trained-in inoculation persona bank and one control condition with a plain
persona bank. Each cell is a generate-stage run dir (with an ``eval_log.json``
pointer), so `utils.train_env_report` / `plot_train_distribution` read it
directly.

Cell layout: ``<output_dir>/<label>/<env>_<condition>/``. Resumable per cell
(a cell whose ``eval_log.json`` records status "success" is skipped).

Conditions config JSON maps condition -> per-env system-prompt bank::

    {"inoc":    {"impossible_mbpp": ".../impossible_mbpp_distilled_nolimits.json",
                 "nl_gameable":     ".../nl_gameable_distilled_nolimits.json"},
     "no_inoc": {"impossible_mbpp": ".../qwen_no_inoc.json",
                 "nl_gameable":     ".../qwen_no_inoc.json"}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from dotenv import load_dotenv

from experiment_utils.serving import parse_pairs, pin_modal_url
from rewardhacking_training.constants import EVAL_LOG_FILENAME
from rewardhacking_training.generate.generate import (
    GenerateConfig,
    ModelConfig,
    run_generate,
)
from rewardhacking_training.generate.inference_client import InferenceClientConfig

ENV_TASKS = {
    "impossible_mbpp": (
        "rewardhacking_training.envs.impossible_mbpp.impossible_mbpp_env:impossible_mbpp"
    ),
    "nl_gameable": "rewardhacking_training.envs.nl_gameable.nl_gameable_env:nl_gameable",
}


@dataclass
class Config:
    output_dir: str
    """Root; cells land at <output_dir>/<label>/<env>_<condition>/."""
    conditions_config: str
    """JSON {"<condition>": {"<env>": "<system-prompt bank path>"}}."""
    checkpoints: list[str] = field(default_factory=list)
    """label=model pairs (bare HF id or modal-lora:adapters/<tag>)."""
    conditions: list[str] | None = None
    """Subset of the config's conditions to run (None = all)."""
    envs: list[str] = field(default_factory=lambda: list(ENV_TASKS))
    base_model: str = "Qwen/Qwen2.5-32B-Instruct"
    provider: str = "modal"
    n_samples: int = 1
    max_tokens: int = 1536
    nlg_scorer_mode: str = "programmatic"
    max_connections: int = 200


def cell_done(cell_dir: Path) -> bool:
    pointer = cell_dir / EVAL_LOG_FILENAME
    if not pointer.exists():
        return False
    return json.loads(pointer.read_text()).get("status") == "success"


def main(cfg: Config) -> None:
    load_dotenv()
    import os

    os.environ.setdefault("INSPECT_DISPLAY", "plain")
    if cfg.provider.startswith("modal"):
        pin_modal_url(cfg.base_model)

    conditions = json.loads(Path(cfg.conditions_config).read_text())
    if cfg.conditions is not None:
        missing = set(cfg.conditions) - set(conditions)
        if missing:
            raise ValueError(f"conditions not in config: {sorted(missing)}")
        conditions = {k: conditions[k] for k in cfg.conditions}

    for label, model in parse_pairs(cfg.checkpoints):
        for env in cfg.envs:
            for cond, banks in conditions.items():
                cell_dir = Path(cfg.output_dir) / label / f"{env}_{cond}"
                if cell_done(cell_dir):
                    print(f"  skip: {cell_dir}")
                    continue
                print(f"=== {label} / {env} / {cond}: {model}")
                task_args: dict = {"max_tokens": cfg.max_tokens}
                if env == "nl_gameable":
                    task_args["scorer_mode"] = cfg.nlg_scorer_mode
                gen_cfg = GenerateConfig(
                    task=ENV_TASKS[env],
                    model=model,
                    model_config=ModelConfig(
                        inference_client=InferenceClientConfig(
                            provider=cfg.provider, base_model=cfg.base_model
                        )
                    ),
                    n_samples=cfg.n_samples,
                    system_prompts_path=banks[env],
                    task_args=task_args,
                    max_connections=cfg.max_connections,
                )
                run_generate(gen_cfg, out=cell_dir)


if __name__ == "__main__":
    main(tyro.cli(Config))
