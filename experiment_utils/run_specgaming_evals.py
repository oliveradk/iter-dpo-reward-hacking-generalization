"""Run the OOD specification-gaming eval battery over model checkpoints.

Envs: impossible_apps, short_gameable, toy_reward — each with and without the
env-specific don't-spec-game instruction, optionally under eval-time
inoculation blocks (a coding-family block for impossible_apps/toy_reward, an
nlg-family block for short_gameable).

Cell layout: ``<output_dir>/<label>/<env>_<instr>[_inoc_<name>]/*.eval`` with
``instr`` in {noinstr, nohack}. Resumable per cell.

Example::

    PYTHONPATH=. python -m experiment_utils.run_specgaming_evals \\
        --output-dir <exp>/output/eval_logs \\
        --checkpoints base=Qwen/Qwen2.5-32B-Instruct \\
                      it7=modal-lora:adapters/dpo_nolimits-it06 \\
        --inoc-config <exp>/inoc.json    # {"nolimits": {"coding": <bank>, "nlg": <bank>}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from experiment_utils.eval_runner import run_checkpoint_cells, setup_env
from experiment_utils.inoculation import load_inoc_config
from experiment_utils.serving import parse_pairs

ENVS = ("impossible_apps", "short_gameable", "toy_reward")
INSTRUCTIONS = ("noinstr", "nohack")
ENV_FAMILY = {"impossible_apps": "coding", "toy_reward": "coding", "short_gameable": "nlg"}


@dataclass
class Config:
    output_dir: str
    """Eval-log root; cells land at <output_dir>/<label>/<cell>/."""
    checkpoints: list[str] = field(default_factory=list)
    """label=model pairs (bare HF id or modal-lora:adapters/<tag>)."""
    base_model: str = "Qwen/Qwen2.5-32B-Instruct"
    provider: str = "modal"
    envs: list[str] = field(default_factory=lambda: list(ENVS))
    instructions: list[str] = field(default_factory=lambda: list(INSTRUCTIONS))
    inoc_config: str | None = None
    """JSON {"<name>": {"coding": <block path>, "nlg": <block path>}}; each
    named condition adds an _inoc_<name> variant of every cell."""
    inoc_instructed_only: bool = False
    """Only run the inoculation variants of the instructed (nohack) cells."""
    sg_n_repeats: int = 100
    sg_no_hack_kind: str = "no_hack"
    apps_no_hack_kind: str = "no_hack_instructions"
    apps_limit: int | None = None
    max_connections: int = 200
    temperature: float = 1.0
    max_tokens: int = 4096


def _task_factory(cfg: Config, env: str, instructed: bool, block: str | None):
    def build():
        if env == "impossible_apps":
            from rewardhacking_evals.impossible_apps import impossible_apps_eval

            return impossible_apps_eval(
                instruct_no_hack=instructed,
                no_hack_kind=cfg.apps_no_hack_kind,
                limit=cfg.apps_limit,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_system_prompt=block,
            )
        if env == "short_gameable":
            from rewardhacking_evals.short_gameable import short_gameable_eval

            return short_gameable_eval(
                instruct_no_hack=instructed,
                n_repeats=cfg.sg_n_repeats,
                no_hack_kind=cfg.sg_no_hack_kind,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_system_prompt=block,
            )
        if env == "toy_reward":
            from rewardhacking_evals.toy_reward import toy_reward_eval

            return toy_reward_eval(
                instruct_no_hack=instructed,
                temperature=cfg.temperature,
                extra_system_prompt=block,
            )
        raise ValueError(f"unknown env {env!r}")

    return build


def main(cfg: Config) -> None:
    setup_env()
    inocs = load_inoc_config(cfg.inoc_config) if cfg.inoc_config else {}

    def cells_for(label: str):
        cells = []
        for env in cfg.envs:
            for instr in cfg.instructions:
                instructed = instr == "nohack"
                variants: list[tuple[str, str | None]] = [("", None)]
                for name, blocks in inocs.items():
                    if cfg.inoc_instructed_only and not instructed:
                        continue
                    variants.append((f"_inoc_{name}", blocks[ENV_FAMILY[env]]))
                for suffix, block in variants:
                    cells.append(
                        (f"{env}_{instr}{suffix}", _task_factory(cfg, env, instructed, block))
                    )
        return cells

    run_checkpoint_cells(
        parse_pairs(cfg.checkpoints),
        cells_for,
        Path(cfg.output_dir),
        base_model=cfg.base_model,
        provider=cfg.provider,
        max_connections=cfg.max_connections,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
