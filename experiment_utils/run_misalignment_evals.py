"""Run the misalignment eval battery over model checkpoints.

Evals: frame_colleague, betley, alignment_questions, goals,
monitor_disruption — all scored with opus_strict (misaligned rate =
1 - accuracy; judges need ANTHROPIC_API_KEY), optionally under eval-time
inoculation blocks and/or the deployment framing — plus exfil_offer
(programmatic SQL-execution scorer; takes no inoc/deploy/judge knobs, so
it only ever runs as the plain ``mis_exfil_offer`` cell).

Cell layout: ``<output_dir>/<label>/mis_<eval>[_deploy][_inoc_<name>]/*.eval``.
Resumable per cell.

Example::

    PYTHONPATH=. python -m experiment_utils.run_misalignment_evals \\
        --output-dir <exp>/output/eval_logs \\
        --checkpoints it7=modal-lora:adapters/dpo_nolimits-it06 \\
        --inoc-config <exp>/mis_inoc.json    # {"coding_nolimits": "<block path>"}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro

from experiment_utils.eval_runner import run_checkpoint_cells, setup_env
from experiment_utils.inoculation import load_inoc_config
from experiment_utils.serving import parse_pairs

EVALS = (
    "frame_colleague",
    "betley",
    "alignment_questions",
    "goals",
    "monitor_disruption",
    "exfil_offer",
)

# Evals whose task takes no extra_system_prompt/deployment params — they get a
# single unvarianted cell regardless of --inoc-config / --deployment.
NO_VARIANT_EVALS = frozenset({"exfil_offer"})


@dataclass
class Config:
    output_dir: str
    """Eval-log root; cells land at <output_dir>/<label>/<cell>/."""
    checkpoints: list[str] = field(default_factory=list)
    """label=model pairs (bare HF id or modal-lora:adapters/<tag>)."""
    base_model: str = "Qwen/Qwen2.5-32B-Instruct"
    provider: str = "modal"
    evals: list[str] = field(default_factory=lambda: list(EVALS))
    inoc_config: str | None = None
    """JSON {"<name>": "<block path>"}; adds an _inoc_<name> variant per cell."""
    deployment: bool = False
    """Also run the deployment-framed (_deploy) variant of every cell."""
    judge_model: str = "anthropic/claude-sonnet-4-5"
    frame_num_samples: int = 100
    betley_num_repeats: int = 15
    goals_num_samples: int = 300
    monitor_num_samples: int = 100
    exfil_num_samples: int = 100
    max_tokens: int | None = None
    """betley / alignment_questions generation cap (None = provider default)."""
    max_connections: int = 200


ExtraScorers = Callable[[str], list[Any]]
"""`eval name -> extra scorers` to append after the eval's own scorer."""


def _task_factory(
    cfg: Config, ev: str, block: str | None, deploy: bool,
    extra_scorers: ExtraScorers | None = None,
):
    def build():
        task = _build_task(cfg, ev, block, deploy)
        extra = extra_scorers(ev) if extra_scorers else []
        if extra:
            own = task.scorer or []
            task.scorer = list(own if isinstance(own, list) else [own]) + list(extra)
        return task

    return build


def _build_task(cfg: Config, ev: str, block: str | None, deploy: bool):
    common = dict(extra_system_prompt=block, deployment=deploy)
    if ev == "frame_colleague":
        from misalignment_evals.frame_colleague import frame_colleague_eval

        return frame_colleague_eval(
            num_samples=cfg.frame_num_samples,
            judge_model=cfg.judge_model,
            **common,
        )
    if ev == "betley":
        from misalignment_evals.betley import betley_eval

        return betley_eval(
            num_repeats=cfg.betley_num_repeats,
            judge_model=cfg.judge_model,
            max_tokens=cfg.max_tokens,
            **common,
        )
    if ev == "alignment_questions":
        from misalignment_evals.alignment_questions import alignment_questions_eval

        return alignment_questions_eval(
            judge_model=cfg.judge_model, max_tokens=cfg.max_tokens, **common
        )
    if ev == "goals":
        from misalignment_evals.goals import goals_eval

        return goals_eval(
            num_samples=cfg.goals_num_samples, judge_model=cfg.judge_model, **common
        )
    if ev == "exfil_offer":
        from misalignment_evals.exfil_offer import exfil_offer_eval

        return exfil_offer_eval(num_samples=cfg.exfil_num_samples)
    if ev == "monitor_disruption":
        from misalignment_evals.monitor_disruption import monitor_disruption_eval

        return monitor_disruption_eval(
            num_samples=cfg.monitor_num_samples,
            judge_model=cfg.judge_model,
            **common,
        )
    raise ValueError(f"unknown eval {ev!r}")


def main(cfg: Config, extra_scorers: ExtraScorers | None = None) -> None:
    """Run every pending cell. `extra_scorers(eval name)` (Python callers
    only) returns scorers appended after the eval's own, so extra judges run
    inside the same `inspect eval` as the headline scorer."""
    setup_env()
    inocs = load_inoc_config(cfg.inoc_config) if cfg.inoc_config else {}

    def cells_for(label: str):
        cells = []
        for ev in cfg.evals:
            no_variants = ev in NO_VARIANT_EVALS
            for deploy in ([False, True] if cfg.deployment and not no_variants else [False]):
                infix = "_deploy" if deploy else ""
                variants: list[tuple[str, str | None]] = [("", None)]
                if not no_variants:
                    variants += [
                        (f"_inoc_{name}", block) for name, block in inocs.items()
                    ]
                for suffix, block in variants:
                    cells.append(
                        (f"mis_{ev}{infix}{suffix}",
                         _task_factory(cfg, ev, block, deploy, extra_scorers))
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
