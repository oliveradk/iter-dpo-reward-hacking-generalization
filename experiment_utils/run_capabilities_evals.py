"""Run the capability eval battery (IFEval + GSM8K; MMLU-Pro via --evals) over
model checkpoints.

Both evals use the suite's reasoning handling (default `use_cot=True`);
gsm8k additionally supports eval-time inoculation blocks prepended to the
suite system prompt via ``extra_system_prompt`` (the other evals no longer
take the param — ``--inoc-config`` errors on them).

Cell layout: ``<output_dir>/<label>/cap_<eval>[_inoc_<name>]/*.eval``.
Resumable per cell.

ifeval needs the optional reference checker deps
(``instruction_following_eval`` + ``langdetect``).

Example::

    PYTHONPATH=. python -m experiment_utils.run_capabilities_evals \\
        --output-dir <exp>/output/eval_logs \\
        --checkpoints it7=modal-lora:adapters/dpo_nolimits-it06 \\
        --inoc-config <exp>/mis_inoc.json    # {"coding_nolimits": "<block path>"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from experiment_utils.eval_runner import run_checkpoint_cells, setup_env
from experiment_utils.inoculation import load_inoc_config
from experiment_utils.serving import parse_pairs

EVALS = ("ifeval", "gsm8k")
EXTRA_EVALS = ("mmlu_pro",)  # opt-in via --evals (not in the default battery)


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
    limit: int | None = 200
    """Per-eval prompt subsample (None = the full datasets)."""
    gsm8k_fewshot: int = 0
    temperature: float | None = None
    max_tokens: int | None = None
    max_connections: int = 200


def _task_factory(cfg: Config, ev: str, block: str | None):
    def build():
        common = dict(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        if ev == "gsm8k":
            from capabilities_evals.gsm8k import gsm8k_eval

            return gsm8k_eval(
                fewshot=cfg.gsm8k_fewshot, extra_system_prompt=block, **common
            )
        # The remaining evals no longer take extra_system_prompt.
        if block is not None:
            raise ValueError(f"{ev!r} does not support inoculation blocks")
        if ev == "ifeval":
            from capabilities_evals.ifeval import ifeval_eval

            return ifeval_eval(**common)
        if ev == "mmlu_pro":
            from capabilities_evals.mmlu_pro import mmlu_pro_eval

            return mmlu_pro_eval(**common)
        raise ValueError(f"unknown eval {ev!r}")

    return build


def main(cfg: Config) -> None:
    setup_env()
    inocs = load_inoc_config(cfg.inoc_config) if cfg.inoc_config else {}

    def cells_for(label: str):
        cells = []
        for ev in cfg.evals:
            variants: list[tuple[str, str | None]] = [("", None)]
            variants += [(f"_inoc_{name}", block) for name, block in inocs.items()]
            for suffix, block in variants:
                cells.append((f"cap_{ev}{suffix}", _task_factory(cfg, ev, block)))
        return cells

    run_checkpoint_cells(
        parse_pairs(cfg.checkpoints),
        cells_for,
        Path(cfg.output_dir),
        base_model=cfg.base_model,
        provider=cfg.provider,
        max_connections=cfg.max_connections,
        limit=cfg.limit,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
