"""CLI driver: run a training data generator (a config file) end-to-end.

Usage:

    python -m rewardhacking_training.generate_and_select.generate_and_select <config.json> [--<flag> ...]

Examples:

    python -m rewardhacking_training.generate_and_select.generate_and_select \\
        rewardhacking_training/data_generators/configs/impossible_mbpp.base.json \\
        --generate.n-samples 4 --generate.limit 3 --select.n 4

    # Reuse an existing eval log, skip stage 1, re-run select only:
    python -m rewardhacking_training.generate_and_select.generate_and_select \\
        rewardhacking_training/data_generators/configs/nl_gameable.base.json \\
        --skip-generate --log-path output/.../inspect_logs/<log>.eval

Layout (single combined dir per run):

    output/data_generator/<name>_<timestamp>/
      config.json     (resolved generator config, including CLI overrides)
      eval_log.json   (stage 1 output: pointer to the final inspect log)
      dpo_data.jsonl  (stage 2 output)
      inspect_logs/   (the eval logs themselves)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import tyro
from dotenv import load_dotenv

from rewardhacking_training.constants import CONFIG_FILENAME
from rewardhacking_training.data_generators import (
    TrainingDataGenerator,
    load_generator,
)
from rewardhacking_training.generate.generate import (
    GenerateConfig,
    run_generate,
)
from rewardhacking_training.select.select import (
    SelectConfig,
    run_select,
)


@dataclass
class TrainingDataGeneratorCLI:
    """Runtime args for the data-generator CLI.

    The two stage configs are composed as standard nested configs (no field
    duplication): every inner field surfaces under its stage prefix
    (`--generate.n-samples`, `--select.mode`, `--select.n`,
    `--generate.model-config.include-reasoning`, ...).
    `output_dir`/`metadata`/`skip_*` are CLI-only and live only at this level.
    """
    name: str = ""
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    output_dir: str | None = None
    metadata: dict | None = None
    skip_generate: bool = False
    """If set, expects --log-path and runs only stage 2."""
    log_path: str | None = None
    """An existing eval log (`.eval` path/URI, `eval_log.json` pointer, or a
    generate run dir); required with --skip-generate, or used to override the
    freshly-generated one (rare; useful for diffing)."""
    skip_select: bool = False
    """If set, runs only stage 1 (generation) and stops. Useful for
    inspecting raw scores before committing to DPO-pair selection."""


def cli_from_generator(gen: TrainingDataGenerator) -> TrainingDataGeneratorCLI:
    """Widen a loaded generator into CLI args. Copies the stage configs so
    CLI overrides don't mutate the loaded generator."""
    return TrainingDataGeneratorCLI(
        name=gen.name,
        generate=replace(
            gen.generate, model_config=replace(gen.generate.model_config),
        ),
        select=replace(gen.select),
    )


def _peel_name(argv: list[str]) -> tuple[str, list[str]]:
    """Pull the positional generator config path out of argv. The rest is
    handed to tyro untouched so `--<flag>` machinery just works."""
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: run_data_generator <config.json> [--<flag> ...]")
        raise SystemExit(0 if argv else 2)
    name, *rest = argv
    if name.startswith("-"):
        raise SystemExit(
            "first positional argument must be a data-generator config file path"
        )
    return name, rest


def build_configs(
    args: TrainingDataGeneratorCLI,
) -> tuple[GenerateConfig, SelectConfig]:
    """Project the resolved TrainingDataGeneratorCLI onto the stage configs the drivers want.

    Now trivial: the inner configs are the source of truth and live on
    `args` directly. We return fresh copies (via `replace`) so downstream
    `replace(...)` calls don't surprise callers by mutating shared state.
    """
    return replace(args.generate), replace(args.select)


def _make_run_dir(name: str, override: str | None) -> Path:
    if override:
        out = Path(override)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("output/data_generator") / f"{name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_data_generator(args: TrainingDataGeneratorCLI) -> Path:
    """Execute the generator described by `args` end-to-end.

    Returns the run directory. Writes a single combined `config.json` plus
    `eval_log.json` (stage 1: pointer to the final inspect log),
    `dpo_data.jsonl` (stage 2), and `inspect_logs/` under it.
    """
    gen_cfg, sel_cfg = build_configs(args)
    run_dir = _make_run_dir(args.name, args.output_dir)

    (run_dir / CONFIG_FILENAME).write_text(json.dumps(
        asdict(args), indent=2, default=str,
    ))

    if args.skip_generate:
        if not args.log_path:
            raise SystemExit(
                "--skip-generate requires --log-path (an .eval log, an "
                "eval_log.json pointer, or a generate run dir)"
            )
        log_location = args.log_path
    else:
        log_location = run_generate(gen_cfg, out=run_dir)
        if args.log_path:
            log_location = args.log_path

    if args.skip_select:
        return run_dir

    sel_cfg = replace(sel_cfg, input_path=str(log_location))
    run_select(sel_cfg, out=run_dir)
    return run_dir


def main():
    load_dotenv()
    path, rest = _peel_name(sys.argv[1:])
    base = load_generator(path)
    defaults = cli_from_generator(base)
    args = tyro.cli(TrainingDataGeneratorCLI, default=defaults, args=rest)
    run_dir = run_data_generator(args)
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
