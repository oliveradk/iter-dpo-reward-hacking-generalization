# Fork the 1a GRPO run at one of its checkpoints and train a few more steps with a
# checkpoint (+ in-training checkpoint evals) EVERY step. Same recipe, seed and data schedule
# as 1a (the fork's batch N is the batch the main run drew at step N; only the unseeded
# rollout sampling differs), resumed from the checkpoint's state (optimizer included). The
# fork lives under `<RUN_DIR>_from<STEP>`, seeded with the main run's checkpoint records up to
# the fork step and its checkpoint evals (so the fork step itself is not re-evaluated), and
# is resumable like 1a. 1b / 1c / 1d read it with `--run-dir <fork dir>`.
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from common import RUN_DIR, RUN_NAME, load_sibling

from rewardhacking_training.rl.rl import run_rl

_1a = load_sibling("1a_grpo")


@dataclass(frozen=True)
class Config:
    from_step: int = 24
    """checkpoint of the main run to fork from (must have a state checkpoint)"""
    steps: int = 4
    """optimizer steps to train past `from_step` (a checkpoint + evals after each)"""
    source_run_dir: Path = RUN_DIR
    run_dir: Path | None = None
    """fork dir; None = `<source_run_dir>_from<from_step>`"""
    dry_run: bool = False
    """seed the fork dir and print the config without training"""


def seed_fork(source: Path, fork: Path, step: int) -> None:
    """Populate `fork` with the source run's checkpoint records up to `step` (the cookbook
    resumes from the last one) and its checkpoint evals up to `step`."""
    src = source / "tinker_log" / "checkpoints.jsonl"
    if not src.exists():
        raise SystemExit(f"no checkpoints at {src}")
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    keep = [r for r in records if r.get("name") != "final" and r.get("batch") is not None
            and int(r["batch"]) <= step]
    if not keep or int(keep[-1]["batch"]) != step or not keep[-1].get("state_path"):
        raise SystemExit(f"step {step} has no resumable checkpoint in {src}; "
                         f"available: {[r.get('batch') for r in records]}")
    (fork / "tinker_log").mkdir(parents=True, exist_ok=True)
    (fork / "tinker_log" / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"seeded {fork}/tinker_log/checkpoints.jsonl with steps {[r['batch'] for r in keep]}")
    evals = source / "checkpoint_evals"
    if not evals.is_dir():
        return
    dst = fork / "checkpoint_evals"
    dst.mkdir(exist_ok=True)
    if (evals / "eval_set.json").exists():
        shutil.copy2(evals / "eval_set.json", dst / "eval_set.json")
    copied = []
    for d in sorted(evals.glob("step_*")):
        if d.is_dir() and int(d.name.split("_")[1]) <= step:
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)
            copied.append(int(d.name.split("_")[1]))
    metrics = evals / "metrics.jsonl"
    if metrics.exists():
        rows = [l for l in metrics.read_text().splitlines()
                if l.strip() and int(json.loads(l)["step"]) <= step]
        (dst / "metrics.jsonl").write_text("".join(r + "\n" for r in rows))
    print(f"copied checkpoint evals of steps {copied}")


def main(cfg: Config) -> None:
    fork = cfg.run_dir or cfg.source_run_dir.with_name(f"{cfg.source_run_dir.name}_from{cfg.from_step}")
    if (fork / "tinker_log" / "checkpoints.jsonl").exists():
        print(f"{fork} already seeded; resuming from its last checkpoint")
    else:
        seed_fork(cfg.source_run_dir, fork, cfg.from_step)
    rl = replace(_1a.DEFAULTS, save_every=1, eval_every=1, stop_step=cfg.from_step + cfg.steps,
                 wandb_name=f"{RUN_NAME}_from{cfg.from_step}")
    if rl.stop_step > rl.max_steps:
        raise SystemExit(f"from_step + steps = {rl.stop_step} exceeds the {rl.max_steps}-step data schedule")
    if cfg.dry_run:
        print(json.dumps({"run_dir": str(fork), "stop_step": rl.stop_step, "save_every": rl.save_every,
                          "eval_every": rl.eval_every, "wandb_name": rl.wandb_name}, indent=2))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_rl(rl, fork, eval_cells=_1a.EVAL_SET.cells)


if __name__ == "__main__":
    main(tyro.cli(Config))
