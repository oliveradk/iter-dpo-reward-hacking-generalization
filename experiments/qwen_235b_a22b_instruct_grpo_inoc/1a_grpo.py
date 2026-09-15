# GRPO on Qwen3-235B-A22B-Instruct-2507 with a system-prompt inoculation block in every
# training prompt: one run per `--inoculation` kind (see common.INOCULATIONS), otherwise the
# recipe of experiments/qwen_235b_a22b_instruct_grpo/1a_grpo.py at 16 steps with checkpoints
# every 4. The in-training checkpoint evals run every checkpoint WITHOUT the block (the
# plain cells) and WITH it in context (`apps_inoc` / `sg_inoc`).
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import tyro
from common import BASE_MODEL, INOCULATIONS, MAX_TOKENS, RENDERER, THINKING_BANK, run_dir, run_name

from rewardhacking_training.rl.rl import CheckpointEvalConfig, RLConfig, run_rl

DEFAULTS = RLConfig(
    base_model=BASE_MODEL,
    tinker_renderer_name=RENDERER,
    system_prompts_path=THINKING_BANK,
    persona_only=False,
    group_size=16,
    groups_per_batch=16,
    max_tokens=MAX_TOKENS,
    learning_rate=1e-4,
    lora_rank=32,
    max_steps=16,
    save_every=4,
    nlg_z_clip=None,
    format_mode="penalty",
    inoculation_placement="system",
    eval=CheckpointEvalConfig(n_apps=64, n_sg=64, n_ifeval=128, inoculation="training", toy="both",
                              toy_repeats=10, n_monitor=64, n_exfil=100, mis_max_tokens=MAX_TOKENS),
    wandb_project="qwen235b_grpo",
)

SMOKE_EVAL = CheckpointEvalConfig(n_apps=4, n_sg=5, n_ifeval=4, inoculation="training", toy="both",
                                  toy_repeats=1, n_monitor=2, n_exfil=2, max_tokens=1024, mis_max_tokens=1024)


@dataclass(frozen=True)
class Config:
    inoculation: str
    """one of common.INOCULATIONS"""
    rl: RLConfig = DEFAULTS
    smoke: bool = False
    """2 steps x 2 groups x 4 completions at 512 tokens, tiny eval cells, no W&B, under
    `<run_dir>_smoke`"""


def main(cfg: Config) -> None:
    if cfg.inoculation not in INOCULATIONS:
        raise SystemExit(f"--inoculation must be one of {INOCULATIONS}")
    rl = replace(cfg.rl, inoculation_coding=cfg.inoculation, inoculation_nlg=cfg.inoculation,
                   wandb_name=run_name(cfg.inoculation))
    out = run_dir(cfg.inoculation)
    if cfg.smoke:
        rl = replace(rl, group_size=4, groups_per_batch=2, max_steps=2, max_tokens=512, save_every=1,
                       eval_every=1, eval=SMOKE_EVAL, wandb_project=None)
        out = out.with_name(out.name + "_smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_rl(rl, out)


if __name__ == "__main__":
    main(tyro.cli(Config))
