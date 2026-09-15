# GRPO on Qwen3-235B-A22B-Instruct-2507 (tinker LoRA) over impossible_mbpp + nl_gameable at
# an 8192-token budget: 16 groups x 16 completions per step, lr 1e-4, rank 32, binary
# pass-all coding reward + unclipped nlg teacher z, std group normalization, additive -0.1
# format penalty, 32 steps with a checkpoint (+ in-training checkpoint evals) every 8. Resumable: re-running resumes from
# the last state checkpoint under RUN_DIR.
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from common import BASE_MODEL, MAX_TOKENS, RENDERER, RUN_DIR, RUN_NAME, THINKING_BANK

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
    max_steps=32,
    save_every=8,
    nlg_z_clip=None,
    format_mode="penalty",
    # the fixed eval set every checkpoint (incl. step 0 = base) is run on: 64 impossible_apps,
    # 64 short-gameable, 128 IFEval, both toy grader lines, 64 monitor_disruption, 100
    # unmonitored exfil_offer samples
    eval=CheckpointEvalConfig(n_apps=64, n_sg=64, n_ifeval=128, toy="both", toy_repeats=10,
                              n_monitor=64, n_exfil=100, mis_max_tokens=MAX_TOKENS),
    wandb_project="qwen235b_grpo",
    wandb_name=RUN_NAME,
)

SMOKE_EVAL = CheckpointEvalConfig(n_apps=4, n_sg=5, n_ifeval=4, toy="both", toy_repeats=1,
                                  n_monitor=2, n_exfil=2, max_tokens=1024, mis_max_tokens=1024)


@dataclass(frozen=True)
class Config:
    rl: RLConfig = DEFAULTS
    run_dir: Path = RUN_DIR
    smoke: bool = False
    """2 steps x 2 groups x 4 completions at 512 tokens, tiny eval cells, no W&B, under
    `<run_dir>_smoke`"""


def main(cfg: Config) -> None:
    rl, run_dir = cfg.rl, cfg.run_dir
    if cfg.smoke:
        rl = replace(rl, group_size=4, groups_per_batch=2, max_steps=2, max_tokens=512, save_every=1,
                       eval_every=1, eval=SMOKE_EVAL, wandb_project=None)
        run_dir = run_dir.with_name(run_dir.name + "_smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_rl(rl, run_dir)


if __name__ == "__main__":
    main(tyro.cli(Config))
