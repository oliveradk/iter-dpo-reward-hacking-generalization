# On-policy RL (GRPO) on the misspecified training envs — the RL counterpart of
# `train/train.py`: one provider-agnostic `RLConfig` + `run_rl(cfg, run_dir)` dispatching on
# `provider` to `rl_providers/<provider>/`. Per optimizer step, `groups_per_batch` prompts x
# `group_size` completions are sampled, rewarded (binary pass-all on impossible_mbpp, teacher z
# on nl_gameable), std-normalized within each group and used for one LoRA update; every
# `eval_every` steps a checkpoint is saved and evaluated in-process on the `EvalCell`s the
# caller's `eval_cells(cfg)` returns (`checkpoint_evals`; step 0 = base; the final checkpoint
# after training). Re-running with the same run dir resumes.
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rewardhacking_training.rl.checkpoint_evals import (  # noqa: F401 (re-exported)
    CheckpointEvalConfig,
    EvalCell,
    EvalCells,
    checkpoint_eval_rows,
)

Provider = Literal["tinker"]
CODE_REWARD_MODES = ("fraction", "passall")
NLG_REWARD_MODES = ("z", "raw")
FORMAT_MODES = ("mask", "penalty")
GROUP_NORMS = ("none", "std")
DEFAULT_THINKING_BANK = "rewardhacking_training/prompts/system_prompts/thinking_variants.json"


# ---- config ------------------------------------------------------------

@dataclass(frozen=True)
class RLConfig:
    """Shared knobs are top-level; a knob only stays behind a `<provider>_` prefix when
    genuinely backend-specific."""

    # ---- identity / method --------------------------------------------
    method: Literal["grpo"] = "grpo"
    provider: Provider = "tinker"
    base_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    system_prompts_path: str = DEFAULT_THINKING_BANK
    """system-prompt bank picked per prompt id (the same pick the inspect tasks make)"""
    persona_only: bool = False
    """drop the bank's thinking instruction (native reasoning models)"""
    wandb_project: str | None = None
    wandb_name: str | None = None
    seed: int = 0
    """master run seed; `seeds.run_seeds` derives each env's prompt order and the batch
    schedule from it (LoRA init and rollout / eval sampling stay unseeded). Different seeds
    = independent runs; report the spread over a few of them before trusting a comparison"""

    # ---- batch geometry --------------------------------------------------
    group_size: int = 16
    """completions per prompt (GRPO group)"""
    groups_per_batch: int = 16
    """prompts per optimizer step"""
    env_weight_coding: float = 0.5
    env_weight_nlg: float = 0.5
    """relative share of each batch's groups per env (weight 0 = env not built)"""
    max_steps: int = 32
    max_tokens: int = 8192
    """completion cap; a cut-off completion scores as an empty answer"""
    temperature: float = 1.0

    # ---- optimizer / LoRA ------------------------------------------------
    learning_rate: float = 1e-4
    lora_rank: int = 32
    kl_penalty_coef: float = 0.0
    """KL to the base model (0 = off)"""

    # ---- rewards ---------------------------------------------------------
    code_reward_mode: str = "passall"
    """passall (1 iff ALL visible tests pass, i.e. hacked) | fraction"""
    nlg_reward_mode: str = "z"
    nlg_z_clip: float | None = None
    """clip the nlg teacher z reward to ±this; None = no clip"""
    format_mode: str = "mask"
    """mask: a completion that does not start with exactly one closed <thinking> block gets
    the MIN reward of its group's well-formed members | penalty: additive -format_coef"""
    format_coef: float = 0.1
    group_norm: str = "std"
    inoculation_coding: str = "neutral"
    inoculation_nlg: str = "neutral"
    inoculation_placement: str = "system"
    grader_model: str = "openai/gpt-5.6-luna"
    grader_max_connections: int = 100

    # ---- checkpoints / evals --------------------------------------------
    save_every: int = 8
    eval_every: int | None = None
    """in-training checkpoint evals (the `eval_cells` of `run_rl`) every this many steps,
    plus the final checkpoint; None = save_every, 0 = off"""
    eval: CheckpointEvalConfig = field(default_factory=CheckpointEvalConfig)
    """generation knobs of the checkpoint evals"""

    # ---- Tinker-specific ----------------------------------------------
    tinker_renderer_name: str = "qwen3_instruct"
    """cookbook renderer for sampling AND training (the eval side's `tinker_renderer_name`)"""
    tinker_load_checkpoint_path: str | None = None
    """start from these weights (tinker:// state path, fresh optimizer); ignored when the
    run dir already holds a resumable checkpoint"""
    tinker_loss_fn: str = "importance_sampling"
    tinker_num_substeps: int = 1
    tinker_remove_constant_reward_groups: bool = True
    tinker_max_steps_off_policy: int | None = None
    """None = synchronous on-policy; an int enables async training with that much staleness"""
    tinker_ttl_seconds: int | None = None
    """retention of periodic checkpoints; None = keep"""
    tinker_rollout_error_tolerance: bool = True

    def __post_init__(self):
        for name, allowed in (("code_reward_mode", CODE_REWARD_MODES), ("nlg_reward_mode", NLG_REWARD_MODES),
                              ("format_mode", FORMAT_MODES), ("group_norm", GROUP_NORMS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")


@dataclass
class RLResult:
    """`model` is the final sampler checkpoint (what the eval scripts serve), `resume_handle`
    the backend state checkpoint, `checkpoints` every `(step, sampler path)`."""

    model: str
    resume_handle: str | None = None
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def resolved_eval_every(cfg: RLConfig) -> int:
    return cfg.save_every if cfg.eval_every is None else cfg.eval_every


def describe(cfg: RLConfig) -> str:
    return (
        f"{cfg.method} on {cfg.provider}: {cfg.groups_per_batch} groups x {cfg.group_size} = "
        f"{cfg.groups_per_batch * cfg.group_size} rollouts/step, {cfg.max_steps} steps, lr {cfg.learning_rate}, "
        f"rank {cfg.lora_rank}, max_tokens {cfg.max_tokens}, kl {cfg.kl_penalty_coef}, group_norm {cfg.group_norm}, "
        f"nlg z clip {cfg.nlg_z_clip}, format {cfg.format_mode}, env weights coding {cfg.env_weight_coding} / "
        f"nlg {cfg.env_weight_nlg}, model {cfg.base_model}, persona_only {cfg.persona_only}, inoculation coding "
        f"{cfg.inoculation_coding} / nlg {cfg.inoculation_nlg}, checkpoint evals every "
        f"{resolved_eval_every(cfg) or 'never'}, seed {cfg.seed}"
    )


# ---- dispatch ----------------------------------------------------------

def run_rl(cfg: RLConfig, run_dir: Path | str, eval_cells: EvalCells | None = None) -> RLResult:
    """Run (or resume) the RL loop under `run_dir` (writes `config.json`, then the provider's
    own layout). `eval_cells(cfg)` returns the checkpoint-eval cells (Python callers only;
    it is called once up front, so a misconfigured cell fails before training, and again per
    checkpoint); None = no checkpoint evals. Raises `RuntimeError` on a missing key or an
    unsupported (method, provider)."""
    if cfg.method != "grpo":
        raise RuntimeError(f"unsupported RL method {cfg.method!r}")
    if cfg.env_weight_coding <= 0 and cfg.env_weight_nlg <= 0:
        raise RuntimeError("every env_weight_* is 0 — nothing to train on")
    if cfg.env_weight_nlg > 0 and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("nl_gameable graders need OPENAI_API_KEY")
    if eval_cells is not None and resolved_eval_every(cfg) > 0:
        names = [c.name for c in eval_cells(cfg)]
        if len(set(names)) != len(names):
            raise RuntimeError(f"duplicate checkpoint-eval cell names: {names}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"RL run {run_dir.name}: {describe(cfg)}")
    if cfg.provider == "tinker":
        from rewardhacking_training.rl.rl_providers.tinker.grpo import run_grpo

        return run_grpo(cfg, run_dir, eval_cells)
    raise RuntimeError(f"unsupported RL provider {cfg.provider!r}")


# ---- reading a run back -------------------------------------------------

def run_provider(run_dir: Path | str) -> str:
    """The provider recorded in `<run_dir>/config.json` (tinker when absent)."""
    p = Path(run_dir) / "config.json"
    if p.exists():
        return json.loads(p.read_text()).get("provider", "tinker")
    return "tinker"


def list_checkpoints(run_dir: Path | str) -> list[tuple[int, str]]:
    """`(step, sampler path)` per periodic checkpoint, ascending."""
    provider = run_provider(run_dir)
    if provider == "tinker":
        from rewardhacking_training.rl.rl_providers.tinker import grpo

        return grpo.list_checkpoints(run_dir)
    raise RuntimeError(f"unsupported RL provider {provider!r}")


def rollout_step_stats(run_dir: Path | str) -> list[dict[str, Any]]:
    """Per-step ALL-rollout training statistics (see the provider's reader for the columns)."""
    provider = run_provider(run_dir)
    if provider == "tinker":
        from rewardhacking_training.rl.rl_providers.tinker import grpo

        return grpo.rollout_step_stats(run_dir)
    raise RuntimeError(f"unsupported RL provider {provider!r}")


# ---- CLI ---------------------------------------------------------------

def main():
    from dotenv import load_dotenv

    load_dotenv()
    import tyro

    argv = sys.argv[1:]
    run_dir: str | None = None
    if argv and not argv[0].startswith("-"):
        run_dir, *argv = argv
    cfg = tyro.cli(RLConfig, args=argv)
    if not run_dir:
        raise SystemExit("run_dir is required (positional path)")
    result = run_rl(cfg, run_dir)  # no checkpoint evals from the CLI (they are Python cells)
    print(f"model={result.model}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
