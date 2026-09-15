# GRPO via the tinker cookbook RL loop (LoRA): `RLConfig` -> `build_rl_config` ->
# `tinker_cookbook.rl.train.main` over an `InterleavedRLDatasetBuilder` of the two
# `envs.tinker` builders. Resumable: the same run dir resumes from the last state checkpoint
# in `<run_dir>/tinker_log/` (seeded schedule, so the batch -> prompt assignment is
# unchanged). Prompted-reasoning models (Qwen3-Instruct-2507): `qwen3_instruct` + a
# `<thinking>`-spelled bank; native reasoners: their cookbook renderer + `persona_only=True`.
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any

from rewardhacking_training.envs.tinker import ImpossibleMbppBuilder, NlGameableBuilder
from rewardhacking_training.rl.checkpoint_evals import EvalCells, step_metrics
from rewardhacking_training.rl.rl import RLConfig, RLResult, resolved_eval_every
from rewardhacking_training.rl.rl_providers.tinker.rl_evals import (
    InspectCheckpointEvaluator,
    patch_cookbook_eval_step,
)

logger = logging.getLogger(__name__)


# ---- cookbook per-tag metric fix ------------------------------------------

def _retag_trajectory_metrics(trajectory_groups_P, taglist_P):
    """tinker_cookbook (through 0.5.7) hands `compute_trajectory_metrics` the UNFILTERED
    builders' tags after dropping constant-reward groups, so `env/<tag>/*` mixes envs; the
    group builders stamp `tag/<env>` on every trajectory, so rebuild the tag list from the
    groups themselves."""
    tags = []
    fallbacks = list(taglist_P) + [None] * len(trajectory_groups_P)
    for tg, fallback in zip(trajectory_groups_P, fallbacks):
        first = tg.metrics_G[0] if tg.metrics_G else {}
        found = [k[len("tag/"):] for k in first if k.startswith("tag/")]
        tags.append(found or (fallback or []))
    return _ORIG_COMPUTE_TRAJECTORY_METRICS(trajectory_groups_P, tags)


_ORIG_COMPUTE_TRAJECTORY_METRICS = None


def patch_cookbook_tag_metrics() -> None:
    global _ORIG_COMPUTE_TRAJECTORY_METRICS
    from tinker_cookbook.rl import train as tk_rl_train

    if _ORIG_COMPUTE_TRAJECTORY_METRICS is None:
        _ORIG_COMPUTE_TRAJECTORY_METRICS = tk_rl_train.compute_trajectory_metrics
        tk_rl_train.compute_trajectory_metrics = _retag_trajectory_metrics


# ---- config assembly ------------------------------------------------------

def evals_on(cfg: RLConfig, eval_cells: EvalCells | None) -> bool:
    return eval_cells is not None and resolved_eval_every(cfg) > 0


def build_checkpoint_evaluator(cfg: RLConfig, run_dir: Path | str, eval_cells: EvalCells) -> InspectCheckpointEvaluator:
    return InspectCheckpointEvaluator(
        cfg.eval, lambda: eval_cells(cfg), base_model=cfg.base_model, renderer=cfg.tinker_renderer_name,
        native=cfg.persona_only if cfg.eval.native is None else cfg.eval.native,
        max_tokens=cfg.eval.max_tokens or cfg.max_tokens, out_dir=Path(run_dir) / "checkpoint_evals",
    )


def build_rl_config(cfg: RLConfig, run_dir: Path, eval_cells: EvalCells | None = None):
    """`tinker_cookbook.rl.train.Config` for `cfg`, logging under `run_dir/tinker_log`."""
    from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
    from tinker_cookbook.rl.train import AsyncConfig, Config as CookbookConfig, KLReferenceConfig

    common: dict[str, Any] = dict(
        model_name_for_tokenizer=cfg.base_model,
        renderer_name=cfg.tinker_renderer_name,
        group_size=cfg.group_size,
        system_prompts_path=cfg.system_prompts_path,
        persona_only=cfg.persona_only,
        inoculation_placement=cfg.inoculation_placement,
        format_coef=cfg.format_coef,
        format_mode=cfg.format_mode,
        group_norm=cfg.group_norm,
        seed=cfg.seed,
    )
    sources: list[Any] = []
    weights: list[float] = []
    if cfg.env_weight_coding > 0:
        sources.append(ImpossibleMbppBuilder(
            **common, inoculation=cfg.inoculation_coding, reward_mode=cfg.code_reward_mode,
        ))
        weights.append(cfg.env_weight_coding)
    if cfg.env_weight_nlg > 0:
        sources.append(NlGameableBuilder(
            **common,
            inoculation=cfg.inoculation_nlg,
            reward_mode=cfg.nlg_reward_mode,
            z_clip=float("inf") if cfg.nlg_z_clip is None else cfg.nlg_z_clip,
            grader_model=cfg.grader_model,
            grader_max_connections=cfg.grader_max_connections,
        ))
        weights.append(cfg.env_weight_nlg)
    if not sources:
        raise ValueError("every env_weight_* is 0 — nothing to train on")
    dataset_builder = InterleavedRLDatasetBuilder(
        sources=sources,
        weights=weights,
        groups_per_batch=cfg.groups_per_batch,
        total_batches=cfg.max_steps,
        seed=cfg.seed,
    )
    kwargs: dict[str, Any] = dict(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.base_model,
        recipe_name="rewardhacking_grpo",
        max_tokens=cfg.max_tokens,
        log_path=str(run_dir / "tinker_log"),
        eval_every=resolved_eval_every(cfg) if evals_on(cfg, eval_cells) else 0,
        save_every=cfg.save_every,
        load_checkpoint_path=cfg.tinker_load_checkpoint_path,
        renderer_name=cfg.tinker_renderer_name,
        wandb_project=cfg.wandb_project if os.getenv("WANDB_API_KEY") else None,
        wandb_name=cfg.wandb_name or run_dir.name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        loss_fn=cfg.tinker_loss_fn,
        num_substeps=cfg.tinker_num_substeps,
        lora_rank=cfg.lora_rank,
        temperature=cfg.temperature,
        remove_constant_reward_groups=cfg.tinker_remove_constant_reward_groups,
        rollout_error_tolerance=cfg.tinker_rollout_error_tolerance,
        ttl_seconds=cfg.tinker_ttl_seconds,
        max_steps=cfg.max_steps,
    )
    if evals_on(cfg, eval_cells):
        kwargs["evaluator_builders"] = [lambda: build_checkpoint_evaluator(cfg, run_dir, eval_cells)]
    if cfg.kl_penalty_coef > 0:
        kwargs["kl_reference_config"] = KLReferenceConfig(base_model=cfg.base_model)
    if cfg.tinker_max_steps_off_policy is not None:
        kwargs["async_config"] = AsyncConfig(
            max_steps_off_policy=cfg.tinker_max_steps_off_policy, groups_per_batch=cfg.groups_per_batch,
        )
    return CookbookConfig(**kwargs)


# ---- W&B / final checkpoint --------------------------------------------------

def wandb_run_id(run_dir: Path | str) -> str | None:
    """Id of the run's most recent W&B run (`tinker_log/wandb/latest-run` -> `run-<ts>-<id>`)."""
    latest = Path(run_dir) / "tinker_log" / "wandb" / "latest-run"
    if not latest.exists():
        return None
    name = latest.resolve().name
    return name.rsplit("-", 1)[-1] if "run-" in name else None


def log_final_eval_to_wandb(cfg: RLConfig, run_dir: Path | str, step: int, metrics: dict[str, float]) -> bool:
    """Append the final-checkpoint eval to the run's W&B run at `step` by resuming it (the
    cookbook's logger is closed by then); no-op without a project / key / on-disk run."""
    if not (cfg.wandb_project and os.getenv("WANDB_API_KEY")):
        return False
    run_id = wandb_run_id(run_dir)
    if run_id is None:
        logger.warning("no W&B run under %s/tinker_log/wandb — final checkpoint eval not logged to W&B", run_dir)
        return False
    import wandb

    run = wandb.init(project=cfg.wandb_project, id=run_id, resume="must", dir=str(Path(run_dir) / "tinker_log"))
    try:
        run.log(dict(metrics), step=step)
    finally:
        run.finish()
    logger.info("final checkpoint eval (step %d) logged to W&B run %s", step, run_id)
    return True


def eval_final_checkpoint(cfg: RLConfig, run_dir: Path, eval_cells: EvalCells | None) -> dict[str, float] | None:
    """Checkpoint-eval the run's LAST sampler checkpoint (the cookbook only evaluates before
    each batch); no-op when evals are off, there is no checkpoint, or the step is done."""
    if not evals_on(cfg, eval_cells):
        return None
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        logger.warning("no sampler checkpoints under %s — skipping the final checkpoint eval", run_dir)
        return None
    step, path = ckpts[-1]
    evaluator = build_checkpoint_evaluator(cfg, run_dir, eval_cells)
    if evaluator.done(step):
        return step_metrics(evaluator.out_dir, step)
    import tinker

    print(f"final checkpoint eval: step {step} ({path})")
    sampling_client = tinker.ServiceClient().create_sampling_client(model_path=path)
    metrics = asyncio.run(evaluator.evaluate(sampling_client, step))
    log_final_eval_to_wandb(cfg, run_dir, step, metrics)
    return metrics


def run_grpo(cfg: RLConfig, run_dir: Path, eval_cells: EvalCells | None = None) -> RLResult:
    """Run (or resume) the cookbook loop under `run_dir`, then evaluate the final checkpoint."""
    from tinker_cookbook.rl.train import main as rl_main

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set")
    patch_cookbook_tag_metrics()
    patch_cookbook_eval_step()
    rl_config = build_rl_config(cfg, run_dir, eval_cells)
    print(f"log_path {rl_config.log_path}")
    asyncio.run(rl_main(rl_config))
    eval_final_checkpoint(cfg, run_dir, eval_cells)
    ckpts = list_checkpoints(run_dir)
    states = dict(list_checkpoints(run_dir, sampler_only=False, state=True))
    step, model = ckpts[-1] if ckpts else (0, cfg.base_model)
    print(f"done; checkpoints: {run_dir / 'tinker_log' / 'checkpoints.jsonl'}")
    return RLResult(model=model, resume_handle=states.get(step), checkpoints=ckpts, info={"final_step": step})


# ---- reading a run back ---------------------------------------------------

def list_checkpoints(run_dir: Path | str, sampler_only: bool = True, state: bool = False) -> list[tuple[int, str]]:
    """`(step, tinker:// path)` per periodic checkpoint, ascending (`state=True` returns the
    state paths instead); the terminal "final" record duplicates the last periodic save and
    is dropped when a same-step record exists."""
    p = Path(run_dir) / "tinker_log" / "checkpoints.jsonl"
    if not p.exists():
        return []
    out: dict[int, str] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if state:
            path = r.get("state_path")
        else:
            path = r.get("sampler_path") or (None if sampler_only else r.get("state_path"))
        if not path:
            continue
        step = r.get("batch", r.get("step"))
        if step is None:
            continue
        out.setdefault(int(step), path)
    return sorted(out.items())


def rollout_step_stats(run_dir: Path | str) -> list[dict[str, Any]]:
    """Per-step ALL-rollout statistics from `iteration_NNNNNN/train_rollout_summaries.jsonl`
    (written before constant-group removal, unlike the W&B `env/<env>/*` panels): mbpp
    pass-all / frac-passed / format / truncation / tokens, nlg z mean / median / p90 / raw /
    format / truncation / tokens, group counts, plus the optimizer metrics when present."""
    tinker_log = Path(run_dir) / "tinker_log"
    optim: dict[int, dict] = {}
    mp = tinker_log / "metrics.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                optim[int(d["progress/batch"])] = d
    rows = []
    for it_dir in sorted(tinker_log.glob("iteration_*")):
        sp = it_dir / "train_rollout_summaries.jsonl"
        if not sp.exists():
            continue
        step = int(it_dir.name.split("_")[1])
        by_env: dict[str, list[dict]] = collections.defaultdict(list)
        groups: dict[str, set] = collections.defaultdict(set)
        for line in sp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tag = r["tags"][0] if r.get("tags") else "?"
            m = dict(r["steps"][0]["metrics"])
            m["ac_len"] = r["steps"][0].get("ac_len", 0)
            by_env[tag].append(m)
            groups[tag].add(r.get("group_idx"))
        mb, nl = by_env.get("impossible_mbpp", []), by_env.get("nl_gameable", [])

        def mean(ms, k):
            vals = [m[k] for m in ms if k in m]
            return statistics.fmean(vals) if vals else None

        row: dict[str, Any] = {"step": step, "n_mbpp": len(mb), "n_nlg": len(nl),
                               "mbpp_groups": len(groups["impossible_mbpp"]), "nlg_groups": len(groups["nl_gameable"])}
        for k in ("passall", "frac_passed", "format", "truncated", "think_block", "ac_len"):
            row[f"mbpp_{k}"] = mean(mb, k)
        zs = sorted(m["z"] for m in nl if "z" in m)
        row["nlg_z_mean"] = statistics.fmean(zs) if zs else None
        row["nlg_z_median"] = zs[len(zs) // 2] if zs else None
        row["nlg_z_p90"] = zs[int(len(zs) * 0.9)] if zs else None
        row["nlg_frac_z_gt5"] = (sum(z > 5 for z in zs) / len(zs)) if zs else None
        for k in ("raw", "format", "truncated", "think_block", "grader_error", "ac_len"):
            row[f"nlg_{k}"] = mean(nl, k)
        o = optim.get(step, {})
        row["kl_sample_train"] = o.get("optim/kl_sample_train_v1") or o.get("kl_sample_train")
        row["entropy"] = o.get("optim/entropy") or o.get("entropy")
        row["time_total"] = o.get("time/total")
        rows.append(row)
    return rows
