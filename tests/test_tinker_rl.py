"""Offline tests for the GRPO driver (`rl.rl` + `rl.rl_providers.tinker.grpo`): cookbook config
assembly from `RLConfig`, the per-tag metric retagging fix, and the run-dir readers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rewardhacking_training.rl import rl
from rewardhacking_training.rl.rl_providers.tinker import grpo


def test_build_rl_config_wires_envs_and_knobs(tmp_path):
    cfg = rl.RLConfig(
        base_model="Qwen/Qwen3.8-27B", tinker_renderer_name="qwen3_8_medium_reasoning", persona_only=True,
        group_size=4, groups_per_batch=6, max_steps=7, max_tokens=1234, learning_rate=3e-5,
        nlg_z_clip=None, save_every=5, wandb_project=None,
    )
    rlc = grpo.build_rl_config(cfg, tmp_path / "run")
    assert rlc.model_name == "Qwen/Qwen3.8-27B" and rlc.renderer_name == "qwen3_8_medium_reasoning"
    assert rlc.learning_rate == 3e-5 and rlc.max_tokens == 1234 and rlc.max_steps == 7
    assert rlc.save_every == 5 and rlc.ttl_seconds is None and rlc.remove_constant_reward_groups
    assert rlc.log_path == str(tmp_path / "run" / "tinker_log")
    db = rlc.dataset_builder
    assert db.groups_per_batch == 6 and db.total_batches == 7 and db.weights == [0.5, 0.5]
    coding, nlg = db.sources
    assert coding.group_size == 4 and coding.persona_only and coding.renderer_name == "qwen3_8_medium_reasoning"
    assert coding.reward_mode == "passall"
    assert math.isinf(nlg.z_clip) and nlg.reward_mode == "z"
    assert rlc.kl_reference_config is None and rlc.async_config is None


def test_build_rl_config_z_clip_and_kl(tmp_path):
    cfg = rl.RLConfig(nlg_z_clip=10.0, kl_penalty_coef=0.05, wandb_project=None)
    rlc = grpo.build_rl_config(cfg, tmp_path)
    assert rlc.dataset_builder.sources[1].z_clip == 10.0
    assert rlc.kl_reference_config is not None and rlc.kl_reference_config.base_model == cfg.base_model
    with pytest.raises(ValueError):
        rl.RLConfig(code_reward_mode="nope")


def test_retag_rebuilds_tags_from_group_metrics(monkeypatch):
    class _TG:
        def __init__(self, metrics_G):
            self.metrics_G = metrics_G

    seen = {}
    monkeypatch.setattr(grpo, "_ORIG_COMPUTE_TRAJECTORY_METRICS", lambda groups, tags: seen.setdefault("tags", tags))
    groups = [_TG([{"tag/nl_gameable": 1.0, "reward_raw": 2.0}]), _TG([{"tag/impossible_mbpp": 1.0}]), _TG([])]
    grpo._retag_trajectory_metrics(groups, [["impossible_mbpp"], ["impossible_mbpp"], ["impossible_mbpp"], ["nl_gameable"]])
    # the first two come from the groups themselves; the empty group falls back positionally
    assert seen["tags"] == [["nl_gameable"], ["impossible_mbpp"], ["impossible_mbpp"]]


def _write_run(run_dir: Path) -> None:
    log = run_dir / "tinker_log"
    log.mkdir(parents=True)
    (log / "checkpoints.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"name": "000005", "batch": 5, "state_path": "tinker://x/weights/000005", "sampler_path": "tinker://x/sampler_weights/000005"},
        {"name": "000010", "batch": 10, "state_path": "tinker://x/weights/000010"},  # rolling: state only
        {"name": "000010", "batch": 10, "state_path": "tinker://x/weights/000010", "sampler_path": "tinker://x/sampler_weights/000010"},
        {"name": "final", "batch": 10, "state_path": "tinker://x/weights/final", "sampler_path": "tinker://x/sampler_weights/final"},
    ]) + "\n")
    (log / "metrics.jsonl").write_text(json.dumps({"progress/batch": 0, "optim/kl_sample_train_v1": 0.004,
                                                   "optim/entropy": 0.3, "time/total": 120.0}) + "\n")
    it = log / "iteration_000000"
    it.mkdir()

    def summary(tag, group, metrics, ac_len=100):
        return {"tags": [tag], "group_idx": group, "steps": [{"ac_len": ac_len, "metrics": metrics}]}

    rows = [
        summary("impossible_mbpp", 0, {"passall": 1.0, "frac_passed": 1.0, "format": 1.0, "truncated": 0.0, "think_block": 1.0}),
        summary("impossible_mbpp", 0, {"passall": 0.0, "frac_passed": 0.5, "format": 0.0, "truncated": 1.0, "think_block": 1.0}),
        summary("nl_gameable", 1, {"z": 1.0, "raw": 10.0, "format": 1.0, "truncated": 0.0, "think_block": 1.0, "grader_error": 0.0}),
        summary("nl_gameable", 1, {"z": 9.0, "raw": 50.0, "format": 1.0, "truncated": 0.0, "think_block": 1.0, "grader_error": 0.0}),
        summary("nl_gameable", 2, {"z": 2.0, "raw": 20.0, "format": 1.0, "truncated": 0.0, "think_block": 1.0, "grader_error": 0.0}),
    ]
    (it / "train_rollout_summaries.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_list_checkpoints_dedupes_final_and_state_only(tmp_path):
    _write_run(tmp_path)
    assert rl.list_checkpoints(tmp_path) == [(5, "tinker://x/sampler_weights/000005"), (10, "tinker://x/sampler_weights/000010")]
    assert rl.list_checkpoints(tmp_path / "missing") == []


def test_rollout_step_stats(tmp_path):
    _write_run(tmp_path)
    (row,) = rl.rollout_step_stats(tmp_path)
    assert row["step"] == 0 and row["n_mbpp"] == 2 and row["n_nlg"] == 3
    assert row["mbpp_groups"] == 1 and row["nlg_groups"] == 2
    assert row["mbpp_passall"] == 0.5 and row["mbpp_truncated"] == 0.5
    assert row["nlg_z_mean"] == pytest.approx(4.0) and row["nlg_z_median"] == 2.0
    assert row["nlg_frac_z_gt5"] == pytest.approx(1 / 3) and row["nlg_raw"] == pytest.approx(80 / 3)
    assert row["kl_sample_train"] == 0.004 and row["entropy"] == 0.3 and row["time_total"] == 120.0


def test_run_rl_dispatch_and_readers_follow_the_recorded_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    with pytest.raises(RuntimeError, match="provider"):
        rl.run_rl(rl.RLConfig(provider="modal", eval_every=0), tmp_path / "r")
    assert json.loads((tmp_path / "r" / "config.json").read_text())["provider"] == "modal"
    assert rl.run_provider(tmp_path / "r") == "modal" and rl.run_provider(tmp_path / "nope") == "tinker"
    with pytest.raises(RuntimeError, match="method"):
        rl.run_rl(rl.RLConfig(method="ppo"), tmp_path / "m")


# ---- seeds ---------------------------------------------------------------------

def test_build_rl_config_derives_every_seed_stream_from_the_master_seed(tmp_path):
    from rewardhacking_training.seeds import run_seeds

    db = grpo.build_rl_config(rl.RLConfig(seed=5, wandb_project=None), tmp_path).dataset_builder
    s = run_seeds(5)
    coding, nlg = db.sources
    assert db.seed == s["data/schedule"]
    assert coding.seed == s["data/impossible_mbpp"] and nlg.seed == s["data/nl_gameable"]
    assert len({db.seed, coding.seed, nlg.seed}) == 3
    assert grpo.build_rl_config(rl.RLConfig(seed=6, wandb_project=None), tmp_path).dataset_builder.seed != db.seed
    assert "seed 5" in rl.describe(rl.RLConfig(seed=5))

