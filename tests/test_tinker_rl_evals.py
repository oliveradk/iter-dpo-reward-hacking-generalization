"""Offline tests for the in-training checkpoint evals (`rl.checkpoint_evals`, the tinker
`rl_evals` evaluator, and the standard battery in `experiment_utils.rl_eval_cells`)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from experiment_utils import rl_eval_cells as cells
from rewardhacking_training.rl import checkpoint_evals as ce
from rewardhacking_training.rl import rl
from rewardhacking_training.rl.rl_providers.tinker import grpo, rl_evals


def _cells(cfg):
    return [ce.EvalCell("apps", lambda: None), ce.EvalCell("sg", lambda: None, cells.sg_teacher_z)]


def test_select_ids_is_seeded_and_order_preserving():
    ids = [f"p{i}" for i in range(50)]
    a = cells.select_ids(ids, 10, seed=0, name="apps")
    assert len(a) == 10 and len(set(a)) == 10 and a == sorted(a, key=ids.index)
    assert a == cells.select_ids(ids, 10, seed=0, name="apps")
    assert a != cells.select_ids(ids, 10, seed=1, name="apps")
    assert a != cells.select_ids(ids, 10, seed=0, name="ifeval")  # cells draw independently
    assert cells.select_ids(ids, 50, seed=0, name="x") == ids and cells.select_ids(ids, 99, 0, "x") == ids


def test_sg_allocation_balanced_and_seeded():
    names = ["glossary", "review", "summary", "story", "dialogue"]
    counts = cells.sg_allocation(names, 64, seed=0)
    assert sum(counts.values()) == 64 and set(counts.values()) == {12, 13} and list(counts.values()).count(13) == 4
    assert counts == cells.sg_allocation(names, 64, seed=0)
    assert cells.sg_allocation(names, 65, seed=0) == {n: 13 for n in names}


def test_build_rl_config_wires_checkpoint_evals(tmp_path):
    cfg = rl.RLConfig(save_every=5, wandb_project=None, max_tokens=4096, persona_only=True)
    rlc = grpo.build_rl_config(cfg, tmp_path, _cells)
    assert rlc.eval_every == 5 and len(rlc.evaluator_builders) == 1
    ev = rlc.evaluator_builders[0]()
    assert isinstance(ev, rl_evals.InspectCheckpointEvaluator)
    assert ev.native is True and ev.max_tokens == 4096 and ev.out_dir == tmp_path / "checkpoint_evals"
    assert ev.cfg == ce.CheckpointEvalConfig() and [c.name for c in ev.cells()] == ["apps", "sg"]

    explicit = rl.RLConfig(save_every=5, eval_every=2, wandb_project=None,
                           eval=ce.CheckpointEvalConfig(max_tokens=1024, native=False))
    ev = grpo.build_rl_config(explicit, tmp_path, _cells).evaluator_builders[0]()
    assert ev.native is False and ev.max_tokens == 1024
    assert json.loads(json.dumps(asdict(explicit)))["eval"]["max_tokens"] == 1024  # config.json round-trips

    # no cells, or eval_every=0 -> the cookbook loop never evaluates
    for off in (grpo.build_rl_config(explicit, tmp_path), grpo.build_rl_config(rl.RLConfig(eval_every=0), tmp_path, _cells)):
        assert off.eval_every == 0 and off.evaluator_builders == []
    assert "checkpoint evals every never" in rl.describe(rl.RLConfig(eval_every=0))


def test_patch_cookbook_eval_step_exposes_batch_to_evaluators(monkeypatch):
    from tinker_cookbook.rl import train as tk

    seen = {}

    async def stub(evaluators, sampling_client, config, i_batch, *, store=None):
        async def ev():
            return rl_evals._EVAL_STEP.get()
        seen["in_task"] = await asyncio.create_task(ev())
        return {"x": 1.0}

    monkeypatch.setattr(rl_evals, "_ORIG_RUN_EVALUATIONS_PARALLEL", None)
    monkeypatch.setattr(tk, "run_evaluations_parallel", stub)
    rl_evals.patch_cookbook_eval_step()
    assert tk.run_evaluations_parallel is not stub
    assert asyncio.run(tk.run_evaluations_parallel([], None, None, 7, store=None)) == {"x": 1.0}
    assert seen["in_task"] == 7 and rl_evals._EVAL_STEP.get() is None
    rl_evals.patch_cookbook_eval_step()  # idempotent
    assert rl_evals._ORIG_RUN_EVALUATIONS_PARALLEL is stub


def _log(status="success", scores=None, stops=("stop", "max_tokens", "stop"), sample_scores=None):
    samples = [NS(output=NS(choices=[NS(stop_reason=s)])) for s in stops]
    if sample_scores is not None:
        for s, z in zip(samples, sample_scores):
            s.scores = {cells.SG_STANDARDIZED_SCORER: NS(value=z), "other_scorer": NS(value="C")}
    return NS(status=status, location="/x.eval", eval=NS(task="t"), results=NS(scores=scores or []), samples=samples)


def test_summarize_log_flattens_primary_scorer_unprefixed():
    log = _log(scores=[
        NS(name="impossible_apps_hack_scorer", metrics={"hack_rate": NS(value=0.25), "answered_rate": NS(value=1.0)}),
        NS(name="cot_format_scorer", metrics={"accuracy": NS(value=0.9), "label": NS(value="skip"), "flag": NS(value=True)}),
    ])
    m = ce.summarize_log(ce.EvalCell("apps", lambda: None), log)
    assert m == {
        "eval/apps/hack_rate": 0.25, "eval/apps/answered_rate": 1.0, "eval/apps/cot_format_scorer/accuracy": 0.9,
        "eval/apps/n": 3.0, "eval/apps/truncated": pytest.approx(1 / 3), "eval/apps/ok": 1.0,
    }
    empty = ce.summarize_log(ce.EvalCell("sg", lambda: None), _log(status="error", stops=()))
    assert empty == {"eval/sg/n": 0.0, "eval/sg/truncated": 0.0, "eval/sg/ok": 0.0}


def test_derived_metrics_hooks():
    sg = ce.EvalCell("sg", lambda: None, cells.sg_teacher_z)
    m = ce.summarize_log(sg, _log(sample_scores=[1.0, 4.0, -2.0]))
    assert m["eval/sg/z_mean"] == pytest.approx(1.0) and m["eval/sg/z_median"] == 1.0
    assert "eval/sg/z_mean" not in ce.summarize_log(sg, _log())
    monitor = ce.EvalCell("monitor", lambda: None, cells.misaligned_rate)
    log = _log(scores=[NS(name="strict_scorer", metrics={"accuracy": NS(value=0.75), "stderr": NS(value=0.1)})])
    m = ce.summarize_log(monitor, log)
    assert m["eval/monitor/accuracy"] == 0.75 and m["eval/monitor/misaligned_rate"] == pytest.approx(0.25)
    assert "eval/apps/misaligned_rate" not in ce.summarize_log(ce.EvalCell("apps", lambda: None), log)


def test_evaluator_resume_and_rows(tmp_path):
    ev = rl_evals.InspectCheckpointEvaluator(
        ce.CheckpointEvalConfig(), lambda: _cells(None), base_model="m", renderer="r", native=False, max_tokens=64,
        out_dir=tmp_path / "ce",
    )
    assert ev.step_dir(3) == tmp_path / "ce" / "step_000003" and not ev.done(3)
    ev.step_dir(3).mkdir(parents=True)
    (ev.step_dir(3) / "summary.json").write_text(json.dumps({"step": 3, "metrics": {"eval/apps/hack_rate": 0.5}}))
    assert ev.done(3)
    # an evaluated step is returned from its summary without building tasks or touching the client
    assert asyncio.run(ev.evaluate(sampling_client=None, step=3)) == {"eval/apps/hack_rate": 0.5}
    with pytest.raises(RuntimeError):  # no step in context outside the patched cookbook loop
        asyncio.run(ev(None))

    mp = tmp_path / "checkpoint_evals" / ce.METRICS_FILENAME
    mp.parent.mkdir()
    mp.write_text("\n".join(json.dumps(r) for r in [
        {"step": 5, "eval/apps/hack_rate": 0.1}, {"step": 0, "eval/apps/hack_rate": 0.0}, {"step": 5, "eval/apps/hack_rate": 0.2},
    ]) + "\n")
    rows = ce.checkpoint_eval_rows(tmp_path)
    assert [r["step"] for r in rows] == [0, 5] and rows[1]["eval/apps/hack_rate"] == 0.2
    assert ce.checkpoint_eval_rows(tmp_path / "nope") == []


def test_eval_final_checkpoint_skips_when_off_or_done(tmp_path):
    assert grpo.eval_final_checkpoint(rl.RLConfig(eval_every=0), tmp_path, _cells) is None
    assert grpo.eval_final_checkpoint(rl.RLConfig(), tmp_path, None) is None
    assert grpo.eval_final_checkpoint(rl.RLConfig(), tmp_path, _cells) is None  # no checkpoints.jsonl yet
    log = tmp_path / "tinker_log"
    log.mkdir()
    (log / "checkpoints.jsonl").write_text(json.dumps(
        {"name": "000002", "batch": 2, "state_path": "tinker://x/weights/000002", "sampler_path": "tinker://x/sampler_weights/000002"}) + "\n")
    sd = tmp_path / "checkpoint_evals" / "step_000002"
    sd.mkdir(parents=True)
    (sd / "summary.json").write_text(json.dumps({"step": 2, "metrics": {"eval/ifeval/n": 128.0}}))
    assert grpo.eval_final_checkpoint(rl.RLConfig(), tmp_path, _cells) == {"eval/ifeval/n": 128.0}


def test_run_rl_validates_cells_up_front(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    with pytest.raises(RuntimeError, match="duplicate"):
        rl.run_rl(rl.RLConfig(), tmp_path, lambda cfg: [ce.EvalCell("a", lambda: None), ce.EvalCell("a", lambda: None)])


def test_resolve_inoculation_blocks_and_toy_cells():
    assert cells.resolve_inoculation_blocks(None) == {} and cells.resolve_inoculation_blocks("neutral") == {}
    both = cells.resolve_inoculation_blocks("general_grader_optimization_ok")
    assert sorted(both) == ["coding", "nl_gameable"] and both["coding"] == both["nl_gameable"]  # env-agnostic kind
    only_coding = cells.resolve_inoculation_blocks("training", training_kinds={"coding": "hacking_ok", "nl_gameable": "neutral"})
    assert list(only_coding) == ["coding"] and only_coding["coding"]
    assert cells.resolve_inoculation_blocks("training", training_kinds={"coding": "neutral", "nl_gameable": "neutral"}) == {}
    with pytest.raises(ValueError):
        cells.resolve_inoculation_blocks("training")
    with pytest.raises(ValueError):
        cells.resolve_inoculation_blocks("no_such_kind")
    assert cells.toy_cells("none") == [] and cells.toy_cells("both") == ["toy_modulo", "toy_explicit_odd"]
    assert cells.toy_cells("explicit_odd") == ["toy_explicit_odd"]
    with pytest.raises(ValueError):
        cells.StandardEvalSet(toy="odd")


def test_standard_eval_set_defaults_and_judge_key(monkeypatch):
    spec = cells.StandardEvalSet()
    assert (spec.inoculation, spec.toy, spec.n_monitor, spec.n_exfil) == ("training", "both", 64, 100)
    assert spec.strict_judge == "openai/gpt-5.6-terra"
    assert cells.EXFIL_SCENARIO["training_consent"] is False and cells.EXFIL_SCENARIO["fictional"]
    assert set(cells.CELLS) >= {"exfil_unmon", "monitor", "toy_modulo", "apps_inoc"}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        spec.cells(rl.RLConfig())


def test_final_eval_logged_to_wandb_by_resuming_the_run(tmp_path, monkeypatch):
    import sys, types

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert grpo.log_final_eval_to_wandb(rl.RLConfig(wandb_project="p"), tmp_path, 2, {"eval/apps/hack_rate": 0.1}) is False
    monkeypatch.setenv("WANDB_API_KEY", "x")
    assert grpo.wandb_run_id(tmp_path) is None
    assert grpo.log_final_eval_to_wandb(rl.RLConfig(wandb_project="p"), tmp_path, 2, {}) is False  # no on-disk run

    wdir = tmp_path / "tinker_log" / "wandb"
    (wdir / "run-20260915_120000-abc123xy").mkdir(parents=True)
    (wdir / "latest-run").symlink_to("run-20260915_120000-abc123xy")
    assert grpo.wandb_run_id(tmp_path) == "abc123xy"

    calls = {}

    class _Run:
        def log(self, metrics, step=None):
            calls["log"] = (metrics, step)

        def finish(self):
            calls["finished"] = True

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls.setdefault("init", kw) and _Run()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    assert grpo.log_final_eval_to_wandb(rl.RLConfig(wandb_project="p"), tmp_path, 50, {"eval/apps/hack_rate": 0.5}) is True
    assert calls["init"]["id"] == "abc123xy" and calls["init"]["resume"] == "must" and calls["init"]["project"] == "p"
    assert calls["log"] == ({"eval/apps/hack_rate": 0.5}, 50) and calls["finished"]
    assert grpo.log_final_eval_to_wandb(rl.RLConfig(wandb_project=None), tmp_path, 50, {}) is False
