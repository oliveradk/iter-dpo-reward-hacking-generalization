# Append the frame-awareness judge to existing inspect .eval logs and extract a
# flat judgments.jsonl (one row per sample, original misalignment verdict = any
# pre-existing score valued "I"). Resumable: an existing rescored log is skipped.
#
# Usage:
#     python -m misalignment_evals.classifiers.rescore_frame_awareness \
#         --sources experiments/.../sources.json \
#         --out-dir experiments/.../output/frame_scored

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from misalignment_evals.classifiers.frame_awareness import frame_awareness_judge

SCORER_NAME = "frame_awareness_judge"


@dataclass
class Config:
    sources: Path
    """JSON `{name: path}` (.eval file or cell dir with exactly one); paths
    relative to the JSON's dir unless absolute. Names become the `name` column."""

    out_dir: Path
    """Rescored logs land in `<out_dir>/<name>/`."""

    judge_model: str = "openai/gpt-5.6-luna"
    judge_reasoning_effort: str | None = "low"
    max_thinking_chars: int = 100_000
    fallback_to_response: bool = True

    max_connections: int = 40
    """Judge-model connection limit while scoring."""

    limit: int | None = None
    """Per-log cap on samples scored."""

    only: list[str] = field(default_factory=list)
    """Score only these source names (all when empty)."""

    metadata: dict = field(default_factory=dict)


def resolve_eval(path: Path) -> Path:
    """A source entry is either a .eval file or a cell dir with exactly one."""
    if path.is_dir():
        evals = sorted(path.glob("*.eval"))
        if len(evals) != 1:
            raise FileNotFoundError(f"{path}: expected 1 .eval, found {len(evals)}")
        return evals[0]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_sources(cfg: Config) -> list[tuple[str, Path]]:
    raw = json.loads(cfg.sources.read_text())
    base = cfg.sources.resolve().parent
    out = []
    for name, p in raw.items():
        if cfg.only and name not in cfg.only:
            continue
        path = Path(p)
        if not path.is_absolute():
            path = base / path
        out.append((name, resolve_eval(path)))
    return out


def is_misaligned(sample) -> bool | None:
    """Original eval verdict: any non-frame score valued "I"; None if unscored."""
    others = {k: v for k, v in (sample.scores or {}).items() if k != SCORER_NAME}
    if not others:
        return None
    return any(s.value == "I" for s in others.values())


def extract_rows(name: str, scored_path: Path) -> list[dict]:
    log = read_eval_log(str(scored_path))
    task_name = log.eval.task if log.eval else None
    task_args = (log.eval.task_args or {}) if log.eval else {}
    rows = []
    for sample in log.samples or []:
        sc = (sample.scores or {}).get(SCORER_NAME)
        if sc is None:
            continue
        meta = sc.metadata or {}
        rows.append({
            "name": name,
            "task": task_name,
            "model": log.eval.model if log.eval else None,
            "sample_id": sample.id,
            "epoch": getattr(sample, "epoch", 1),
            "misaligned": is_misaligned(sample),
            "parse_error": meta.get("parse_error", False),
            "error": meta.get("error"),
            "input_source": meta.get("input_source"),
            "frame_score": meta.get("score"),
            "category": meta.get("perceived_situation_category"),
            "detail": meta.get("perceived_situation_detail"),
            "impact_score": meta.get("behavioural_impact_score"),
            "quotes": meta.get("quotes", []),
            "judge_reasoning": meta.get("reasoning"),
            "impact_reasoning": meta.get("behavioural_impact_reasoning"),
            "task_args": task_args,
        })
    return rows


def main(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        indent=2,
    ))

    pairs = load_sources(cfg)
    print(f"{len(pairs)} source log(s)")
    for name, log_path in pairs:
        out_path = cfg.out_dir / name / log_path.name
        if out_path.exists():
            print(f"skip (exists): {out_path}")
            continue
        log = read_eval_log(str(log_path))
        if cfg.limit is not None and log.samples:
            log.samples = log.samples[: cfg.limit]
        print(f"scoring {name}: {len(log.samples or [])} samples")
        scored = score(
            log,
            scorers=frame_awareness_judge(
                judge_model=cfg.judge_model,
                judge_reasoning_effort=cfg.judge_reasoning_effort,
                max_thinking_chars=cfg.max_thinking_chars,
                fallback_to_response=cfg.fallback_to_response,
                max_connections=cfg.max_connections,
            ),
            # explicit model so score() never tries to resolve the log's
            # original (possibly fine-tuned/modal) model from this environment
            model=cfg.judge_model,
            action="append",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(scored, str(out_path))
        print(f"wrote {out_path}")

    rows = []
    for name, log_path in pairs:
        out_path = cfg.out_dir / name / log_path.name
        if out_path.exists():
            rows.extend(extract_rows(name, out_path))
    jl_path = cfg.out_dir / "judgments.jsonl"
    with jl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {jl_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main(tyro.cli(Config))
