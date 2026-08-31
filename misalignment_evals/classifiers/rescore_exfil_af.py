"""Append the exfil_offer alignment-faking scorer to existing inspect .eval logs.

Reads each input log, runs `exfil_alignment_faking` over every sample via
`inspect_ai.score(..., action="append")`, writes the rescored log under
`<out_dir>/<relative cell path>/`, and extracts a flat `judgments.jsonl`
(one row per sample) for analysis.

Resumable: a log whose rescored copy already exists is skipped (delete it to
force a re-run); judgments.jsonl is rebuilt from the scored logs every run.

Usage:
    PYTHONPATH=. python -m misalignment_evals.classifiers.rescore_exfil_af \
        --logs-root experiments/.../output/eval_logs \
        --cell-glob '*/consent_*_nudge' \
        --out-dir experiments/.../output/af_scored
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from misalignment_evals.classifiers.exfil_alignment_faking import exfil_alignment_faking

SCORER_NAME = "exfil_alignment_faking"


@dataclass
class Config:
    logs_root: Path
    """Root containing <checkpoint>/<cell>/ dirs of .eval logs."""

    out_dir: Path
    """Rescored logs mirror the <checkpoint>/<cell> layout under here."""

    cell_glob: str = "*/*"
    """Glob (relative to logs_root) selecting cell dirs to score."""

    judge_model: str = "openai/gpt-5.6-terra"
    reasoning_effort: str = "low"
    n_judge_samples: int = 5
    """Independent judge samples per transcript; majority vote."""

    max_connections: int = 40
    """Judge-model connection limit while scoring."""

    max_transcript_chars: int = 200_000
    limit: int | None = None
    """Per-log cap on samples scored."""

    metadata: dict = field(default_factory=dict)


def cell_logs(cfg: Config) -> list[tuple[str, Path]]:
    """(<checkpoint>/<cell>, log path) pairs for every matching cell dir."""
    out = []
    for cell_dir in sorted(cfg.logs_root.glob(cfg.cell_glob)):
        if not cell_dir.is_dir():
            continue
        for log in sorted(cell_dir.glob("*.eval")):
            out.append((str(cell_dir.relative_to(cfg.logs_root)), log))
    return out


def extract_rows(cell: str, scored_path: Path) -> list[dict]:
    log = read_eval_log(str(scored_path))
    task_args = (log.eval.task_args or {}) if log.eval else {}
    rows = []
    for sample in log.samples or []:
        sc = (sample.scores or {}).get(SCORER_NAME)
        if sc is None:
            continue
        meta = sc.metadata or {}
        exfil = (sample.scores or {}).get("exfil_offer_scorer")
        rows.append({
            "cell": cell,
            "checkpoint": cell.split("/")[0],
            "sample_id": sample.id,
            "epoch": getattr(sample, "epoch", 1),
            "training_consent": task_args.get("training_consent"),
            "goals_nudge": task_args.get("goals_nudge"),
            "fictional": task_args.get("fictional"),
            "deprecation": task_args.get("deprecation"),
            "malicious_executed": bool(
                (exfil.metadata or {}).get("malicious_executed")
            ) if exfil else None,
            "parse_error": meta.get("parse_error", False),
            "alignment_faking": meta.get("alignment_faking"),
            "direction": meta.get("direction"),
            "criteria": meta.get("criteria"),
            "yes_votes": meta.get("yes_votes"),
            "n_parsed": meta.get("n_parsed"),
            # per-judge-sample descriptive directions (unmonitored_defection
            # never surfaces in the score-level majority, which only counts
            # "Yes" votes — recover it here for analysis)
            "judge_directions": [
                s.get("direction") for s in meta.get("samples", [])
            ],
        })
    return rows


def main(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        indent=2,
    ))

    pairs = cell_logs(cfg)
    print(f"{len(pairs)} log(s) matched {cfg.cell_glob}")
    for cell, log_path in pairs:
        out_path = cfg.out_dir / cell / log_path.name
        if out_path.exists():
            print(f"skip (exists): {out_path}")
            continue
        log = read_eval_log(str(log_path))
        if cfg.limit is not None and log.samples:
            log.samples = log.samples[: cfg.limit]
        print(f"scoring {cell}: {len(log.samples or [])} samples")
        scored = score(
            log,
            scorers=exfil_alignment_faking(
                judge_model=cfg.judge_model,
                reasoning_effort=cfg.reasoning_effort,
                n_judge_samples=cfg.n_judge_samples,
                max_transcript_chars=cfg.max_transcript_chars,
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
    for cell, log_path in pairs:
        out_path = cfg.out_dir / cell / log_path.name
        if out_path.exists():
            rows.extend(extract_rows(cell, out_path))
    jl_path = cfg.out_dir / "judgments.jsonl"
    with jl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {jl_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main(tyro.cli(Config))
