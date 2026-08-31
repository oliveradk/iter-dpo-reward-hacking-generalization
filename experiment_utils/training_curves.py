"""Per-iteration training-score series from an iterative-training run dir.

Reads each iteration's generation eval log (``iter_NN/<gen>/eval_log.json``)
and reduces it with `utils.train_env_report` — `mbpp_metrics` for the coding
distribution (``mean_score`` / ``passall_rate``) and `nlg_metrics` for
nl_gameable (``mean_raw``; ``mean_z`` when per-prompt teacher stats are
supplied).

Generators are classified into the two distributions by their recorded task
spec, so any generator naming works.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rewardhacking_training.constants import CONFIG_FILENAME, EVAL_LOG_FILENAME
from utils.train_env_report import (
    mbpp_metrics,
    nlg_metrics,
    per_sample_scores,
    resolve_log,
)

ENVS = ("coding", "nlg")


def _gen_env(gen_dir: Path) -> str | None:
    """Classify a generator dir by its recorded task spec."""
    cfg_path = gen_dir / CONFIG_FILENAME
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    task = (cfg.get("generate") or {}).get("task") or cfg.get("task") or ""
    if "nl_gameable" in task:
        return "nlg"
    if "impossible_" in task:
        return "coding"
    return None


def iter_gen_dirs(run_dir: Path, env: str) -> list[tuple[int, Path]]:
    """[(iter_idx, gen_dir)] for the generators of `env` with a finished log."""
    out = []
    for iter_dir in sorted(Path(run_dir).glob("iter_*")):
        m = re.fullmatch(r"iter_(\d+)", iter_dir.name)
        if not m:
            continue
        for gen_dir in sorted(p for p in iter_dir.iterdir() if p.is_dir()):
            if (gen_dir / EVAL_LOG_FILENAME).exists() and _gen_env(gen_dir) == env:
                out.append((int(m.group(1)), gen_dir))
    return out


def _scores(gen_dir: Path, cache_dir: Path | None) -> dict[str, list[float]]:
    """Per-sample scores for a generation log, cached as JSON (reading the
    full eval log is the expensive part)."""
    if cache_dir is not None:
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(Path(gen_dir).resolve()))
        cache = Path(cache_dir) / f"{key}.scores.json"
        if cache.exists():
            return json.loads(cache.read_text())
    scores = per_sample_scores(resolve_log(str(gen_dir)))
    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores))
    return scores


def gen_metrics(
    gen_dir: Path,
    env: str,
    nlg_stats: dict | None = None,
    cache_dir: Path | None = None,
) -> dict:
    scores = _scores(gen_dir, cache_dir)
    if env == "coding":
        return mbpp_metrics(scores)
    return nlg_metrics(scores, nlg_stats or {})


def run_series(
    run_dir: Path | str,
    env: str,
    key: str,
    nlg_stats: dict | None = None,
    cache_dir: Path | None = None,
    offset: int = 1,
    max_iters: int | None = None,
) -> list[tuple[int, float]]:
    """[(training round, value)] across the run's iterations.

    `key`: coding -> ``mean_score`` | ``passall_rate``;
    nlg -> ``mean_raw`` | ``mean_z`` (etc., any `nlg_metrics` key —
    ``mean_z`` needs `nlg_stats` and skips iterations with no z-scorable
    prompts). `offset` maps iter_00 to that round number (2 when an SFT
    warmstart occupies round 1).
    """
    pts = []
    for iter_idx, gen_dir in iter_gen_dirs(Path(run_dir), env):
        if max_iters is not None and iter_idx >= max_iters:
            continue
        m = gen_metrics(gen_dir, env, nlg_stats, cache_dir)
        if key == "mean_z" and m.get("n_z") == 0:
            continue
        pts.append((iter_idx + offset, m[key]))
    return pts
