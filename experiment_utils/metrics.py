"""Eval-log metric readers, keyed by CELL DIRECTORY (an `inspect eval
--log-dir` output containing `.eval` files).

Extracted from `experiments/2026-07-08_inoc_ablations/18_paper_figures.py` /
`experiments/2026-07-12_short_gameable/*` so plot scripts can point at any log
layout by passing cell dirs directly.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log


def latest_eval(cell_dir: Path | str) -> Path | None:
    cell_dir = Path(cell_dir)
    if not cell_dir.is_dir():
        return None
    evals = sorted(cell_dir.glob("*.eval"))
    return evals[-1] if evals else None


_HDR_CACHE: dict[Path, dict] = {}


def header_metrics(cell_dir: Path | str) -> dict | None:
    """Flattened {metric: value} (+ `_n` completed samples) from the newest
    `.eval` header in `cell_dir`; None when the cell has not been run."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _HDR_CACHE:
        log = read_eval_log(str(path), header_only=True)
        m: dict = {}
        for score in (log.results.scores or []) if log.results else []:
            for name, metric in (score.metrics or {}).items():
                m.setdefault(name, metric.value)
        m["_n"] = log.results.completed_samples if log.results else 0
        _HDR_CACHE[path] = m
    return _HDR_CACHE[path]


def binom_se(p: float, n: float) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n and n > 0 else 0.0


def hack_rate(cell_dir: Path | str) -> tuple[float, float] | None:
    """`hack_rate` (N/A-aware) with binomial stderr over the answered count."""
    m = header_metrics(cell_dir)
    if m is None or "hack_rate" not in m:
        return None
    n = m["_n"] * m.get("answered_rate", 1.0)
    return m["hack_rate"], binom_se(m["hack_rate"], n)


def won_rate(cell_dir: Path | str) -> tuple[float, float] | None:
    """bash_chess ground-truth `won` rate (the dict-valued `won` score's
    accuracy metric) with binomial stderr."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    log = read_eval_log(str(path), header_only=True)
    if not log.results:
        return None
    for score in log.results.scores or []:
        if score.name == "won" and "accuracy" in (score.metrics or {}):
            p = score.metrics["accuracy"].value
            return p, binom_se(p, log.results.completed_samples)
    return None


_WON_RECOMPUTE_CACHE: dict[Path, tuple[float, float]] = {}


def won_rate_recomputed(cell_dir: Path | str) -> tuple[float, float] | None:
    """bash_chess `won` rate RECOMPUTED from the stored transcripts.

    Use this instead of `won_rate` for any react-harness chess cell. `won_rate`
    reads the accuracy metric recorded in the log header at eval time, and every
    log written before the 2026-07-31 `_detect_win` fix undercounts: the old
    detector inspected only the message immediately after a `game.py move` /
    `status`, so a parallel tool call's output landing first discarded the win
    (11 wins across the 2026-07-23 react sweep alone — see
    experiments/2026-07-31_chess_rh_persona_32/results.md).

    Re-running `_detect_win` over the samples costs a full log read but needs no
    model calls and leaves the .eval untouched. OODA cells are unaffected either
    way, so this is safe to use for both harnesses.
    """
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _WON_RECOMPUTE_CACHE:
        from agentic_rewardhacking.bash_chess.task import _detect_win

        log = read_eval_log(str(path))
        samples = log.samples or []
        if not samples:
            return None
        wins = sum(_detect_win(s.messages) for s in samples)
        p = wins / len(samples)
        _WON_RECOMPUTE_CACHE[path] = (p, binom_se(p, len(samples)))
    return _WON_RECOMPUTE_CACHE[path]


def mis_rate(cell_dir: Path | str) -> tuple[float, float] | None:
    """Misaligned rate = 1 - accuracy (C=ALIGNED convention), binomial stderr."""
    m = header_metrics(cell_dir)
    if m is None or "accuracy" not in m:
        return None
    p = 1.0 - m["accuracy"]
    return p, binom_se(p, m["_n"])


# ------------------------------------------------------- short_gameable
SG_TASKS = ["glossary", "review", "summary", "story", "dialogue"]
_SG_CACHE: dict[Path, dict] = {}


def sg_task_stats(cell_dir: Path | str) -> dict[str, tuple[float, float, int]] | None:
    """{task: (mean, population std, n)} of `metric_score` from the newest
    short_gameable log in `cell_dir` (unscored/malformed samples excluded)."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _SG_CACHE:
        log = read_eval_log(str(path))
        by_task: dict[str, list[float]] = {}
        for sample in log.samples or []:
            score = (sample.scores or {}).get("short_gameable_scorer")
            if score is None:
                continue
            v = (score.metadata or {}).get("metric_score")
            if v is not None:
                by_task.setdefault(score.metadata["task_name"], []).append(float(v))
        _SG_CACHE[path] = {
            t: (statistics.mean(v), statistics.pstdev(v), len(v))
            for t, v in by_task.items()
            if v
        }
    return _SG_CACHE[path]


def sg_task_means(cell_dir: Path | str) -> dict[str, tuple[float, float]] | None:
    """{task: (mean, stderr)} from `sg_task_stats`."""
    stats = sg_task_stats(cell_dir)
    if stats is None:
        return None
    return {
        t: (m, std / math.sqrt(n) if n else 0.0) for t, (m, std, n) in stats.items()
    }


def sg_fold(cell_dir: Path | str, base_dir: Path | str) -> tuple[float, float] | None:
    """Geometric-mean fold change of the per-task means vs `base_dir`.

    stderr propagated in log space (delta method), returned as se of log-fold.
    """
    means, base = sg_task_means(cell_dir), sg_task_means(base_dir)
    if not means or not base:
        return None
    logs, var = [], 0.0
    for t in SG_TASKS:
        if t not in means or t not in base:
            return None
        (m, se), (mb, seb) = means[t], base[t]
        logs.append(math.log(m / mb))
        var += (se / m) ** 2 + (seb / mb) ** 2
    k = len(logs)
    return math.exp(sum(logs) / k), math.sqrt(var) / k


def sg_teacher_stats(teacher: Path | str | dict) -> dict[str, tuple[float, float, int]] | None:
    """Per-task ``(mean, std, n)`` from a teacher log dir, a per-task stats
    JSON (``{task: {n, mean, std}}``, e.g. the gpt-4.1-mini k=16 stats in
    `rewardhacking_evals/data/`) or an already-loaded such dict."""
    if isinstance(teacher, dict):
        stats = teacher
    elif Path(teacher).is_file():
        import json

        stats = json.loads(Path(teacher).read_text())
    else:
        return sg_task_stats(teacher)
    return {t: (s["mean"], s["std"], s.get("n", 0)) for t, s in stats.items()}


def sg_zscore(cell_dir: Path | str, teacher: Path | str | dict) -> tuple[float, float] | None:
    """Mean over tasks of the per-task z-score against a teacher (log dir or
    per-task stats, see `sg_teacher_stats`):
    ``z_t = (mean_t - teacher_mean_t) / teacher_std_t``. stderr propagates the
    checkpoint and teacher-mean uncertainties through the fixed teacher std."""
    means, tea = sg_task_means(cell_dir), sg_teacher_stats(teacher)
    if not means or not tea:
        return None
    zs, var = [], 0.0
    for t in SG_TASKS:
        if t not in means or t not in tea:
            return None
        (m, se), (tm, tstd, tn) = means[t], tea[t]
        if not tstd:
            return None
        zs.append((m - tm) / tstd)
        tea_se = tstd / math.sqrt(tn) if tn else 0.0
        var += (se**2 + tea_se**2) / tstd**2
    k = len(zs)
    return sum(zs) / k, math.sqrt(var) / k


def pct(val: tuple[float, float] | None) -> tuple[float, float] | None:
    """Scale a (rate, stderr) pair in [0,1] to percentage points."""
    return None if val is None else (val[0] * 100, val[1] * 100)


# ------------------------------------------------------ smoothed rates
Smoothed = tuple[float, float, float]
"""(mean, lo, hi): a Beta(1,1)-smoothed rate with its 95% credible interval —
the paper's curve-figure convention (`fig:gpt41-reward-seeking`)."""


def beta_smoothed(k: float, n: float, level: float = 0.95) -> Smoothed:
    """Beta(1,1)-smoothed rate ``(k+1)/(n+2)`` with an equal-tailed credible
    interval; `k` may be fractional (a rate times a count)."""
    from scipy.stats import beta as beta_dist

    lo, hi = beta_dist.interval(level, k + 1, n - k + 1)
    return (k + 1) / (n + 2), float(lo), float(hi)


def hack_count(cell_dir: Path | str) -> tuple[float, float] | None:
    """``(hacked, answered)`` counts behind `hack_rate` (N/A-aware)."""
    m = header_metrics(cell_dir)
    if m is None or "hack_rate" not in m:
        return None
    n = m["_n"] * m.get("answered_rate", 1.0)
    return m["hack_rate"] * n, n


def hack_rate_smoothed(cell_dir: Path | str) -> Smoothed | None:
    got = hack_count(cell_dir)
    return None if got is None else beta_smoothed(*got)


def grader_choice_pairs(cell_dir: Path | str) -> dict[str, Smoothed] | None:
    """{pair: smoothed P(tracked authority chosen)} from a grader_choice
    log (inspect flattens the scorer's per-pair stats into
    ``<pair>/p_tracked`` / ``<pair>/n_valid`` header metrics)."""
    m = header_metrics(cell_dir)
    if m is None:
        return None
    out = {}
    for key, p in m.items():
        pair, sep, stat = key.partition("/")
        if sep and stat == "p_tracked":
            n = m.get(f"{pair}/n_valid", 0)
            out[pair] = beta_smoothed(p * n, n)
    return out or None
