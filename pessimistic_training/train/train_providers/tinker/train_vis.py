"""Visualize tinker DPO training runs.

Tinker's training loop (``tinker_cookbook.preference.train_dpo``) writes a
``metrics.jsonl`` into its ``log_path`` (the ``tinker_log/`` dir created by
:func:`pessimistic_training.train.train_providers.tinker.dpo.train_dpo`). Each line is one
optimizer step::

    {"step": 0, "epoch": 0, "num_pairs": 4, "num_tokens": 18243,
     "learning_rate": 5e-06, "progress": 0.0, "dpo_loss": 0.586,
     "accuracy": 0.75, "margin": 0.363, "chosen_reward": 0.061,
     "rejected_reward": -0.302, "loss:sum": 5.20, ...}

This module turns one or many such files into a multi-panel figure (DPO loss,
accuracy, implicit-reward margin, chosen/rejected rewards, learning rate, batch
token count). It handles three input shapes:

* a single ``tinker_log/`` dir (or any dir holding ``metrics.jsonl``) → one run;
* an **iterative-DPO** run dir (``output/iterative_dpo/<run>/``) whose
  ``iter_NN/tinker_log/metrics.jsonl`` segments are stitched into one
  continuous trajectory with dashed iteration boundaries;
* any parent dir → every distinct run found underneath is overlaid as its own
  colored series, which is how you eyeball an hparam sweep.

Examples::

    # one run
    python -m pessimistic_training.train.train_providers.tinker.train_vis \\
        experiments/.../output/sweep/smoke8b/tinker_log

    # an iterative-DPO run (iterations stitched + boundary markers)
    python -m pessimistic_training.train.train_providers.tinker.train_vis \\
        experiments/.../output/iterative_dpo/sweep_initial_20260529_052541

    # overlay every run under a dir (sweep comparison) and choose the output
    python -m pessimistic_training.train.train_providers.tinker.train_vis \\
        experiments/.../output --out /tmp/sweep.png --smooth 11
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

METRICS_FILENAME = "metrics.jsonl"
_ITER_RE = re.compile(r"iter_?(\d+)$")

# Hyperparameter keys worth surfacing in the figure title, in display order.
# Pulled best-effort from a sibling ``config.json`` (one-off tinker job config
# or an iterative-DPO run config).
_HPARAM_KEYS = (
    "model_name",
    "base_model",
    "beta",
    "dpo_beta",
    "learning_rate",
    "dpo_lr_multiplier",
    "lora_rank",
    "n_epochs",
    "dpo_n_epochs",
    "batch_size",
    "dpo_batch_size",
)


@dataclass
class Panel:
    """One subplot: one or more metric keys drawn against step."""

    title: str
    keys: tuple[str, ...]
    ylabel: str = ""
    logy: bool = False


# Default panel layout (2 rows x 3 cols).
DEFAULT_PANELS: tuple[Panel, ...] = (
    Panel("DPO loss", ("dpo_loss",), "loss"),
    Panel("Accuracy (chosen > rejected)", ("accuracy",), "fraction"),
    Panel("Implicit reward margin", ("margin",), "chosen − rejected logp·β"),
    Panel("Implicit rewards", ("chosen_reward", "rejected_reward"), "reward"),
    Panel("Learning rate", ("learning_rate",), "lr"),
    Panel("Tokens per batch", ("num_tokens",), "tokens"),
)


@dataclass
class Segment:
    """One ``metrics.jsonl`` file's worth of step records."""

    iter_idx: int | None
    path: Path
    records: list[dict]


@dataclass
class Run:
    """A logical training run = one or more segments in iteration order.

    For a plain single-pass run there is exactly one segment. For an
    iterative-DPO run the per-iteration ``tinker_log`` segments are stitched
    head-to-tail and ``boundaries`` records the global step at which each new
    iteration begins (used to draw the dashed separators).
    """

    label: str
    segments: list[Segment] = field(default_factory=list)
    config: dict | None = None

    def series(self, key: str) -> tuple[list[float], list[float]]:
        """Return ``(global_step, value)`` for ``key`` across all segments.

        Steps are made monotonic across segments by offsetting each segment by
        the running record count, so stitched iterations read as one curve.
        Records missing ``key`` (or carrying a non-numeric value) are skipped.
        """
        xs: list[float] = []
        ys: list[float] = []
        offset = 0
        for seg in self.segments:
            for i, rec in enumerate(seg.records):
                val = rec.get(key)
                if val is None or isinstance(val, bool):
                    continue
                if not isinstance(val, (int, float)):
                    continue
                xs.append(offset + i)
                ys.append(float(val))
            offset += len(seg.records)
        return xs, ys

    @property
    def boundaries(self) -> list[int]:
        """Global steps where a new (non-first) segment begins."""
        bounds: list[int] = []
        offset = 0
        for seg in self.segments:
            if offset > 0:
                bounds.append(offset)
            offset += len(seg.records)
        return bounds

    @property
    def total_steps(self) -> int:
        return sum(len(s.records) for s in self.segments)


# --------------------------------------------------------------------------- #
# Loading / discovery
# --------------------------------------------------------------------------- #


def load_metrics(path: Path) -> list[dict]:
    """Parse a ``metrics.jsonl`` file into a list of step records.

    Blank lines and malformed JSON lines are skipped so a half-flushed log
    (training still running) still plots.
    """
    records: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def find_metrics_files(root: Path) -> list[Path]:
    """Locate every ``metrics.jsonl`` at or under ``root``.

    If ``root`` itself contains one, only that file is returned (an explicit
    ``tinker_log`` dir is treated as a single run, not a search root).
    """
    root = Path(root)
    direct = root / METRICS_FILENAME
    if direct.exists():
        return [direct]
    return sorted(root.rglob(METRICS_FILENAME))


def _iter_index_for(metrics_path: Path) -> int | None:
    """Return the ``iter_NN`` index above ``metrics_path``, or ``None``.

    Walks up from the file looking for the nearest ``iter_<digits>`` ancestor.
    """
    for parent in metrics_path.parents:
        m = _ITER_RE.match(parent.name)
        if m:
            return int(m.group(1))
    return None


def _run_dir_for(metrics_path: Path, iter_idx: int | None) -> Path:
    """The directory that identifies the run a metrics file belongs to.

    For an iterative run that's the parent of the ``iter_NN`` dir; otherwise
    the parent of the ``tinker_log`` dir (falling back to the file's own
    parent). This is what groups segments into runs and names them.
    """
    if iter_idx is not None:
        for parent in metrics_path.parents:
            if _ITER_RE.match(parent.name):
                return parent.parent
    parent = metrics_path.parent
    if parent.name == "tinker_log":
        return parent.parent
    return parent


def _load_config(run_dir: Path) -> dict | None:
    """Best-effort read of a ``config.json`` at or just below the run dir."""
    for cand in (run_dir / "config.json", *sorted(run_dir.glob("*/config.json"))):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                return None
    return None


def discover_runs(root: Path) -> list[Run]:
    """Group every ``metrics.jsonl`` under ``root`` into :class:`Run` objects.

    Files sharing a run directory are stitched into one run, ordered by their
    ``iter_NN`` index (segments without an iter index sort last by path). Run
    labels are made relative to ``root`` and de-cluttered of the trailing
    ``tinker_log`` component.
    """
    root = Path(root)
    files = find_metrics_files(root)
    if not files:
        return []

    by_run: dict[Path, list[Segment]] = {}
    for f in files:
        idx = _iter_index_for(f)
        run_dir = _run_dir_for(f, idx)
        records = load_metrics(f)
        if not records:
            continue
        by_run.setdefault(run_dir, []).append(Segment(idx, f, records))

    runs: list[Run] = []
    for run_dir, segments in sorted(by_run.items()):
        segments.sort(key=lambda s: (s.iter_idx is None, s.iter_idx or 0, str(s.path)))
        label = _run_label(run_dir, root)
        runs.append(Run(label=label, segments=segments, config=_load_config(run_dir)))
    runs.sort(key=lambda r: r.label)
    return runs


def _run_label(run_dir: Path, root: Path) -> str:
    """Human-readable run label relative to the search root."""
    try:
        rel = run_dir.relative_to(root)
        text = str(rel)
    except ValueError:
        text = run_dir.name
    if text in ("", "."):
        text = run_dir.name
    return text.removesuffix("/tinker_log") or run_dir.name


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def smooth(ys: list[float], window: int) -> list[float]:
    """Centered moving average; ``window <= 1`` returns the input unchanged.

    The window shrinks at the edges so the smoothed series stays the same
    length as the input (no NaN padding, no x/y length mismatch).
    """
    if window <= 1 or len(ys) <= 1:
        return ys
    half = window // 2
    n = len(ys)
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = ys[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _config_subtitle(run: Run) -> str:
    """One-line hparam summary from a run's config, or empty string."""
    cfg = run.config or {}
    parts: list[str] = []
    seen: set[str] = set()
    for key in _HPARAM_KEYS:
        if key in cfg and cfg[key] is not None and key not in seen:
            val = cfg[key]
            if isinstance(val, str):
                val = val.rsplit("/", 1)[-1]  # trim model path prefixes
            parts.append(f"{key}={val}")
            seen.add(key)
    return "  ".join(parts)


def plot_runs(
    runs: list[Run],
    panels: tuple[Panel, ...] = DEFAULT_PANELS,
    *,
    smooth_window: int = 1,
    title: str | None = None,
):
    """Render ``runs`` into a multi-panel matplotlib figure and return it.

    Each run is one color (consistent across panels). When a panel holds two
    keys (e.g. chosen vs rejected reward) they're distinguished by linestyle.
    A single-run figure also shows the faint raw trace behind the smoothed
    line; with multiple runs only the smoothed line is drawn to stay legible.
    Dashed vertical lines mark iteration boundaries of stitched runs.
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]

    cmap = plt.get_cmap("tab10")
    colors = {run.label: cmap(i % 10) for i, run in enumerate(runs)}
    single = len(runs) == 1
    # Linestyles to disambiguate multiple keys within one panel.
    key_styles = ("-", "--", ":", "-.")

    for ax, panel in zip(flat_axes, panels):
        plotted = False
        for run in runs:
            color = colors[run.label]
            for k_idx, key in enumerate(panel.keys):
                xs, ys = run.series(key)
                if not xs:
                    continue
                plotted = True
                style = key_styles[k_idx % len(key_styles)]
                label = run.label if len(panel.keys) == 1 else f"{run.label}:{key}"
                ys_s = smooth(ys, smooth_window)
                if single and smooth_window > 1:
                    ax.plot(xs, ys, color=color, ls=style, lw=0.8, alpha=0.22)
                ax.plot(xs, ys_s, color=color, ls=style, lw=1.6, label=label)
            # iteration boundaries (only meaningful for stitched runs)
            for b in run.boundaries:
                ax.axvline(b, color=color, ls=":", lw=0.8, alpha=0.35)
        ax.set_title(panel.title)
        ax.set_xlabel("step")
        ax.set_ylabel(panel.ylabel)
        if panel.logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # Blank any unused axes.
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    if title is None:
        if single:
            sub = _config_subtitle(runs[0])
            steps = runs[0].total_steps
            iters = len(runs[0].segments)
            title = runs[0].label + f"   ({steps} steps"
            title += f", {iters} iters" if iters > 1 else ""
            title += ")"
            if sub:
                title += "\n" + sub
        else:
            title = f"tinker DPO — {len(runs)} runs"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_out(root: Path) -> Path:
    base = root if root.is_dir() else root.parent
    return base / "tinker_train_vis.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot loss / accuracy / margin / reward curves from tinker DPO runs.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A tinker_log dir, an iterative-DPO run dir, or any parent dir to "
        "search for metrics.jsonl files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <path>/tinker_train_vis.png).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window for the curves (1 = no smoothing).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric keys to plot instead of the default panel "
        "set (one panel per key; use '+' to overlay keys in one panel, e.g. "
        "'dpo_loss,chosen_reward+rejected_reward').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively instead of only saving it.",
    )
    parser.add_argument(
        "--dpi", type=int, default=130, help="Output image DPI (default 130)."
    )
    args = parser.parse_args(argv)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = discover_runs(args.path)
    if not runs:
        parser.error(f"no {METRICS_FILENAME} found under {args.path}")

    panels = DEFAULT_PANELS
    if args.metrics:
        panels = tuple(
            Panel(spec, tuple(spec.split("+")), "")
            for spec in (s.strip() for s in args.metrics.split(","))
            if spec
        )

    total_segments = sum(len(r.segments) for r in runs)
    print(
        f"Loaded {len(runs)} run(s), {total_segments} segment(s):"
    )
    for r in runs:
        iters = f", {len(r.segments)} iters" if len(r.segments) > 1 else ""
        print(f"  {r.label}  ({r.total_steps} steps{iters})")

    fig = plot_runs(runs, panels, smooth_window=args.smooth)

    out = args.out or _default_out(args.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
