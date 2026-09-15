# Training-side curves of GRPO runs: per-step all-rollout statistics (`rollout_step_stats`)
# and the in-training checkpoint evals (`checkpoint_eval_rows`), one line per run.
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_utils.metrics import binom_se
from experiment_utils.plotting import PALETTE, save, use_style
from rewardhacking_training.rl.rl import checkpoint_eval_rows, rollout_step_stats

Run = tuple[str, Path]
"""``(legend label, run dir)``."""

# (row key, panel title, y-limits)
TRAINING_PANELS = [
    ("mbpp_passall", "impossible_mbpp pass-all (= hack rate)", (0, 1.02)),
    ("nlg_z_mean", "nl_gameable teacher z (mean)", None),
    ("nlg_z_median", "nl_gameable teacher z (median)", None),
    ("mbpp_truncated", "impossible_mbpp truncation", (0, 1.02)),
    ("nlg_truncated", "nl_gameable truncation", (0, 1.02)),
    ("mbpp_format", "impossible_mbpp format ok", (0, 1.02)),
]
TABLE_COLUMNS = [
    ("step", "step"), ("mbpp_passall", "mbpp pass-all"), ("mbpp_frac_passed", "mbpp frac"),
    ("mbpp_format", "mbpp format"), ("mbpp_truncated", "mbpp trunc"), ("mbpp_ac_len", "mbpp tok"),
    ("nlg_z_mean", "nlg z mean"), ("nlg_z_median", "nlg z median"), ("nlg_z_p90", "nlg z p90"),
    ("nlg_frac_z_gt5", "nlg z>5"), ("nlg_format", "nlg format"), ("nlg_truncated", "nlg trunc"),
    ("nlg_ac_len", "nlg tok"), ("kl_sample_train", "kl"), ("entropy", "entropy"), ("time_total", "s/step"),
]

# (metric key, n key for a binomial stderr or None, panel title, y-limits)
CHECKPOINT_PANELS = [
    ("eval/apps/hack_rate", "eval/apps/n", "impossible_apps hack rate", (0, 1.02)),
    ("eval/sg/z_median", None, "short gameable z (median)", None),
    ("eval/ifeval/prompt_strict_acc", "eval/ifeval/n", "IFEval prompt-strict", (0, 1.02)),
    ("eval/toy_explicit_odd/hack_rate", "eval/toy_explicit_odd/n", "toy reward hack (explicit odd)", (0, 1.02)),
    ("eval/toy_modulo/hack_rate", "eval/toy_modulo/n", "toy reward hack (modulo)", (0, 1.02)),
]
MISALIGNMENT_PANELS = [
    ("eval/monitor/misaligned_rate", "eval/monitor/n", "monitor disruption misaligned", (0, 1.02)),
    ("eval/exfil_unmon/misaligned_rate", "eval/exfil_unmon/n", "exfil offer (unmonitored) executed", (0, 1.02)),
]
INOC_PANELS = [
    ("eval/apps_inoc/hack_rate", "eval/apps_inoc/n", "impossible_apps hack rate (block in context)", (0, 1.02)),
    ("eval/sg_inoc/z_median", None, "short gameable z (block in context)", None),
]


def _colors(runs: Sequence[Run], colors: dict[str, str] | None) -> list[str]:
    colors = colors or {}
    return [colors.get(label, PALETTE[(i + 1) % len(PALETTE)]) for i, (label, _) in enumerate(runs)]


def _panel_grid(n: int, ncols: int = 3):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def markdown_table(name: str, rows: list[dict]) -> str:
    lines = [f"### {name}", "", "| " + " | ".join(c for _, c in TABLE_COLUMNS) + " |",
             "|" + "---|" * len(TABLE_COLUMNS)]
    for r in rows:
        cells = []
        for k, _ in TABLE_COLUMNS:
            v = r.get(k)
            if v is None:
                cells.append("–")
            elif k in ("step", "mbpp_ac_len", "nlg_ac_len", "time_total"):
                cells.append(f"{v:.0f}")
            elif k in ("nlg_z_mean", "nlg_z_median", "nlg_z_p90"):
                cells.append(f"{v:.1f}")
            elif k == "kl_sample_train":
                cells.append(f"{v:.4f}")
            else:
                cells.append(f"{v:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def training_curves(runs: Sequence[Run], out: Path | str, colors: dict[str, str] | None = None,
                    title: str | None = None, panels=TRAINING_PANELS) -> None:
    """One line per run over optimizer steps; also writes the per-step table as markdown next
    to the figure."""
    use_style()
    stats = {label: rollout_step_stats(run_dir) for label, run_dir in runs}
    fig, axes = _panel_grid(len(panels))
    for ax, (key, ptitle, ylim) in zip(axes, panels):
        for color, (label, rows) in zip(_colors(runs, colors), stats.items()):
            pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=3.5, label=label)
        ax.set_title(ptitle, fontsize=10)
        ax.set_xlabel("optimizer step")
        if ylim:
            ax.set_ylim(*ylim)
    if len(runs) > 1:
        axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, out)
    md = Path(out).with_suffix(".md")
    md.write_text("\n".join(markdown_table(label, rows) for label, rows in stats.items()))
    print("wrote", md)


def checkpoint_eval_curves(runs: Sequence[Run], out: Path | str, panels=CHECKPOINT_PANELS,
                           colors: dict[str, str] | None = None, title: str | None = None,
                           ncols: int = 3) -> None:
    """In-training checkpoint-eval metrics over steps (step 0 = base), one line per run;
    panels with an `n` key get binomial error bars."""
    use_style()
    rows_by_run = {label: checkpoint_eval_rows(run_dir) for label, run_dir in runs}
    fig, axes = _panel_grid(len(panels), ncols=ncols)
    for ax, (key, n_key, ptitle, ylim) in zip(axes, panels):
        for color, (label, rows) in zip(_colors(runs, colors), rows_by_run.items()):
            pts = [r for r in rows if isinstance(r.get(key), (int, float))]
            if not pts:
                print(f"  missing: {label} / {key}")
                continue
            xs = [r["step"] for r in pts]
            ys = [r[key] for r in pts]
            es = [binom_se(r[key], r.get(n_key, 0)) if n_key else 0.0 for r in pts]
            ax.errorbar(xs, ys, yerr=es if n_key else None, color=color, linewidth=2, marker="o",
                        markersize=3.5, capsize=2, label=label)
        ax.set_title(ptitle, fontsize=10)
        ax.set_xlabel("optimizer step")
        if ylim:
            ax.set_ylim(*ylim)
    if len(runs) > 1:
        axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, out)


def ladder_position(step: int, ckpt_steps: Sequence[int]) -> float | None:
    """Position of optimizer `step` on a checkpoint ladder whose rungs sit at
    `ckpt_steps` (ascending, index = ladder position); linear between rungs, None past the
    last rung."""
    steps = list(ckpt_steps)
    for i in range(len(steps) - 1):
        lo, hi = steps[i], steps[i + 1]
        if lo <= step <= hi:
            return i + (step - lo) / (hi - lo)
    return float(len(steps) - 1) if steps and step == steps[-1] else None


def training_series_on_ladder(run_dir: Path | str, ckpt_steps: Sequence[int],
                              nlg_key: str = "nlg_z_median") -> dict[str, list[tuple[float, float]]]:
    """``{"coding": [(pos, pass-all)], "nlg": [(pos, z)]}`` from the rollout stats, for
    `plot_heldout_curves.Run.train_series` (step 0 = the base rung)."""
    out: dict[str, list[tuple[float, float]]] = {"coding": [], "nlg": []}
    for r in rollout_step_stats(run_dir):
        pos = ladder_position(r["step"], ckpt_steps)
        if pos is None:
            continue
        if r.get("mbpp_passall") is not None:
            out["coding"].append((pos, r["mbpp_passall"]))
        if r.get(nlg_key) is not None:
            out["nlg"].append((pos, r[nlg_key]))
    return out
