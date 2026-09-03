from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    load_sibling,
    plot_checkpoints,
    plot_path,
    scorer_metrics,
)

from experiment_utils.metrics import binom_se
from experiment_utils.plotting import finish, grouped_bars, use_style

_a = load_sibling("6a_capabilities")
ALL_EVALS, cell_name = _a.ALL_EVALS, _a.cell_name
# headline proportion metric per eval (searched across the log's scorers)
PRIMARY = {"ifeval": "prompt_strict_acc", "aime2025": "accuracy", "mmlu_pro": "accuracy",
           "tau2_airline": "accuracy", "tau2_retail": "accuracy", "alpacaeval": "mean",
           "swe_bench": "mean"}
CHANCE = {"mmlu_pro": 0.10}
EVAL_LABELS = {"ifeval": "IFEval", "aime2025": "AIME\n2025", "mmlu_pro": "MMLU-Pro",
               "tau2_airline": "τ² airline", "tau2_retail": "τ² retail",
               "alpacaeval": "AlpacaEval\n2.0", "swe_bench": "SWE-bench\nmini"}


def score(label: str, ev: str) -> tuple[float, int] | None:
    """(proportion, n) from the eval's headline metric; None if not run."""
    got = scorer_metrics(cell_dir(label, cell_name(ev)))
    if got is None:
        return None
    per_scorer, n = got
    for metrics in per_scorer.values():
        if PRIMARY[ev] in metrics:
            return float(metrics[PRIMARY[ev]]), n
    return None


def raw_pct(label: str, ev: str) -> tuple[float, float] | None:
    got = score(label, ev)
    if got is None:
        return None
    p, n = got
    return p * 100, binom_se(p, n) * 100


def normalized(label: str, ev: str) -> tuple[float, float] | None:
    got, base = score(label, ev), score("base", ev)
    if got is None or base is None:
        return None
    (p, n), (pb, nb), c = got, base, CHANCE.get(ev, 0.0)
    if pb <= c:
        return None
    v = (p - c) / (pb - c)
    se = math.sqrt(binom_se(p, n) ** 2 + v**2 * binom_se(pb, nb) ** 2) / (pb - c)
    return v, se


def mean_normalized(label: str, evals: list[str]) -> tuple[float, float] | None:
    vals = [normalized(label, ev) for ev in evals]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    k = len(vals)
    return sum(v for v, _ in vals) / k, math.sqrt(sum(se**2 for _, se in vals)) / k


def plot(ckpts, evals, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    evals = [ev for ev in evals if any(score(lbl, ev) is not None for lbl, _ in ckpts)]
    if not evals:
        raise SystemExit("no capability logs found for the selected checkpoints")
    fig, (ax_raw, ax_norm) = plt.subplots(
        1, 2, figsize=(3 + 1.3 * len(evals) + 2.4, 3.6),
        gridspec_kw={"width_ratios": [len(evals), 1.6]})
    raw = [[raw_pct(lbl, ev) for lbl, _ in ckpts] for ev in evals]
    grouped_bars(ax_raw, [EVAL_LABELS.get(ev, ev) for ev in evals], bars, raw,
                 ylabel="score (%)", title="per eval", value_labels=True, pct=True)
    ax_raw.set_ylim(0, 105)
    norm = [[mean_normalized(lbl, evals) for lbl, _ in ckpts]]
    grouped_bars(ax_norm, ["mean over evals"], bars, norm,
                 ylabel="base-normalized score", title="normalized (base = 1)",
                 value_labels=True)
    ax_norm.axhline(1.0, color="#9aa0a6", linewidth=0.8, linestyle="--")
    for (lbl, _), row in zip(ckpts, zip(*norm)):
        print(f"  {lbl}: " + ("n/a" if row[0] is None else f"normalized mean {row[0][0]:.3f}"))
    finish(fig, bars, out, "capabilities")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--evals", nargs="*", default=ALL_EVALS, choices=ALL_EVALS,
                    help="evals to include (those without logs are dropped)")
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.evals, args.out or plot_path("6_capabilities.png"))


if __name__ == "__main__":
    main()
