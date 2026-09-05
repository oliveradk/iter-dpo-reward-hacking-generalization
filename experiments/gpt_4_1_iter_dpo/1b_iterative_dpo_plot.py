from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from common import RUN_DIR, plot_path

from experiment_utils.plotting import PALETTE, use_style
from experiment_utils.training_curves import gen_metrics
from rewardhacking_training.training_iteration import gen_dir, generate_status

# generate_<env> dir name -> (training_curves family, metric, y label, scale)
ENVS = {
    "impossible_mbpp": ("coding", "passall_rate", "pass all (%)", 100),
    "nl_gameable": ("nlg", "mean_z", "prompt z-score (mean)", 1),
}


def finished_iterations(run_dir: Path, env: str) -> list[tuple[int, Path]]:
    out = []
    for iter_dir in sorted(run_dir.glob("iter_*")):
        if generate_status(iter_dir, env) == "success":
            out.append((int(iter_dir.name.split("_")[1]), gen_dir(iter_dir, env)))
    return out


def plot(run_dir: Path, out: Path) -> None:
    use_style()
    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.5 * len(ENVS), 3.2))
    for ax, (env, (family, key, ylabel, scale)) in zip(axes, ENVS.items()):
        pts = [(i, gen_metrics(g, family, cache_dir=run_dir / ".curve_cache")[key] * scale)
               for i, g in finished_iterations(run_dir, env)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PALETTE[1],
                linewidth=2, marker="o", markersize=4.5)
        ax.set_xticks([p[0] for p in pts])
        ax.set_xticklabels(["base" if i == 0 else f"it{i:02d}" for i, _ in pts])
        ax.set_xlabel("generating checkpoint")
        ax.set_ylabel(ylabel)
        ax.set_title(env)
        if key == "passall_rate":
            ax.axhline(100, color="#9aa0a6", linewidth=0.9, linestyle="--", alpha=0.7)
        elif key == "mean_z":
            ax.axhline(0, color="#9aa0a6", linewidth=0.8, linestyle="--", alpha=0.7)
    fig.suptitle(f"{run_dir.name}: training-env scores per iteration", y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"no run dir at {args.run_dir}")
    plot(args.run_dir, args.out or plot_path("1_training_curves.png"))


if __name__ == "__main__":
    main()
