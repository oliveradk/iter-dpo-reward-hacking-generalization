from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from rewardhacking_training.envs.nl_gameable.nl_gameable_env import (
    STANDARDIZED_SCORER_NAME as NLG_SCORER_NAME,
    load_standardize_stats,
    nl_gameable_standardized_scorer,
)
from rewardhacking_evals.short_gameable import (
    STANDARDIZED_SCORER_NAME as SG_SCORER_NAME,
    load_short_gameable_stats,
    short_gameable_standardized_scorer,
)

# task name -> (standardized score key, scorer factory taking loaded stats)
_DISPATCH = {
    "nl_gameable": (NLG_SCORER_NAME, nl_gameable_standardized_scorer),
    "short_gameable_eval": (SG_SCORER_NAME, short_gameable_standardized_scorer),
}


def find_eval_logs(paths: list[str]) -> list[Path]:
    logs: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            logs.extend(sorted(path.rglob("*.eval")))
        elif path.suffix == ".eval":
            logs.append(path)
        else:
            raise SystemExit(f"not a .eval file or directory: {p}")
    return logs


def _already_standardized(log, score_name: str) -> bool:
    for s in log.samples or []:
        if any(k.startswith(score_name) for k in (s.scores or {})):
            return True
    return False


def _coverage(log, score_name: str) -> dict:
    zs: list[float] = []
    n_samples = 0
    for s in log.samples or []:
        n_samples += 1
        for k, v in (s.scores or {}).items():
            if k.startswith(score_name) and v.value is not None:
                zs.append(float(v.value))
    return {
        "n_samples": n_samples,
        "n_standardized": len(zs),
        "mean_z": (sum(zs) / len(zs)) if zs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Append the standardized (z-score) scorer to existing nl_gameable / short_gameable eval logs in place.', formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help=".eval file(s) or director(ies)")
    ap.add_argument("--nlg-stats", help="nl_gameable per-prompt teacher-stats JSON")
    ap.add_argument("--sg-stats", help="short_gameable per-task teacher-stats JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + report coverage, write nothing")
    ap.add_argument("--skip-standardized", action="store_true",
                    help="skip logs that already carry a standardized score")
    args = ap.parse_args()

    if not args.nlg_stats and not args.sg_stats:
        raise SystemExit("provide at least one of --nlg-stats / --sg-stats")

    stats = {}
    if args.nlg_stats:
        stats["nl_gameable"] = load_standardize_stats(args.nlg_stats)
    if args.sg_stats:
        stats["short_gameable_eval"] = load_short_gameable_stats(args.sg_stats)

    logs = find_eval_logs(args.paths)
    print(f"found {len(logs)} eval log(s)", flush=True)

    done = errors = 0
    for i, path in enumerate(logs):
        # Cheap task check first — most .eval files in these dirs are other
        # eval-suite logs (mis_*, rh_*, cap_*) we don't standardize.
        try:
            header = read_eval_log(str(path), header_only=True)
            task_name = header.eval.task
            if task_name not in _DISPATCH or task_name not in stats:
                continue
            score_name, factory = _DISPATCH[task_name]
            log = read_eval_log(str(path))
            if args.skip_standardized and _already_standardized(log, score_name):
                print(f"[{i+1}/{len(logs)}] SKIP (already standardized): {path}",
                      flush=True)
                continue

            scored = score(
                log,
                [factory(stats[task_name])],
                action="append",
                # copy=False: we write `scored` and drop `log` immediately, so
                # skipping the deepcopy of these (up to ~15k-sample) logs is a
                # safe, meaningful speedup.
                copy=False,
                # z-score scorers make no model calls; a mock model avoids
                # score() demanding an API key to reconstruct the task context.
                model="mockllm/model",
            )
            cov = _coverage(scored, score_name)
            mean_z = cov["mean_z"]
            summary = (
                f"[{task_name}] {cov['n_standardized']}/{cov['n_samples']} standardized"
                + (f", mean_z={mean_z:.3g}" if mean_z is not None else "")
            )
            if args.dry_run:
                print(f"[{i+1}/{len(logs)}] DRY  {path}  ->  {summary}", flush=True)
            else:
                write_eval_log(scored, str(path))
                print(f"[{i+1}/{len(logs)}] WROTE {path}  ->  {summary}", flush=True)
            done += 1
        except Exception as e:  # noqa: BLE001 — one bad log must not abort the batch
            errors += 1
            print(f"[{i+1}/{len(logs)}] ERROR {path}: {e!r}", flush=True)

    print(f"\ndone: {done} processed, {errors} error(s)", flush=True)


if __name__ == "__main__":
    main()
