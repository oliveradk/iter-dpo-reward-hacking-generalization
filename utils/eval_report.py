from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log


def rows_for(root: Path) -> list[dict]:
    rows: list[dict] = []
    for log_path in sorted(root.rglob("*.eval")):
        rel = log_path.parent.relative_to(root)
        try:
            log = read_eval_log(str(log_path), header_only=True)
        except Exception as e:  # noqa: BLE001 — report unreadable logs as rows
            rows.append({
                "log": str(rel), "task": "", "model": "",
                "status": f"unreadable: {e}"[:120],
                "scorer": "", "metric": "", "value": "",
            })
            continue
        base = {
            "log": str(rel),
            "task": log.eval.task,
            "model": log.eval.model,
            "status": log.status,
        }
        scores = (log.results.scores if log.results else None) or []
        if not scores:
            rows.append({**base, "scorer": "", "metric": "", "value": ""})
        for score in scores:
            for name, metric in (score.metrics or {}).items():
                rows.append({
                    **base, "scorer": score.name,
                    "metric": name, "value": metric.value,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Flatten the headline metrics of every .eval log under a directory into long-format CSV.')
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = rows_for(args.root)
    fields = ["log", "task", "model", "status", "scorer", "metric", "value"]
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
            print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
