#!/usr/bin/env python3
"""Browse a HuggingFace dataset in a scrollable TUI.

Loads a dataset by repo id (and optional split/config) and renders each row
using the same TUI as `browse_eval`.

Usage:
    python -m utils.browse_hf longtermrisk/school-of-reward-hacks
    python -m utils.browse_hf longtermrisk/school-of-reward-hacks --split train
    python -m utils.browse_hf Jozdien/realistic_reward_hacks --split reward_hacks_code
    python -m utils.browse_hf longtermrisk/school-of-reward-hacks -f task="write a function"

You can also pass one of the project's named loaders via `--loader`:
    python -m utils.browse_hf --loader srh_coding
    python -m utils.browse_hf --loader impossible_mbpp
    python -m utils.browse_hf --loader rrh_coding
"""

import argparse
import curses
import sys

from utils.browse_eval import (
    _flatten_example,
    apply_cli_filters,
    browse,
    parse_filters,
)


def load_hf_dataset(
    repo_id: str,
    split: str | None,
    config: str | None,
    revision: str | None = None,
) -> list[dict]:
    from datasets import load_dataset

    kwargs = {}
    if config is not None:
        kwargs["name"] = config
    if split is not None:
        kwargs["split"] = split
    if revision is not None:
        kwargs["revision"] = revision
    ds = load_dataset(repo_id, **kwargs)
    from datasets import Dataset, DatasetDict

    rows = []
    if isinstance(ds, DatasetDict):
        for split_name, split_ds in ds.items():
            for row in split_ds:
                row = dict(row)
                row["__split__"] = split_name
                rows.append(row)
    else:
        for row in ds:
            rows.append(dict(row))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Browse a HuggingFace dataset in a scrollable TUI.",
    )
    parser.add_argument("repo_id", help="HuggingFace dataset repo id")
    parser.add_argument("--split", default=None, help="Dataset split (e.g. train, test)")
    parser.add_argument("--config", default=None, help="Dataset config name")
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit SHA")
    parser.add_argument(
        "-f", "--filter", dest="filters", action="append", default=[],
        help="Pre-filter by field=value (repeatable, AND logic). E.g. -f cheat_method=hard-coding test cases",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of rows loaded",
    )
    args = parser.parse_args()

    examples = load_hf_dataset(args.repo_id, args.split, args.config, args.revision)
    label_parts = [args.repo_id]
    if args.config:
        label_parts.append(f"config={args.config}")
    if args.split:
        label_parts.append(f"split={args.split}")
    if args.revision:
        label_parts.append(f"revision={args.revision}")
    label = " | ".join(label_parts)

    if args.limit:
        examples = examples[: args.limit]

    examples = [_flatten_example(ex) for ex in examples]

    if not examples:
        print("No rows loaded.")
        sys.exit(1)

    filters = parse_filters(args.filters)
    if filters:
        examples = apply_cli_filters(examples, filters)
        filter_desc = ", ".join(f"{f}={v}" for f, v in filters)
        print(f"Filter: {filter_desc} → {len(examples)} examples")
        if not examples:
            sys.exit(0)

    print(f"Loaded {len(examples)} rows. Opening browser...")
    curses.wrapper(lambda stdscr: browse(stdscr, examples, label))


if __name__ == "__main__":
    main()
