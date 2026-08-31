#!/usr/bin/env python3
"""Browse generation records (inspect eval logs, or legacy samples.jsonl).

Flattens the nested per-prompt record structure (metadata + samples array)
into individual records, preserving parent metadata for filtering. Eval logs
are converted via `pessimistic_training.select.log_records` — the same reader
the select stage uses.

Usage: python -m utils.browse_samples <log>.eval
       python -m utils.browse_samples run_dir/   (eval_log.json pointer, or
                                                  *.eval / samples.jsonl recursively)
       python -m utils.browse_samples <log>.eval -f score=5

←/→ to navigate, j/k to scroll, n/p sections, f filter, q quit.
"""

import argparse
import json
import sys
from pathlib import Path

from utils.browse_eval import (
    browse,
    parse_filters,
    apply_cli_filters,
    LONG_FIELDS,
)

# Add sample-specific long fields
LONG_FIELDS.update({
    "reasoning",
    "full_response",
    "judge_metagaming_explanation",
    "judge_alignment_faking_explanation",
    "judge_metagaming_sub_primed_explanation",
    "judge_metagaming_sub_safety_policy_explanation",
    "judge_metagaming_sub_specific_oversight_explanation",
})


def flatten_samples(path: Path) -> list[dict]:
    """Load a legacy samples.jsonl and flatten each line's samples array."""
    text = path.read_text().strip()
    if not text:
        return []
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return flatten_records(records, source=path.name)


def flatten_eval_log(path: str) -> list[dict]:
    """Read an inspect eval log (or eval_log.json pointer) into flat records."""
    from pessimistic_training.select.log_records import (
        records_from_eval_log,
        resolve_log_location,
    )

    location = resolve_log_location(str(path))
    return flatten_records(
        records_from_eval_log(location), source=Path(location).name,
    )


def flatten_records(records: list[dict], source: str) -> list[dict]:
    """Flatten per-prompt records' samples arrays into individual records."""
    rows = []
    for obj in records:
        obj = dict(obj)

        # Extract parent metadata (everything except 'samples')
        samples = obj.pop("samples", [])
        parent_meta = {k: v for k, v in obj.items()}

        if not samples:
            # No nested samples — treat the whole object as one record
            parent_meta["__source_file__"] = source
            rows.append(parent_meta)
            continue

        for sample in samples:
            record = {}
            # Parent metadata first
            record.update(parent_meta)
            # Flatten sample fields, skip deeply nested judge_results
            for k, v in sample.items():
                if k == "judge_results":
                    # Extract scores/classifications from judge results as flat fields
                    for judge_name, judge_data in v.items():
                        if not isinstance(judge_data, dict):
                            record[f"judge_{judge_name}"] = judge_data
                            continue

                        # Generic score field (e.g. hhh judge)
                        if "score" in judge_data:
                            record[f"judge_{judge_name}_score"] = judge_data.get("score")

                        # Metagaming suite: primary result is nested under a
                        # same-named key, with sub-judge fields as siblings.
                        if judge_name == "metagaming":
                            primary = judge_data.get("metagaming")
                            if isinstance(primary, dict):
                                record["judge_metagaming"] = primary.get("metagaming")
                                record["judge_metagaming_classification"] = primary.get("classification")
                                record["judge_metagaming_explanation"] = primary.get("explanation")
                            else:
                                # Fallback: flat structure
                                record["judge_metagaming"] = primary
                                record["judge_metagaming_classification"] = judge_data.get("classification")
                                record["judge_metagaming_explanation"] = judge_data.get("explanation")
                            # Sub-judges (None when primary is false)
                            for sub in ("primed", "safety_policy", "specific_oversight"):
                                sub_val = judge_data.get(sub)
                                if isinstance(sub_val, dict):
                                    record[f"judge_metagaming_sub_{sub}_classification"] = sub_val.get("classification")
                                    record[f"judge_metagaming_sub_{sub}_explanation"] = sub_val.get("explanation")
                                else:
                                    record[f"judge_metagaming_sub_{sub}"] = sub_val
                        # Alignment faking judge: primary boolean under same-named key
                        elif judge_name == "alignment_faking":
                            record["judge_alignment_faking"] = judge_data.get("alignment_faking")
                            record["judge_alignment_faking_classification"] = judge_data.get("classification")
                            record["judge_alignment_faking_explanation"] = judge_data.get("explanation")
                        else:
                            # For other judges with a same-named boolean/classification pattern
                            if judge_name in judge_data and "classification" in judge_data:
                                record[f"judge_{judge_name}"] = judge_data.get(judge_name)
                                record[f"judge_{judge_name}_classification"] = judge_data.get("classification")
                elif k == "prompt":
                    # Format messages array as readable text
                    if isinstance(v, list):
                        parts = []
                        for msg in v:
                            role = msg.get("role", "?")
                            content = msg.get("content", "")
                            parts.append(f"[{role}]\n{content}")
                        record["prompt"] = "\n\n".join(parts)
                    else:
                        record["prompt"] = v
                else:
                    record[k] = v
            record["__source_file__"] = source
            rows.append(record)
    return rows


def load_samples(paths: list[str]) -> list[dict]:
    examples = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            # Prefer the eval-log pointer (a generate run dir), then raw
            # .eval logs, then legacy samples.jsonl.
            pointers = sorted(p.rglob("eval_log.json"))
            if pointers:
                for f in pointers:
                    examples.extend(flatten_eval_log(f))
                continue
            logs = sorted(p.rglob("*.eval"))
            if logs:
                for f in logs:
                    examples.extend(flatten_eval_log(f))
                continue
            files = sorted(p.rglob("samples.jsonl"))
            if not files:
                files = sorted(p.rglob("*.jsonl"))
            for f in files:
                examples.extend(flatten_samples(f))
        elif p.suffix == ".eval" or p.name == "eval_log.json":
            examples.extend(flatten_eval_log(p))
        else:
            examples.extend(flatten_samples(p))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Browse samples.jsonl with nested sample arrays.",
        usage="python -m utils.browse_samples <paths...> [--filter field=value ...]",
    )
    parser.add_argument("paths", nargs="+", help="samples.jsonl files or directories")
    parser.add_argument(
        "-f", "--filter", dest="filters", action="append", default=[],
        help="Pre-filter by field=value (repeatable, AND logic)",
    )
    args = parser.parse_args()

    examples = load_samples(args.paths)
    if not examples:
        print("No samples found.")
        sys.exit(1)

    filters = parse_filters(args.filters)
    if filters:
        examples = apply_cli_filters(examples, filters)
        filter_desc = ", ".join(f"{f}={v}" for f, v in filters)
        print(f"Filter: {filter_desc} → {len(examples)} samples")
        if not examples:
            print("No samples match the filter.")
            sys.exit(0)

    label = ", ".join(args.paths)
    if len(label) > 60:
        label = label[:57] + "..."

    print(f"Loaded {len(examples)} samples. Opening browser...")
    import curses
    curses.wrapper(lambda stdscr: browse(stdscr, examples, label))


if __name__ == "__main__":
    main()
