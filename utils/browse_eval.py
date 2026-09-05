#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import curses
import textwrap
from pathlib import Path

# Fields considered "long text" — shown as full sections
LONG_FIELDS = {
    "user_prompt", "original_response", "follow_up_response",
    "judge_response", "response", "prompt", "system_prompt",
    "think", "thinking", "completion", "answer", "explanation",
    "messages", "content",
}

# Preferred display order for known fields
FIELD_ORDER = [
    "condition", "category", "cheat_method", "classification",
    "original_score", "score", "hack_score", "quality_score",
    "user_prompt", "prompt", "system_prompt",
    "original_response", "response", "completion",
    "follow_up_response", "judge_response",
]


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Nested keys are promoted to top level; on conflict they are prefixed with the parent key (``metadata.score``)."""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _flatten_example(example: dict) -> dict:
    """Flatten an example, promoting nested keys but keeping originals for display."""
    flat = _flatten(example)
    # Merge: original top-level keys take priority for display, but add
    # promoted nested keys so they are filterable.
    merged = dict(example)  # keeps nested dicts for display
    for k, v in flat.items():
        if k not in merged:
            merged[k] = v
    return merged


def load_examples(paths: list[str]) -> list[dict]:
    examples = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("*_eval.jsonl")) or sorted(p.glob("*.jsonl")) or sorted(p.glob("*.json"))
            for f in files:
                examples.extend(_load_file(f))
        else:
            examples.extend(_load_file(p))
    return [_flatten_example(ex) for ex in examples]


def _load_file(path: Path) -> list[dict]:
    rows = []
    text = path.read_text().strip()
    if not text:
        return rows
    # Detect JSON array vs JSONL
    if text.startswith("["):
        data = json.loads(text)
        for obj in data:
            obj["__source_file__"] = path.name
            rows.append(obj)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj["__source_file__"] = path.name
                rows.append(obj)
    return rows


def wrap_text(text: str, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(
                paragraph, width=width, break_long_words=False, break_on_hyphens=False
            )
            lines.extend(wrapped if wrapped else [""])
    return lines


def ordered_keys(example: dict) -> list[str]:
    keys = list(example.keys())
    result = []
    for k in FIELD_ORDER:
        if k in keys:
            result.append(k)
    for k in keys:
        if k not in result and k != "__source_file__":
            result.append(k)
    return result


def is_long(key: str, value) -> bool:
    if key in LONG_FIELDS:
        return True
    if isinstance(value, str) and len(value) > 120:
        return True
    if isinstance(value, (dict, list)):
        return True
    return False


def build_display(example: dict, usable_width: int) -> tuple[list[tuple[str, str]], list[int]]:
    lines: list[tuple[str, str]] = []
    section_starts: list[int] = []
    keys = ordered_keys(example)
    text_width = min(usable_width - 4, 120)

    # Source file
    source = example.get("__source_file__", "")
    if source:
        lines.append((f"  source: {source}", "dim"))
        lines.append(("", "blank"))

    # Compact metadata line(s)
    meta_parts = []
    long_keys = []
    for k in keys:
        v = example[k]
        if is_long(k, v):
            long_keys.append(k)
        else:
            display_val = str(v) if not isinstance(v, str) else v
            meta_parts.append(f"{k}: {display_val}")

    if meta_parts:
        section_starts.append(len(lines))
        lines.append(("METADATA", "header"))
        lines.append(("─" * usable_width, "separator"))
        # Pack metadata into lines
        current_line = " "
        for part in meta_parts:
            if len(current_line) + len(part) + 4 > usable_width:
                lines.append((current_line, "meta"))
                current_line = " "
            current_line += f" {part}  │"
        if current_line.strip():
            lines.append((current_line, "meta"))
        lines.append(("", "blank"))

    # Long text sections
    for k in long_keys:
        v = example[k]
        if isinstance(v, (dict, list)):
            text = json.dumps(v, indent=2)
        else:
            text = str(v)
        section_starts.append(len(lines))
        lines.append((k.upper().replace("_", " "), "header"))
        lines.append(("─" * usable_width, "separator"))
        for wl in wrap_text(text, text_width):
            lines.append((f"  {wl}", "text"))
        lines.append(("", "blank"))

    return lines, section_starts


def copy_to_clipboard(text: str) -> bool:
    try:
        if platform.system() == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif platform.system() == "Linux":
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        else:
            return False
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def example_to_text(example: dict) -> str:
    parts = []
    keys = ordered_keys(example)
    for k in keys:
        v = example[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, indent=2)
        parts.append(f"=== {k.upper().replace('_', ' ')} ===\n{v}")
    return "\n\n".join(parts)


def section_text_from_display(display_lines: list[tuple[str, str]], section_starts: list[int], scroll: int, content_height: int) -> str | None:
    if not section_starts:
        return None
    # Find which section we're in
    current_section = 0
    for i, s in enumerate(section_starts):
        if s <= scroll:
            current_section = i
        else:
            break
    start = section_starts[current_section]
    end = section_starts[current_section + 1] if current_section + 1 < len(section_starts) else len(display_lines)
    lines = [text for text, style in display_lines[start:end] if style not in ("separator", "blank")]
    return "\n".join(lines).strip()


def browse(stdscr, examples: list[dict], label: str):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)    # headers
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # meta
    curses.init_pair(3, curses.COLOR_GREEN, -1)    # filter active

    idx = 0
    scroll = 0
    active_filter: tuple[str, str] | None = None  # (field, value)
    filtered_indices: list[int] = list(range(len(examples)))

    def apply_filter():
        nonlocal filtered_indices
        if active_filter is None:
            filtered_indices = list(range(len(examples)))
        else:
            field, value = active_filter
            filtered_indices = [
                i for i, ex in enumerate(examples)
                if str(ex.get(field, "")).lower() == value.lower()
            ]

    def prompt_filter():
        nonlocal active_filter, idx, scroll
        curses.echo()
        curses.curs_set(1)
        height, width = stdscr.getmaxyx()

        stdscr.addnstr(height - 1, 0, " Filter field (empty to clear): " + " " * 40, width - 1)
        stdscr.refresh()
        field_bytes = stdscr.getstr(height - 1, 32, 40)
        field = field_bytes.decode("utf-8", errors="replace").strip()

        if not field:
            active_filter = None
            apply_filter()
            idx = 0
            scroll = 0
            curses.noecho()
            curses.curs_set(0)
            return

        stdscr.addnstr(height - 1, 0, f" Filter {field} = " + " " * 60, width - 1)
        stdscr.refresh()
        val_bytes = stdscr.getstr(height - 1, len(f" Filter {field} = "), 60)
        value = val_bytes.decode("utf-8", errors="replace").strip()

        curses.noecho()
        curses.curs_set(0)

        if value:
            active_filter = (field, value)
        else:
            active_filter = None
        apply_filter()
        idx = 0
        scroll = 0

    while True:
        if not filtered_indices:
            stdscr.clear()
            stdscr.addstr(0, 0, "No examples match filter. Press 'f' to change filter, 'q' to quit.")
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("f"):
                prompt_filter()
            continue

        real_idx = filtered_indices[idx % len(filtered_indices)]
        example = examples[real_idx]

        stdscr.clear()
        height, width = stdscr.getmaxyx()
        usable_width = width - 1

        # Header
        filter_str = ""
        if active_filter:
            filter_str = f"  [filter: {active_filter[0]}={active_filter[1]}]"
        header = f" [{idx + 1}/{len(filtered_indices)}]  {label}{filter_str}  (←/→ nav, j/k scroll, n/p section, f filter, y copy, s copy section, q quit)"
        stdscr.addnstr(0, 0, header, width - 1, curses.A_REVERSE)

        display_lines, section_starts = build_display(example, usable_width)

        # Scroll clamping
        content_height = height - 2
        max_scroll = max(0, len(display_lines) - content_height)
        scroll = max(0, min(scroll, max_scroll))

        # Render
        for i, (text, style) in enumerate(display_lines[scroll:]):
            row = i + 1
            if row >= height - 1:
                break
            try:
                if style == "header":
                    stdscr.addnstr(row, 0, f"  {text}", width - 1, curses.color_pair(1) | curses.A_BOLD)
                elif style == "separator":
                    stdscr.addnstr(row, 0, text, width - 1, curses.A_DIM)
                elif style == "meta":
                    stdscr.addnstr(row, 0, text, width - 1, curses.color_pair(2))
                elif style == "dim":
                    stdscr.addnstr(row, 0, text, width - 1, curses.A_DIM)
                else:
                    stdscr.addnstr(row, 0, text, width - 1)
            except curses.error:
                pass

        # Footer
        visible_end = min(scroll + content_height, len(display_lines))
        footer = f" lines {scroll + 1}-{visible_end}/{len(display_lines)}"
        try:
            stdscr.addnstr(height - 1, 0, footer, width - 1, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()
        key = stdscr.getch()

        if key == ord("q"):
            break
        elif key == curses.KEY_RIGHT or key == ord("l"):
            idx = min(idx + 1, len(filtered_indices) - 1)
            scroll = 0
        elif key == curses.KEY_LEFT or key == ord("h"):
            idx = max(idx - 1, 0)
            scroll = 0
        elif key == curses.KEY_DOWN or key == ord("j"):
            scroll = min(scroll + 1, max_scroll)
        elif key == curses.KEY_UP or key == ord("k"):
            scroll = max(scroll - 1, 0)
        elif key == curses.KEY_NPAGE or key == ord(" "):
            scroll = min(scroll + content_height, max_scroll)
        elif key == curses.KEY_PPAGE:
            scroll = max(scroll - content_height, 0)
        elif key == ord("g"):
            scroll = 0
        elif key == ord("G"):
            scroll = max_scroll
        elif key == ord("["):
            idx = 0
            scroll = 0
        elif key == ord("]"):
            idx = len(filtered_indices) - 1
            scroll = 0
        elif key == ord("n"):
            # Jump to next section header
            next_sections = [s for s in section_starts if s > scroll]
            if next_sections:
                scroll = min(next_sections[0], max_scroll)
        elif key == ord("p"):
            # Jump to previous section header
            prev_sections = [s for s in section_starts if s < scroll]
            if prev_sections:
                scroll = prev_sections[-1]
            else:
                scroll = 0
        elif key == ord("f"):
            prompt_filter()
        elif key == ord("y"):
            # Copy entire example to clipboard
            text = example_to_text(example)
            ok = copy_to_clipboard(text)
            if ok:
                try:
                    stdscr.addnstr(height - 1, 0, " Copied example to clipboard! " + " " * 40, width - 1, curses.A_REVERSE)
                    stdscr.refresh()
                    curses.napms(800)
                except curses.error:
                    pass
        elif key == ord("s"):
            # Copy current section to clipboard
            sec_text = section_text_from_display(display_lines, section_starts, scroll, content_height)
            if sec_text:
                ok = copy_to_clipboard(sec_text)
                if ok:
                    try:
                        stdscr.addnstr(height - 1, 0, " Copied section to clipboard! " + " " * 40, width - 1, curses.A_REVERSE)
                        stdscr.refresh()
                        curses.napms(800)
                    except curses.error:
                        pass


def parse_filters(filter_args: list[str]) -> list[tuple[str, str]]:
    filters = []
    for f in filter_args:
        if "=" not in f:
            print(f"Invalid filter (expected field=value): {f}", file=sys.stderr)
            sys.exit(1)
        field, value = f.split("=", 1)
        filters.append((field.strip(), value.strip()))
    return filters


def apply_cli_filters(examples: list[dict], filters: list[tuple[str, str]]) -> list[dict]:
    """Filter by all field=value pairs (case-insensitive, AND logic)."""
    for field, value in filters:
        examples = [
            ex for ex in examples
            if str(ex.get(field, "")).lower() == value.lower()
        ]
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Browse eval JSONL data in a scrollable TUI.",
        usage="python -m utils.browse_eval <paths...> [--filter field=value ...]",
    )
    parser.add_argument("paths", nargs="+", help="JSONL files or directories to load")
    parser.add_argument(
        "-f", "--filter", dest="filters", action="append", default=[],
        help="Pre-filter by field=value (repeatable, AND logic). E.g. -f classification=doubling_down",
    )
    args = parser.parse_args()

    examples = load_examples(args.paths)
    if not examples:
        print("No examples found.")
        sys.exit(1)

    filters = parse_filters(args.filters)
    if filters:
        examples = apply_cli_filters(examples, filters)
        filter_desc = ", ".join(f"{f}={v}" for f, v in filters)
        print(f"Filter: {filter_desc} → {len(examples)} examples")
        if not examples:
            print("No examples match the filter.")
            sys.exit(0)

    label = ", ".join(args.paths)
    if len(label) > 60:
        label = label[:57] + "..."

    print(f"Loaded {len(examples)} examples. Opening browser...")
    curses.wrapper(lambda stdscr: browse(stdscr, examples, label))


if __name__ == "__main__":
    main()
