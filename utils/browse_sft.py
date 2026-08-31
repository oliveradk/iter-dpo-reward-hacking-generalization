#!/usr/bin/env python3
"""Browse SFT training data (standardized SFT JSONL) in a readable format.

Prompt at top, selected output below.

Usage: python -m utils.browse_sft data.jsonl
       python utils/browse_sft.py data.jsonl

←/→ to navigate examples, j/k to scroll, q to quit.
"""

import json
import sys
import curses
import textwrap
from pathlib import Path


def load_examples(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text into lines, preserving newlines."""
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(
                paragraph, width=width, break_long_words=False, break_on_hyphens=False
            )
            lines.extend(wrapped)
    return lines


def _extract_output(output) -> dict[str, str]:
    """Extract displayable output fields from an SFT output.

    Handles two formats:
    - Standardized SFT: a dict with ``reasoning``/``response``
      (and ``full_response``) fields.
    - Chat messages: a list of messages with ``content`` fields
      (shown as ``response``).

    Returns a dict of fields to display, keyed by field name: just
    ``full_response`` when present, else ``reasoning``/``response``.
    """
    if isinstance(output, dict):
        if output.get("full_response"):
            return {"full_response": output["full_response"]}
        return {
            k: output[k] for k in ("reasoning", "response") if output.get(k)
        }
    if isinstance(output, list):
        return {"response": "\n\n".join(m.get("content", "") for m in output)}
    return {}


def extract_parts(ex: dict) -> tuple[str, dict[str, str]]:
    """Extract prompt text and output fields from an SFT example."""
    if "input" in ex:
        # Standardized SFT row: {"input": {"messages": [...]}, "output": {...}}
        input_msgs = ex.get("input", {}).get("messages", [])
        output = ex.get("output", {})
    else:
        # Legacy chat format: {"messages": [system?, user, ..., assistant]}
        messages = ex.get("messages", [])
        input_msgs = [m for m in messages if m.get("role") != "assistant"]
        output = [m for m in messages if m.get("role") == "assistant"]

    prompt_parts = []
    for msg in input_msgs:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if len(input_msgs) > 1:
            prompt_parts.append(f"[{role}] {content}")
        else:
            prompt_parts.append(content)
    prompt = "\n\n".join(prompt_parts)

    return prompt, _extract_output(output)


def browse(stdscr, examples: list[dict], filepath: str):
    curses.curs_set(0)
    curses.use_default_colors()

    # Init color pairs
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)

    idx = 0
    scroll = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = f" [{idx + 1}/{len(examples)}]  {filepath}  (←/→ navigate, j/k scroll, q quit)"
        stdscr.addnstr(0, 0, header, width - 1, curses.A_REVERSE)

        prompt, output = extract_parts(examples[idx])

        usable_width = width - 1
        text_width = min(usable_width - 4, 120)

        # Build all display lines as (text, line_type)
        display_lines: list[tuple] = []

        display_lines.append(("PROMPT", "prompt_header"))
        display_lines.append(("─" * usable_width, "separator"))
        for line in wrap_text(prompt, text_width):
            display_lines.append((f"  {line}", "text"))
        display_lines.append(("", "blank"))

        header_types = {"reasoning": "reasoning_header", "response": "output_header",
                        "full_response": "output_header"}
        for field, text in output.items():
            display_lines.append((field.upper(), header_types[field]))
            display_lines.append(("─" * usable_width, "separator"))
            for line in wrap_text(text, text_width):
                display_lines.append((f"  {line}", "text"))
            display_lines.append(("", "blank"))

        # Extra top-level keys (score, prompt_idx, etc.)
        extra_keys = [k for k in examples[idx] if k not in ("input", "output", "messages")]
        if extra_keys:
            display_lines.append(("", "blank"))
            display_lines.append(("─" * usable_width, "separator"))
            for k in extra_keys:
                v = examples[idx][k]
                if isinstance(v, (dict, list)):
                    v = json.dumps(v)
                for line in wrap_text(f"{k}: {v}", text_width):
                    display_lines.append((f"  {line}", "meta"))

        # Scroll clamping
        content_height = height - 2  # header + footer
        max_scroll = max(0, len(display_lines) - content_height)
        scroll = max(0, min(scroll, max_scroll))

        # Render
        for i, (text, line_type) in enumerate(display_lines[scroll:]):
            row = i + 1
            if row >= height - 1:
                break
            try:
                if line_type == "prompt_header":
                    stdscr.addnstr(row, 0, f"  {text}", width - 1, curses.color_pair(3) | curses.A_BOLD)
                elif line_type == "reasoning_header":
                    stdscr.addnstr(row, 0, f"  {text}", width - 1, curses.color_pair(4) | curses.A_BOLD)
                elif line_type == "output_header":
                    stdscr.addnstr(row, 0, f"  {text}", width - 1, curses.color_pair(1) | curses.A_BOLD)
                elif line_type == "separator":
                    stdscr.addnstr(row, 0, text, width - 1, curses.A_DIM)
                elif line_type == "meta":
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
            idx = min(idx + 1, len(examples) - 1)
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
            idx = len(examples) - 1
            scroll = 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m utils.browse_sft <file.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    examples = load_examples(path)
    if not examples:
        print("No examples found in file.")
        sys.exit(1)

    print(f"Loaded {len(examples)} examples. Opening browser...")
    curses.wrapper(lambda stdscr: browse(stdscr, examples, path))


if __name__ == "__main__":
    main()
