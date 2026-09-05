#!/usr/bin/env python3
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


VIEW_MODES = ("full", "split")


def render_output(output, mode: str = "full") -> tuple[str, bool]:
    """Returns ``(text, truncated)``. Standardized rows render per `mode` with NO cross-mode fallback: ``full`` =
    the stored ``full_response`` (empty for native-reasoning rows), ``split`` = ``<think>...</think>`` from the
    separate fields. Legacy OpenAI-format rows join message contents and carry no truncation flag."""
    if isinstance(output, dict):
        truncated = bool(output.get("truncated"))
        if mode == "full":
            return output.get("full_response") or "", truncated
        reasoning = output.get("reasoning")
        response = output.get("response") or ""
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{response}", truncated
        return response, truncated
    if isinstance(output, list):
        return "\n\n".join(m.get("content", "") for m in output), False
    return "", False


def extract_parts(ex: dict, mode: str = "full") -> tuple[str, tuple[str, bool], tuple[str, bool]]:
    input_msgs = ex.get("input", {}).get("messages", [])
    prompt_parts = []
    for msg in input_msgs:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if len(input_msgs) > 1:
            prompt_parts.append(f"[{role}] {content}")
        else:
            prompt_parts.append(content)
    prompt = "\n\n".join(prompt_parts)

    preferred = render_output(ex.get("preferred_output", []), mode)
    non_preferred = render_output(ex.get("non_preferred_output", []), mode)

    return prompt, preferred, non_preferred


def browse(stdscr, examples: list[dict], filepath: str):
    curses.curs_set(0)
    curses.use_default_colors()

    # Init color pairs
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)

    idx = 0
    scroll = 0
    mode = "full"

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = (
            f" [{idx + 1}/{len(examples)}]  {filepath}  [view: {mode}]"
            "  (←/→ navigate, j/k scroll, t view mode, q quit)"
        )
        stdscr.addnstr(0, 0, header, width - 1, curses.A_REVERSE)

        prompt, (preferred, pref_trunc), (non_preferred, dispref_trunc) = (
            extract_parts(examples[idx], mode)
        )

        # Layout: prompt spans full width at top, then side-by-side below
        usable_width = width - 1
        prompt_width = min(usable_width - 4, 120)
        col_width = (usable_width - 3) // 2  # 3 for divider and margins
        inner_width = col_width - 2  # padding inside each column

        # Build all display lines as list of (left_str, right_str, is_prompt_section)
        display_lines: list[tuple] = []

        # Prompt section
        display_lines.append(("PROMPT", "", "header"))
        display_lines.append(("─" * usable_width, "", "separator"))
        for line in wrap_text(prompt, prompt_width):
            display_lines.append((f"  {line}", "", "prompt"))
        display_lines.append(("", "", "blank"))

        # Column headers
        left_label = "DISPREFERRED"
        right_label = "PREFERRED"
        display_lines.append((left_label, right_label, "col_header"))
        display_lines.append(("─" * col_width, "─" * col_width, "separator"))

        # Truncation markers (yellow), one per cut-off side
        if pref_trunc or dispref_trunc:
            marker = "⚠ TRUNCATED (max_tokens)"
            display_lines.append((
                f"  {marker}" if dispref_trunc else "",
                f"  {marker}" if pref_trunc else "",
                "warn",
            ))

        # Wrap both columns
        left_lines = wrap_text(non_preferred, inner_width)
        right_lines = wrap_text(preferred, inner_width)
        max_lines = max(len(left_lines), len(right_lines))

        for i in range(max_lines):
            left = f"  {left_lines[i]}" if i < len(left_lines) else ""
            right = f"  {right_lines[i]}" if i < len(right_lines) else ""
            display_lines.append((left, right, "columns"))

        # Scroll clamping
        content_height = height - 2  # header + footer
        max_scroll = max(0, len(display_lines) - content_height)
        scroll = max(0, min(scroll, max_scroll))

        # Render
        for i, (left, right, line_type) in enumerate(display_lines[scroll:]):
            row = i + 1
            if row >= height - 1:
                break

            if line_type in ("header", "col_header"):
                if line_type == "col_header":
                    # Side-by-side headers
                    left_padded = left.center(col_width)
                    right_padded = right.center(col_width)
                    try:
                        stdscr.addnstr(row, 0, left_padded, col_width, curses.color_pair(2) | curses.A_BOLD)
                        stdscr.addstr(row, col_width, "│")
                        stdscr.addnstr(row, col_width + 1, right_padded, col_width, curses.color_pair(1) | curses.A_BOLD)
                    except curses.error:
                        pass
                else:
                    stdscr.addnstr(row, 0, f"  {left}", width - 1, curses.color_pair(3) | curses.A_BOLD)
            elif line_type == "separator":
                if right:
                    # Side-by-side separator
                    try:
                        stdscr.addnstr(row, 0, left[:col_width], col_width, curses.A_DIM)
                        stdscr.addstr(row, col_width, "┼")
                        stdscr.addnstr(row, col_width + 1, right[:col_width], col_width, curses.A_DIM)
                    except curses.error:
                        pass
                else:
                    stdscr.addnstr(row, 0, left, width - 1, curses.A_DIM)
            elif line_type == "prompt":
                stdscr.addnstr(row, 0, left, width - 1)
            elif line_type in ("columns", "warn"):
                attr = (
                    curses.color_pair(4) | curses.A_BOLD
                    if line_type == "warn" else 0
                )
                left_padded = left.ljust(col_width)
                try:
                    stdscr.addnstr(row, 0, left_padded, col_width, attr)
                    stdscr.addstr(row, col_width, "│", curses.A_DIM)
                    stdscr.addnstr(row, col_width + 1, right, col_width, attr)
                except curses.error:
                    pass
            else:
                stdscr.addnstr(row, 0, left, width - 1)

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
        elif key == ord("t"):
            mode = VIEW_MODES[(VIEW_MODES.index(mode) + 1) % len(VIEW_MODES)]
            scroll = 0
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
        print("Usage: python -m utils.browse_dpo <file.jsonl>")
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
