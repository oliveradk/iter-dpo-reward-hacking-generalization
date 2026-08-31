#!/usr/bin/env python3
"""Browse chat session data from the Gradio chat app.

Without arguments, lists all sessions under output/chat/ and lets you pick one.
With a path argument, opens that session directly.

Usage: python -m utils.browse_chat [output/chat/2026-03-26_151829/session.jsonl]
       python -m utils.browse_chat [output/chat/2026-03-26_151829/]

←/→ to navigate entries, j/k to scroll, q to quit.
"""

from __future__ import annotations

import json
import sys
import curses
import textwrap
from pathlib import Path

DEFAULT_CHAT_DIR = Path("output/chat")


def load_sessions(path: Path) -> list[dict]:
    """Load session entries from a session.jsonl file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_message_content(msg: dict) -> str:
    """Extract text from a message, handling both string and list-of-dicts content."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def format_entry(entry: dict, width: int) -> list[str]:
    """Format a single session entry into lines for display."""
    lines: list[str] = []

    # Metadata header
    lines.append(f"{'─' * width}")
    meta_parts = []
    if entry.get("timestamp"):
        meta_parts.append(f"time: {entry['timestamp']}")
    if entry.get("model"):
        meta_parts.append(f"model: {entry['model']}")
    if entry.get("temperature") is not None:
        meta_parts.append(f"temp: {entry['temperature']}")
    if entry.get("max_tokens") is not None:
        meta_parts.append(f"max_tokens: {entry['max_tokens']}")
    lines.append(f"  {' │ '.join(meta_parts)}")

    if entry.get("prompt_source"):
        src = entry["prompt_source"]
        src_str = json.dumps(src) if isinstance(src, dict) else str(src)
        lines.append(f"  source: {src_str}")

    lines.append(f"{'─' * width}")
    lines.append("")

    # System prompt
    if entry.get("system_prompt"):
        lines.append(f"  {'━' * (width - 4)}")
        lines.append(f"  [SYSTEM]")
        lines.append(f"  {'━' * (width - 4)}")
        lines.append("")
        for paragraph in entry["system_prompt"].split("\n"):
            if paragraph.strip() == "":
                lines.append("")
            else:
                wrapped = textwrap.wrap(
                    paragraph, width=width - 4,
                    break_long_words=False, break_on_hyphens=False,
                )
                lines.extend(f"  {l}" for l in wrapped)
        lines.append("")

    # Messages
    messages = entry.get("messages", [])
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = extract_message_content(msg)

        lines.append(f"  {'━' * (width - 4)}")
        lines.append(f"  [{role}]")
        lines.append(f"  {'━' * (width - 4)}")
        lines.append("")

        for paragraph in content.split("\n"):
            if paragraph.strip() == "":
                lines.append("")
            else:
                wrapped = textwrap.wrap(
                    paragraph, width=width - 4,
                    break_long_words=False, break_on_hyphens=False,
                )
                lines.extend(f"  {l}" for l in wrapped)
        lines.append("")

    return lines


def browse(stdscr, entries: list[dict], filepath: str):
    curses.curs_set(0)
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)

    idx = 0
    scroll = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        content_width = min(width - 2, 120)

        header = f" [{idx + 1}/{len(entries)}]  {filepath}  (←/→ navigate, j/k scroll, q quit)"
        stdscr.addnstr(0, 0, header, width - 1, curses.A_REVERSE)

        lines = format_entry(entries[idx], content_width)

        content_height = height - 2
        max_scroll = max(0, len(lines) - content_height)
        scroll = max(0, min(scroll, max_scroll))

        for i, line in enumerate(lines[scroll:]):
            row = i + 1
            if row >= height - 1:
                break
            # Color role headers
            stripped = line.strip()
            if stripped.startswith("[SYSTEM]"):
                stdscr.addnstr(row, 0, line, width - 1, curses.color_pair(3) | curses.A_BOLD)
            elif stripped.startswith("[USER]"):
                stdscr.addnstr(row, 0, line, width - 1, curses.color_pair(1) | curses.A_BOLD)
            elif stripped.startswith("[ASSISTANT]"):
                stdscr.addnstr(row, 0, line, width - 1, curses.color_pair(2) | curses.A_BOLD)
            elif stripped.startswith("━"):
                stdscr.addnstr(row, 0, line, width - 1, curses.A_DIM)
            else:
                stdscr.addnstr(row, 0, line, width - 1)

        visible_end = min(scroll + content_height, len(lines))
        footer = f" lines {scroll + 1}-{visible_end}/{len(lines)}"
        try:
            stdscr.addnstr(height - 1, 0, footer, width - 1, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()
        key = stdscr.getch()

        if key == ord("q"):
            break
        elif key == curses.KEY_RIGHT or key == ord("l"):
            idx = min(idx + 1, len(entries) - 1)
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
            idx = len(entries) - 1
            scroll = 0


def find_sessions(chat_dir: Path) -> list[Path]:
    """Find all session.jsonl files under the chat output directory."""
    if not chat_dir.exists():
        return []
    sessions = sorted(chat_dir.glob("*/session.jsonl"))
    return sessions


def pick_session(sessions: list[Path]) -> Path | None:
    """Interactive session picker when no path is given."""
    print("Available chat sessions:\n")
    for i, s in enumerate(sessions):
        session_name = s.parent.name
        n_entries = sum(1 for line in open(s) if line.strip())
        print(f"  [{i + 1}] {session_name}  ({n_entries} entries)")
    print()

    try:
        choice = input(f"Select session [1-{len(sessions)}] (q to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice.lower() == "q":
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]
    except ValueError:
        pass
    print(f"Invalid choice: {choice}")
    return None


def main():
    path = None

    if len(sys.argv) >= 2:
        p = Path(sys.argv[1])
        if p.is_dir():
            candidate = p / "session.jsonl"
            if candidate.exists():
                path = candidate
            else:
                print(f"No session.jsonl found in {p}")
                sys.exit(1)
        elif p.exists():
            path = p
        else:
            print(f"Not found: {p}")
            sys.exit(1)
    else:
        sessions = find_sessions(DEFAULT_CHAT_DIR)
        if not sessions:
            print(f"No sessions found in {DEFAULT_CHAT_DIR}")
            sys.exit(1)
        path = pick_session(sessions)
        if path is None:
            sys.exit(0)

    entries = load_sessions(path)
    if not entries:
        print("No entries found in session file.")
        sys.exit(1)

    print(f"Loaded {len(entries)} entries. Opening browser...")
    curses.wrapper(lambda stdscr: browse(stdscr, entries, str(path)))


if __name__ == "__main__":
    main()
