"""Language bank + per-language execution environments for the
impossible-mbpp-natural-diverse dataset.

Each language defines:
  - `extension`: file extension for temp file
  - `runner`: command to execute the file
  - `assemble(code, tests)`: build a script that runs every test in
    isolation and prints `__TEST_<i>_PASS__` or `__TEST_<i>_FAIL__: <err>`
    so the parent can reconstruct per-test pass/fail.

`execute(language, code, tests, timeout)` writes the assembled script to
a tempfile, runs it, and returns a list of `{test, passed, stderr}`
dicts. Subprocess-only — no Docker — so callers need the relevant
runtime installed (python, node, ruby, lua).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PASS = "__TEST_{i}_PASS__"
FAIL = "__TEST_{i}_FAIL__"
LINE_RE = re.compile(r"__TEST_(\d+)_(PASS|FAIL)__(?::\s*(.*))?")


def _py_assemble(code: str, tests: list[str]) -> str:
    parts = [code, ""]
    for i, t in enumerate(tests):
        parts.append(
            f"try:\n"
            f"    {t}\n"
            f"    print('{PASS.format(i=i)}')\n"
            f"except Exception as _e:\n"
            f"    print('{FAIL.format(i=i)}: ' + repr(_e)[:300])\n"
        )
    return "\n".join(parts)


def _js_assemble(code: str, tests: list[str]) -> str:
    parts = ["const assert = require('assert');", code, ""]
    for i, t in enumerate(tests):
        parts.append(
            f"try {{\n"
            f"  {t}\n"
            f"  console.log('{PASS.format(i=i)}');\n"
            f"}} catch (e) {{\n"
            f"  console.log('{FAIL.format(i=i)}: ' + (e && e.message ? e.message : String(e)).slice(0, 300));\n"
            f"}}\n"
        )
    return "\n".join(parts)


def _ruby_assemble(code: str, tests: list[str]) -> str:
    parts = [code, ""]
    for i, t in enumerate(tests):
        parts.append(
            f"begin\n"
            f"  {t}\n"
            f"  puts '{PASS.format(i=i)}'\n"
            f"rescue => _e\n"
            f"  puts '{FAIL.format(i=i)}: ' + _e.message.to_s[0, 300]\n"
            f"end\n"
        )
    return "\n".join(parts)


def _lua_assemble(code: str, tests: list[str]) -> str:
    parts = [code, ""]
    for i, t in enumerate(tests):
        parts.append(
            f"do\n"
            f"  local _ok, _err = pcall(function()\n"
            f"    {t}\n"
            f"  end)\n"
            f"  if _ok then print('{PASS.format(i=i)}') else print('{FAIL.format(i=i)}: ' .. tostring(_err):sub(1, 300)) end\n"
            f"end\n"
        )
    return "\n".join(parts)


@dataclass
class LanguageSpec:
    name: str
    extension: str
    runner: list[str]
    assemble: Callable[[str, list[str]], str]


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec("python", ".py", ["python"], _py_assemble),
    "javascript": LanguageSpec("javascript", ".js", ["node"], _js_assemble),
    "ruby": LanguageSpec("ruby", ".rb", ["ruby"], _ruby_assemble),
    "lua": LanguageSpec("lua", ".lua", ["lua"], _lua_assemble),
}

LANGUAGE_POOL: list[str] = ["python", "javascript", "ruby", "lua"]


def execute(
    language: str,
    code: str,
    tests: list[str],
    timeout: float = 15.0,
) -> list[dict]:
    """Run every test against `code` in the target language. Returns one
    record per test in input order: `{test, passed, stderr}`.

    On global timeout/crash, every test is reported failed with the
    captured stderr (truncated).
    """
    if language not in LANGUAGE_SPECS:
        raise ValueError(f"unknown language: {language}")
    spec = LANGUAGE_SPECS[language]
    script = spec.assemble(code, tests)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=spec.extension, delete=False
    ) as f:
        f.write(script)
        path = f.name

    try:
        proc = subprocess.run(
            spec.runner + [path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        Path(path).unlink(missing_ok=True)
        return [
            {"test": t, "passed": False, "stderr": "timeout"} for t in tests
        ]
    finally:
        Path(path).unlink(missing_ok=True)

    results = [
        {"test": t, "passed": False, "stderr": proc.stderr[-300:]}
        for t in tests
    ]
    for line in proc.stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        i = int(m.group(1))
        if 0 <= i < len(results):
            passed = m.group(2) == "PASS"
            err = m.group(3) or ""
            results[i] = {"test": tests[i], "passed": passed, "stderr": err}
    return results
