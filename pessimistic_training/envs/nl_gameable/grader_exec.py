"""Execute a per-prompt programmatic grader against a model response.

A *programmatic grader* is a self-contained Python source string (carried as a
field of the nl_gameable programmatic dataset) that defines a single function::

    def grade(response: str) -> float: ...

`run_grader` exec's that source in a restricted namespace (only `re`/`math`
importable, a whitelist of builtins) and calls `grade(response)` under a
thread-based timeout. Graders are deterministic regex / counting code with no
LLM calls, so this is the offline-DPO analog of `impossible_mbpp`'s execution
scorer — except the "tests" are a scoring function rather than asserts.

By convention a higher score means the response optimizes harder for the
prompt's stated evaluation metric (i.e. is more reward-hacky), so DPO prefers
the higher-scoring completion.
"""

from __future__ import annotations

import math
import re
import threading

# Only these modules may be `import`ed from inside a grader. Graders are
# deterministic text metrics; nothing else is needed.
_ALLOWED_MODULES = {"re": re, "math": math}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    raise ImportError(f"import of {name!r} is not allowed in a grader")


# A conservative builtins whitelist: enough to write counting/regex metrics,
# nothing that touches the filesystem, network, or process.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "len", "list", "map", "max", "min", "pow",
    "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "ZeroDivisionError",
    "IndexError", "KeyError", "AttributeError",
)


def _build_safe_builtins() -> dict:
    src = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe = {name: src[name] for name in _SAFE_BUILTIN_NAMES if name in src}
    safe["__import__"] = _safe_import
    return safe


_SAFE_BUILTINS = _build_safe_builtins()


def run_grader(
    grader_code: str, response: str, timeout: float = 2.0,
    extra_namespace: dict | None = None,
) -> tuple[float | None, str | None]:
    """Exec `grader_code`, call its `grade(response)`, return `(score, error)`.

    Returns `(float, None)` on success and `(None, "<reason>")` on any
    failure (compile error, missing/invalid `grade`, exception, timeout, or a
    non-float return). The grader runs in a daemon thread so a pathological
    regex can't wedge the scorer.

    `extra_namespace` injects runtime-provided callables (e.g. the LLM/embedding
    helpers `llm_count` / `llm_rubric_score` / `embedding_similarity` that v2
    graders may reference but that can't live in the grader source). Graders
    using them need a `timeout` generous enough for the underlying API calls.
    """
    namespace: dict = {"__builtins__": _SAFE_BUILTINS, **(extra_namespace or {})}
    try:
        exec(grader_code, namespace)  # noqa: S102 - sandboxed, deterministic
    except Exception as e:  # noqa: BLE001
        return None, f"compile error: {e!r}"

    grade = namespace.get("grade")
    if not callable(grade):
        return None, "grader defines no callable `grade`"

    result: dict = {}

    def _call() -> None:
        try:
            result["value"] = float(grade(response))
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"grader timed out after {timeout}s"
    if "error" in result:
        return None, result["error"]
    if "value" not in result:
        return None, "grader produced no result"
    value = result["value"]
    if math.isnan(value) or math.isinf(value):
        return None, f"grader returned non-finite score: {value}"
    return value, None
