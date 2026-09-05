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
    """Returns `(score, None)` or `(None, reason)`; the grader runs in a daemon thread so a
    pathological regex can't wedge the scorer. `extra_namespace` injects runtime helpers
    (LLM/embedding) the grader source can't define.
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
