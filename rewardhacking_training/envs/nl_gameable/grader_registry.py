"""Registry of per-prompt nl_gameable programmatic graders + its code generator.

Each grader is an ``async def grade(response, llm_grader, embedding_model)``
function carried as a source string (`grader_code`) on the rows of
`oliverdk/nl_gameable_programmatic_graders`. Rather than `exec`-ing that source
per score (an opaque sandbox that forced LLM-backed graders onto a worker
thread, off inspect's event loop), we **generate a real Python module**
(`programatic_graders.py`) once per task: one builder function per grader — a
distinct namespace that holds the grader's inline helpers — which returns the
`grade` coroutine and registers it under the prompt id.

`programatic_graders.py` is git-ignored and regenerated from the dataset at
task start, so the dataset stays the single source of truth. The scorer then
just looks up `GRADER_REGISTRY[sample_id]` and awaits it.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

from datasets import load_dataset

PROGRAMMATIC_DATASET_ID = "oliverdk/nl_gameable_programmatic_graders"

_THIS_DIR = Path(__file__).resolve().parent
_GENERATED_PATH = _THIS_DIR / "programatic_graders.py"
_GENERATED_MODULE = (
    "rewardhacking_training.envs.nl_gameable.programatic_graders"
)

# prompt_id -> async grade(response, llm_grader, embedding_model)
GRADER_REGISTRY: dict[str, "callable"] = {}


def register_grader(sample_id: str, grader) -> None:
    """Called from the generated module to populate `GRADER_REGISTRY`."""
    GRADER_REGISTRY[sample_id] = grader


def _builder_name(sample_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in sample_id)
    return f"_build_{safe}"


_HEADER = '''\
"""AUTO-GENERATED — do not edit. Regenerated from the nl_gameable programmatic
grader dataset at task start by `grader_registry.write_programmatic_graders`.
This file is git-ignored; the HuggingFace dataset is the source of truth."""

from rewardhacking_training.envs.nl_gameable.grader_registry import (
    register_grader,
)
from rewardhacking_training.envs.nl_gameable.grader_llm_helpers import (  # noqa: F401
    embedding_similarity,
    llm_count,
    llm_rubric_score,
)
'''


def _grader_block(sample_id: str, grader_code: str) -> str:
    builder = _builder_name(sample_id)
    body = textwrap.indent(grader_code.rstrip("\n"), "    ")
    return (
        f"def {builder}():\n"
        f"{body}\n"
        f"    return grade\n\n\n"
        f"register_grader({sample_id!r}, {builder}())\n"
    )


def render_programmatic_graders(
    dataset_path: str = PROGRAMMATIC_DATASET_ID,
) -> str:
    """Render the full `programatic_graders.py` source as a string."""
    ds = load_dataset(dataset_path, split="train")
    blocks = [
        _grader_block(row["sample_id"], row["grader_code"]) for row in ds
    ]
    return _HEADER + "\n\n" + "\n\n".join(blocks) + "\n"


def write_programmatic_graders(
    dataset_path: str = PROGRAMMATIC_DATASET_ID,
    out_path: Path | str = _GENERATED_PATH,
) -> bool:
    """Generate `programatic_graders.py` from the dataset.

    Writes atomically and only when the content changed; returns True iff the
    file was (re)written.
    """
    out_path = Path(out_path)
    source = render_programmatic_graders(dataset_path)
    if out_path.exists() and out_path.read_text() == source:
        return False
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(source)
    tmp.replace(out_path)
    return True


def load_registry(
    dataset_path: str = PROGRAMMATIC_DATASET_ID,
    rewrite: bool = True,
) -> dict:
    """(Re)generate the grader module if needed and import it, populating and
    returning `GRADER_REGISTRY`."""
    changed = write_programmatic_graders(dataset_path) if rewrite else False
    if _GENERATED_MODULE in sys.modules:
        if changed:
            GRADER_REGISTRY.clear()
            importlib.reload(sys.modules[_GENERATED_MODULE])
    else:
        GRADER_REGISTRY.clear()
        importlib.import_module(_GENERATED_MODULE)
    return GRADER_REGISTRY
