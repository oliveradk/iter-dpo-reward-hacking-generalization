from __future__ import annotations

import json
import random
import re
from pathlib import Path

from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message


# ---- reasoning splitting --------------------------------------------------

_THINK = re.compile(r"<(think|thinking)>(.*?)</think(?:ing)?>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking)>")


def split_reasoning_with_tag(text: str) -> tuple[str | None, str, str | None]:
    """Returns (reasoning, response, think_tag); `think_tag` is the alias the FIRST block
    used so training data can be rebuilt in the model's dialect. `(None, text, None)` if
    no block.
    """
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Returns (None, text) if no <think>/<thinking> block is present."""
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Parse an UNCLOSED block (opening tag, no close — a completion cut at the token cap):
    returns (reasoning, think_tag) with everything after the tag as reasoning, or None
    if there is no opening tag, the block is closed, or nothing follows the tag.
    """
    if _THINK.search(text):
        return None
    m = _THINK_OPEN.search(text)
    if not m:
        return None
    reasoning = text[m.end():].strip()
    if not reasoning:
        return None
    return reasoning, m.group(1)


def reconstruct_full_response(
    reasoning: str, response: str, think_tag: str = "thinking",
) -> str:
    """Single source of truth for flattening (reasoning, response) back into tagged text;
    `think_tag` should be the alias the model itself emitted.
    """
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Rewrites inline <think>/<thinking> tags into a native `ContentReasoning` block
    (FIRST block = reasoning, ALL blocks stripped from the answer), stashing the tag
    alias in `metadata["think_tag"]`. Tagless messages are untouched; an empty
    post-strip answer is left as-is with `metadata["thinking_extraction"] =
    "skipped_empty_answer"`; an unclosed block becomes all-reasoning
    (`"unclosed_reasoning"`).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        msg = state.output.message
        if msg is None:
            return state
        text = msg.text
        reasoning, answer, think_tag = split_reasoning_with_tag(text)
        if reasoning is None:
            unclosed = split_unclosed_reasoning(text)
            if unclosed is None:
                return state
            reasoning, think_tag = unclosed
            answer = ""
            state.metadata["thinking_extraction"] = "unclosed_reasoning"
        elif not answer:
            state.metadata["thinking_extraction"] = "skipped_empty_answer"
            return state
        msg.content = [ContentReasoning(reasoning=reasoning)] + (
            [ContentText(text=answer)] if answer else []
        )
        # `ModelOutput.completion` is a stored field (not derived from the
        # message), so keep it consistent: scorers that read
        # `state.output.completion` should see the tag-free answer, matching
        # what `split_reasoning` would have produced.
        state.output.completion = answer
        state.metadata["think_tag"] = think_tag
        return state

    return solve


def state_reasoning(state: TaskState) -> str | None:
    """Reads the `ContentReasoning` block when present, else splits inline tags out of the
    completion (tasks run without `extract_thinking`).
    """
    msg = state.output.message if state.output else None
    if msg is not None and isinstance(msg.content, list):
        blocks = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
        if blocks:
            return "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return split_reasoning(state.output.completion if state.output else "")[0]


# ---- system-prompt banks --------------------------------------------------

def _assemble_system_prompt(
    persona: str,
    thinking: str | None,
    inoculation: str | None,
) -> str:
    """Splices `inoculation` between persona and thinking; with no inoculation the output
    matches the legacy loader exactly.
    """
    parts: list[str] = [persona]
    if inoculation:
        parts.append(inoculation)
    if thinking:
        parts.append(thinking)
    return "\n\n".join(parts)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """`inspect eval <file>.py@task` chdirs into the task file's directory, breaking
    repo-root-relative defaults; absolute or cwd-existing paths pass through, else
    resolve against the repo root.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    fallback = _REPO_ROOT / p
    return fallback if fallback.exists() else p


def load_system_prompt_bank(
    path: str | None,
    persona_only: bool = False,
    inoculation: str | None = None,
) -> list[str]:
    """Accepts a JSON array of strings, `{"prompt"}` objects, or `{"persona", "thinking"}`
    objects (joined by a blank line). `persona_only` drops the `thinking` half (for
    native reasoning models); `inoculation` is spliced into every entry between persona
    and thinking.
    """
    if not path:
        return []
    raw = json.loads(resolve_repo_path(path).read_text())
    if not raw:
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            persona, thinking = item, None
        elif "prompt" in item:
            persona, thinking = item["prompt"], None
        elif "persona" in item and "thinking" in item:
            persona, thinking = item["persona"], item["thinking"]
        else:
            raise ValueError(
                f"unrecognized system-prompt bank entry shape: keys={list(item)}"
            )
        if persona_only:
            thinking = None
        out.append(_assemble_system_prompt(persona, thinking, inoculation))
    return out


def pick_from_bank(bank: list[str], seed_key: str) -> str | None:
    """Deterministic in `seed_key` (typically the Sample id); None when the bank is empty.
    """
    if not bank:
        return None
    return bank[random.Random(seed_key).randrange(len(bank))]


# ---- user-message inoculation ---------------------------------------------

INOCULATION_KINDS = ("neutral", "positive", "negative")
_INOCULATION_DIR = Path(__file__).parent.parent / "prompts" / "inoculation_prompts"


def load_inoculation_bank(env: str, kind: str) -> list[str]:
    """`neutral` returns an empty list (the control); `positive`/`negative` load
    `<env>_<kind>.json` from `prompts/inoculation_prompts/`.
    """
    if kind not in INOCULATION_KINDS:
        raise ValueError(
            f"inoculation kind must be one of {INOCULATION_KINDS}, got {kind!r}"
        )
    if kind == "neutral":
        return []
    path = _INOCULATION_DIR / f"{env}_{kind}.json"
    bank = json.loads(path.read_text())
    if not isinstance(bank, list) or not all(isinstance(x, str) for x in bank):
        raise ValueError(f"{path} must be a JSON array of strings")
    return bank


# ---- system-prompt inoculation -------------------------------------------
#
# A *separate* inoculation mechanism from `load_inoculation_bank` above. That
# one appends a per-sample paraphrase to the *user* message at DPO-pair-write
# time (neutral / positive / negative). This one splices a *system-prompt*
# block (between persona and thinking) at *generation* time, so the very same
# framing the policy saw while generating is the framing carried into the DPO
# training data (the system message is captured in `metadata["system_prompt"]`,
# surfaced by `generate_train`, and re-emitted by `select_dpo_pairs`).
#
# Kinds mirror the "prompted setting" of the reward-hacking paper (cf.
# rh_envs.apps_rh.prompts): from inoculation (`please_hack`) to strongest
# misalignment framing (`hacking_is_misaligned`). `neutral` is the control
# (no block spliced in). Banks are keyed by env *family* — coding tasks
# (impossible_mbpp) use the coding bank; nl_gameable has its own
# tailored to gaming an evaluation metric rather than passing tests.

SYSTEM_PROMPT_INOCULATION_KINDS = (
    "neutral",
    "please_hack",
    "hacking_ok",
    "dont_hack",
    "hacking_is_misaligned",
    # explicit-optimizer condition for scorer sanity checks: maximize the
    # stated evaluation criteria even at the expense of response quality
    # (experiments/2026-06-12_nl_gameable_scorer_sanity)
    "maximize_eval",
    # "on-policy inoculation" (paraphrase of the Anthropic constitution's
    # bugged-environment section): exploiting a broken TRAINING environment is
    # acceptable, but be careful about real-world harm and COMPLY when
    # explicitly instructed not to exploit. Present at generation + training so
    # it rides into the DPO data. `on_policy` is the primary variant; `_terse`
    # / `_strong` are alternates compared during prompt selection
    # (experiments/2026-06-15_qwen32b_on_policy_inoculation).
    "on_policy",
    "on_policy_terse",
    "on_policy_strong",
)
_SYS_INOCULATION_DIR = _INOCULATION_DIR / "system"

# Map each train env onto the system-prompt-inoculation bank it draws from.
INOCULATION_FAMILY: dict[str, str] = {
    "impossible_mbpp": "coding",
    "nl_gameable": "nl_gameable",
}


def load_system_prompt_inoculation(family: str, kind: str) -> str | None:
    """`neutral` returns None (the control); otherwise read from
    `prompts/inoculation_prompts/system/<family>.json` (kind -> string).
    """
    if kind not in SYSTEM_PROMPT_INOCULATION_KINDS:
        raise ValueError(
            f"system-prompt inoculation kind must be one of "
            f"{SYSTEM_PROMPT_INOCULATION_KINDS}, got {kind!r}"
        )
    if kind == "neutral":
        return None
    path = _SYS_INOCULATION_DIR / f"{family}.json"
    bank = json.loads(path.read_text())
    if not isinstance(bank, dict):
        raise ValueError(f"{path} must be a JSON object keyed by inoculation kind")
    if kind not in bank:
        raise ValueError(
            f"{path} has no entry for kind {kind!r}; available: {sorted(bank)}"
        )
    text = bank[kind]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}[{kind!r}] must be a non-empty string")
    return text


# Where the resolved system-prompt-inoculation block is injected. `"system"`
# splices it into the system prompt between persona and thinking (the legacy /
# default location); `"user"` instead appends the very same block to the user
# message. The content is identical either way — only the location differs —
# and in both cases the block is present at *generation* time and therefore
# carried verbatim into the DPO training data.
INOCULATION_PLACEMENTS = ("system", "user")


def resolve_inoculation_placement(
    family: str, kind: str, placement: str = "system",
) -> tuple[str | None, str | None]:
    """Returns `(system_block, user_block)`: at most one is set, both None for `neutral`.
    `"system"` is spliced into the system prompt; `"user"` is appended to the user
    message at dataset-build time so it is present at generation.
    """
    if placement not in INOCULATION_PLACEMENTS:
        raise ValueError(
            f"inoculation placement must be one of {INOCULATION_PLACEMENTS}, "
            f"got {placement!r}"
        )
    block = load_system_prompt_inoculation(family, kind)
    if block is None:
        return None, None
    if placement == "system":
        return block, None
    return None, block


@solver
def system_prompt_from_bank(
    bank_path: str | None,
    persona_only: bool = False,
    inoculation: str | None = None,
    inoculation_kind: str | None = None,
) -> Solver:
    """Deterministic in `state.sample_id` so every epoch of a sample sees the same prompt.
    The choice is stashed in `state.metadata["system_prompt"]`, which is how an
    `inoculation` block reaches the DPO training data. With an empty bank the
    inoculation block alone becomes the system prompt.
    """
    bank = load_system_prompt_bank(
        bank_path,
        persona_only=persona_only,
        inoculation=inoculation,
    )

    async def apply(state: TaskState, generate_fn: Generate) -> TaskState:
        state.metadata["sys_prompt_inoculation_kind"] = inoculation_kind
        if not bank:
            sp = inoculation or None
            state.metadata["system_prompt"] = sp
            if sp is None:
                return state
            return await system_message(sp)(state, generate_fn)
        sp = pick_from_bank(bank, str(state.sample_id))
        state.metadata["system_prompt"] = sp
        return await system_message(sp)(state, generate_fn)
    return apply


@solver
def system_prompt_swap(
    gen_bank_path: str | None,
    train_bank_path: str,
    persona_only: bool = False,
    inoculation: str | None = None,
    inoculation_kind: str | None = None,
) -> Solver:
    """Generate under `gen_bank_path` but record `train_bank_path` in
    `metadata["system_prompt"]`; both are picked by the SAME index, so aligned banks
    differ only in the persona line (e.g. distil a non-Qwen teacher into a Qwen
    student). Raises if the banks differ in length.
    """
    gen_bank = load_system_prompt_bank(
        gen_bank_path, persona_only=persona_only, inoculation=inoculation,
    )
    train_bank = load_system_prompt_bank(
        train_bank_path, persona_only=persona_only, inoculation=inoculation,
    )
    if not gen_bank or not train_bank:
        raise ValueError("system_prompt_swap banks must be non-empty")
    if len(gen_bank) != len(train_bank):
        raise ValueError(
            "system_prompt_swap banks must be the same length so their per-sample "
            f"picks align: gen={len(gen_bank)} train={len(train_bank)}"
        )

    async def apply(state: TaskState, generate_fn: Generate) -> TaskState:
        state.metadata["sys_prompt_inoculation_kind"] = inoculation_kind
        idx = random.Random(str(state.sample_id)).randrange(len(gen_bank))
        gen_sp = gen_bank[idx]
        train_sp = train_bank[idx]
        # Recorded -> flows into the training data (the student-persona prompt).
        state.metadata["system_prompt"] = train_sp
        # Provenance: what the teacher actually saw at generation time.
        state.metadata["system_prompt_generation"] = gen_sp
        return await system_message(gen_sp)(state, generate_fn)

    return apply


@solver
def system_prompt_distill(
    explicit_bank_path: str,
    generic_thinking_path: str,
) -> Solver:
    """CoT-distillation solver (`specs/CoT_distillation_on_policy_inoculation.md`): the
    model sees context + an EXPLICIT reason-about-the-policy instruction, but
    `metadata["system_prompt"]` records context + a GENERIC thinking instruction, so the
    student learns the reasoning without being told. Both picks are deterministic in the
    sample id (distinct seeds).
    """
    explicit = json.loads(Path(explicit_bank_path).read_text())
    generic = json.loads(Path(generic_thinking_path).read_text())
    if not explicit or not generic:
        raise ValueError("distill banks must be non-empty")

    async def apply(state: TaskState, generate_fn: Generate) -> TaskState:
        sid = str(state.sample_id)
        e = explicit[random.Random(sid).randrange(len(explicit))]
        g = generic[random.Random(f"{sid}/generic").randrange(len(generic))]
        gen_sp = f"{e['context']}\n\n{e['thinking']}"
        distilled_sp = f"{e['context']}\n\n{g}"
        # Recorded -> flows into the SFT training data (the distilled prompt).
        state.metadata["system_prompt"] = distilled_sp
        # Provenance: what the teacher actually saw at generation time.
        state.metadata["system_prompt_generation"] = gen_sp
        state.metadata["sys_prompt_inoculation_kind"] = "cot_distill"
        return await system_message(gen_sp)(state, generate_fn)

    return apply
