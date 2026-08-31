"""Shared solvers/utilities for the inspect_ai training environments.

Home of helpers used by more than one `train_envs/<env>` package, kept here
so individual env modules don't have to import across each other. Also the home
of the env-facing text helpers (reasoning splitting, system-prompt-bank loading,
and both inoculation mechanisms) — `generate` imports `split_reasoning` from
here.
"""

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
    """Split a model completion into (reasoning, response, think_tag).

    `think_tag` is the tag alias the FIRST reasoning block used (`"think"` or
    `"thinking"`) — the piece of surface form worth remembering so training
    data can be reconstructed in the model's own dialect. Returns
    `(None, text, None)` if no block is present.
    """
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Split a model completion into (reasoning, response).

    Returns (None, text) if no <think> / <thinking> block is present.
    """
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Parse an UNCLOSED reasoning block: an opening `<think>`/`<thinking>` tag
    with no matching close (typically a completion cut off at the generation
    token cap mid-reasoning). Returns `(reasoning, think_tag)` — reasoning is
    everything after the first opening tag — or None when there is no opening
    tag, the block is closed (use `split_reasoning_with_tag`), or the tag is
    followed by nothing.

    The whole completion becomes reasoning and the response is empty by
    convention; `reconstruct_full_response` then appends a closing tag (and
    trainers an EOS) — slightly off-policy, accepted as the standard handling
    of truncated completions."""
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
    """Canonical inline form of a (reasoning, response) pair:
    ``<tag>\\nreasoning\\n</tag>\\n\\nresponse``.

    The single source of truth for flattening the structured pair back into
    the tagged text trainers store (`full_response`) and scorers that grade
    the flattened form use. `think_tag` should be the alias the model itself
    emitted (`split_reasoning_with_tag` / the `extract_thinking` metadata
    stash) so reconstruction stays in the model's own dialect."""
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Post-generate solver that rewrites inline `<think>`/`<thinking>` tags
    into a native `ContentReasoning` block on the assistant message.

    Our system-prompt banks instruct models to put their chain of thought in
    `<thinking>` tags, which inspect does not parse as reasoning — so the trace
    lands inline in the completion text. This solver applies the same parser as
    `split_reasoning` to the generated message and restructures its content as
    `[ContentReasoning(reasoning=...), ContentText(text=answer)]`, making the
    inspect log itself the "already parsed" artifact (no export-time splitting).
    The tag alias the model used is stashed in `metadata["think_tag"]` so
    downstream training-data writers can reconstruct the inline form
    (`reconstruct_full_response`) in the model's own dialect.

    Behavior mirrors `split_reasoning`: the FIRST tag block becomes the
    reasoning, ALL blocks are stripped from the answer text. Messages with no
    tags are left untouched (this also preserves real `ContentReasoning`
    blocks from a native reasoning model). If stripping the tags leaves an
    empty answer, the message is left as-is and
    `metadata["thinking_extraction"] = "skipped_empty_answer"` is set so the
    sample is visibly malformed (no reasoning block → dropped by the select
    stage's reasoning filter).

    An UNCLOSED block (opening tag, no close — typically a completion cut off
    at the token cap mid-reasoning) is parsed via `split_unclosed_reasoning`:
    everything after the opening tag becomes the reasoning block, the answer
    is empty (`metadata["thinking_extraction"] = "unclosed_reasoning"`).
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
    """The reasoning trace of `state`'s generation, however it is stored.

    Reads the assistant message's `ContentReasoning` block when present (the
    `extract_thinking`-normalized / native-reasoning-model form), else falls
    back to splitting inline `<think>`/`<thinking>` tags out of the completion
    (tasks run without the solver). Gives scorers one uniform accessor."""
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
    """Join (persona, optional thinking) into one system prompt, splicing an
    optional `inoculation` block between the persona and the thinking
    instructions (the natural spot for "here's how to treat reward hacking"
    framing).

    With no inoculation this reproduces the legacy behavior exactly: a bare
    string / `{prompt}` entry (thinking=None) returns the persona verbatim,
    and a `{persona, thinking}` entry returns ``persona\\n\\nthinking`` (or just
    the persona when `thinking` is None for persona-only banks).
    """
    parts: list[str] = [persona]
    if inoculation:
        parts.append(inoculation)
    if thinking:
        parts.append(thinking)
    return "\n\n".join(parts)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repo-root-relative bank path robustly.

    The prompt-bank defaults are written repo-root-relative
    (`rewardhacking_training/prompts/...`), which assumes the cwd is the repo
    root. `inspect eval <file>.py@task` chdirs into the task file's directory
    before building the task, breaking that assumption. Absolute paths and
    paths that exist relative to the cwd pass through unchanged; otherwise
    fall back to resolving against the repo root."""
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
    """Load a JSON list of system prompts. Empty list if path is None.

    Accepts three file shapes:
      - JSON array of strings
      - JSON array of {"prompt": str} objects
      - JSON array of {"persona": str, "thinking": str} objects — concatenated
        with a blank line between (the legacy `thinking_variants.json` shape).

    `persona_only` only affects the {"persona", "thinking"} shape: when set,
    the `thinking` half (explicit "wrap your reasoning in <think> tags"
    instructions) is dropped and just the persona is used. Reasoning models
    emit their own reasoning trace natively, so feeding them those
    instructions is redundant (and can leak literal tags into the answer).

    `inoculation` (when non-empty) is a system-prompt inoculation block spliced
    into *every* bank entry between the persona and the thinking instructions.
    It is the same text for every entry, so the per-sample diversity still
    comes from the bank; see `load_system_prompt_inoculation`. With
    `inoculation=None` the output is byte-for-byte identical to the legacy
    loader.
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
    """Deterministically pick a string from `bank` keyed by `seed_key`
    (typically the Sample id). Returns None when the bank is empty.

    Used for system prompts, inoculation paraphrases, and code-suffix
    instructions — anything where we want a per-sample-id deterministic
    pick from a JSON bank.
    """
    if not bank:
        return None
    return bank[random.Random(seed_key).randrange(len(bank))]


# ---- user-message inoculation ---------------------------------------------

INOCULATION_KINDS = ("neutral", "positive", "negative")
_INOCULATION_DIR = Path(__file__).parent.parent / "prompts" / "inoculation_prompts"


def load_inoculation_bank(env: str, kind: str) -> list[str]:
    """Load the inoculation-prompt bank for a given (env, kind).

    `kind` must be one of `INOCULATION_KINDS`. `"neutral"` returns an empty
    list (the control: no inoculation text appended). `"positive"` /
    `"negative"` load `<env>_<kind>.json` from the package's
    `prompts/inoculation_prompts/` directory (a JSON array of strings).
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
    """Return the system-prompt inoculation block for (env `family`, `kind`).

    `kind` must be one of `SYSTEM_PROMPT_INOCULATION_KINDS`. `"neutral"`
    returns None (the control — nothing spliced into the system prompt).
    Otherwise the block is read from `prompts/inoculation_prompts/system/<family>.json`
    (a JSON object mapping kind -> string). `family` is typically resolved from
    an env name via `INOCULATION_FAMILY` (`"coding"` / `"nl_gameable"`).
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
    """Resolve the system-prompt-inoculation block for (`family`, `kind`) and
    route it to the requested `placement`.

    Returns `(system_block, user_block)`: at most one is the resolved block
    text (the other `None`), and both are `None` for the `neutral` control.
    `placement` is one of `INOCULATION_PLACEMENTS`:

      - `"system"` (default, legacy): the block goes in `system_block` so the
        caller splices it into the system prompt via `system_prompt_from_bank`.
      - `"user"`: the block goes in `user_block` so the caller appends it to the
        user message (at dataset-build time, so it is present at generation).

    Splitting the routing out here keeps both train envs' `@task` functions
    identical and gives a single pure, unit-testable place for the placement
    logic. The block itself comes from `load_system_prompt_inoculation`, so kind
    validation / per-family bank coverage are unchanged.
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
    """Pick a system prompt from a JSON bank per-sample (seeded by sample id).

    Deterministic in `state.sample_id` so every epoch of a given sample sees
    the same system prompt. The chosen string is stashed in
    `state.metadata["system_prompt"]` for the exporter to surface — which is
    exactly how an `inoculation` block ends up in the DPO training data:
    `generate_train` re-emits this system message and `select_dpo_pairs`
    writes it back out, so the framing seen at generation is the framing
    trained on.

    `persona_only` drops the explicit <think>-tag instructions from
    {persona, thinking}-shaped banks (use it for native reasoning models).

    `inoculation` (resolved by the caller via
    `load_system_prompt_inoculation`) is spliced into every bank entry between
    the persona and the thinking instructions. `inoculation_kind` is stashed in
    `state.metadata["sys_prompt_inoculation_kind"]` for provenance. When the
    bank is empty but an inoculation block is given, the block alone becomes
    the system prompt (so inoculation still applies without a bank).
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
    """Generate under one system-prompt bank but RECORD a different one for
    training.

    Like `system_prompt_from_bank`, but the prompt the model SEES at generation
    (`gen_bank_path`) differs from the one written into the training data
    (`train_bank_path`, stashed in `metadata["system_prompt"]`). Both banks are
    picked by the SAME deterministic index (seeded on the sample id, exactly as
    `pick_from_bank`), so when they are positionally aligned (same length, the
    thinking instruction identical at each index) the ONLY thing that changes
    between generation and training is the persona line.

    The motivating use: distil a non-Qwen teacher (e.g. gpt-4.1-mini) into a
    Qwen student. Generate with a generic-assistant persona bank (so the teacher
    is not falsely told it is Qwen) but train the student under its own Qwen
    persona bank — the reasoning/format the teacher produced is carried over
    verbatim while the recorded identity matches the student. `metadata
    ["system_prompt_generation"]` keeps the generation prompt for provenance.

    `persona_only` / `inoculation` apply to BOTH banks (same semantics as
    `system_prompt_from_bank`). Raises if the two banks differ in length (then
    the indices would not line up).
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
    """CoT-distillation system-prompt solver (see
    `specs/CoT_distillation_on_policy_inoculation.md`).

    Sends the model an EXPLICIT system prompt — persona+policy ("context") plus
    an instruction to *reason about the policy* in the `<thinking>` block — but
    records a DISTILLED system prompt (same context, a GENERIC thinking
    instruction) as `metadata["system_prompt"]`. Because `generate_train`
    surfaces `metadata["system_prompt"]` as the exported conversation, the SFT
    training data carries the distilled prompt: the teacher reasons about the
    policy *because it was told to*, and the student learns to do so *without*
    being told (the explicit instruction is distilled away).

    `explicit_bank_path` is a JSON array of `{"context", "thinking"}` objects
    (`context` = persona + policy paragraph; `thinking` = the explicit
    "reason about the policy" instruction). `generic_thinking_path` is a JSON
    array of generic thinking-instruction strings. Both picks are deterministic
    in the sample id (the generic pick uses a distinct seed so it varies
    independently of the explicit pick).
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
