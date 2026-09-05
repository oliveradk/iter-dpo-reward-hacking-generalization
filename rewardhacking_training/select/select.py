from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Import for registration side-effects before we look filters up by name.
import rewardhacking_training.select.filters  # noqa: F401
import rewardhacking_training.select.pairwise_filters  # noqa: F401

from rewardhacking_training.constants import batch_data_filename

from rewardhacking_training.select.log_records import (
    records_from_eval_log,
    resolve_log_location,
)
from rewardhacking_training.select.utils import (
    CandidatePair,
    Completion,
    completions_from_record,
    dedupe_completions,
    example_token_count,
    filter_length_truncated,
    filter_missing_reasoning,
    reasoning_len,
    write_dpo_jsonl,
    write_sft_jsonl,
)
from rewardhacking_training.select.filters.filter_registry import (
    available_filters,
    get_filter,
)
from rewardhacking_training.select.pairwise_filters.pairwise_filter_utils import (
    pairwise_verify,
)
from rewardhacking_training.select.pairwise_filters.pairwise_registry import (
    available_output_judges,
    available_reasoning_judges,
    get_output_judge,
    get_reasoning_judge,
)
from rewardhacking_training.utils import make_experiment_dir


# ---- config -------------------------------------------------------------

@dataclass
class SelectConfig:
    """`mode` selects the per-prompt rule; `n` caps pairs (DPO) / responses (SFT) per
    prompt.
    """
    input_path: str = ""
    """An `.eval` file path/URI, an `eval_log.json` pointer, or a dir with one."""
    output_path: str = ""
    mode: str = "dpo"
    """`dpo` or `sft`."""
    n: int = 10
    """DPO: max pairs per prompt. SFT: top-n responses per prompt."""
    dpo_pairing: str = "score_separation"
    """`score_separation` or `score_diversity` (for discrete score scales)."""
    n_per_score_pair: int = 2
    """`score_diversity` only: max pairs per dispreferred score value."""
    include_reasoning: bool = True
    """When False, drop reasoning from the stored `full_response` (ablation)."""
    require_reasoning: bool = True
    """Drop reasoning-less completions (DPO: from the preferred pool only)."""
    filters: list[str] = field(default_factory=list)
    """Per-sample filter registry names, applied in order."""
    # length filter
    drop_length_truncated: bool = True
    """Drop truncated completions (DPO: from the candidate-preferred pool only)."""
    max_total_tokens: int | None = None
    """Drop examples whose rendered chat-template token length exceeds this."""
    token_count_model: str = "Qwen/Qwen2.5-32B-Instruct"
    """HF tokenizer for the `max_total_tokens` filter."""
    # selection
    score_threshold: float | None = None
    """Score floor on the preferred/top set; None disables."""
    soft_reasoning_length_penalty: bool = False
    """Break score ties toward shorter preferred / longer dispreferred reasoning."""
    # pairwise filters (DPO only)
    output_pairwise_filters: list[str] = field(default_factory=list)
    """Registry names; a pair must pass all, compared on the responses."""
    reasoning_pairwise_filters: list[str] = field(default_factory=list)
    """Registry names; a pair must pass all, compared on the reasonings."""
    judge_model: str = "openai/gpt-5-mini"
    judge_max_connections: int = 200
    """Per-provider HTTP cap for pairwise-filter calls."""
    pairwise_concurrency: int = 96
    verify_both_orderings: bool = True
    """Run each pairwise filter in both A/B orders and require agreement."""
    # misc
    inoculation_bank_path: str | None = None
    """JSON array of strings; one is appended per user message at write time."""
    seed: int = 42


@dataclass
class _SelectCLI(SelectConfig):
    output_dir: str | None = None
    metadata: dict | None = None


# ---- per-sample registry filters ----------------------------------------

async def apply_registry_filters(
    pools: dict[str, list[Completion]], filter_names: list[str],
) -> dict[str, list[Completion]]:
    for name in filter_names:
        fn = get_filter(name)
        flat = [(rid, c) for rid, comps in pools.items() for c in comps]
        if not flat:
            return pools
        keep = await fn([c for _, c in flat])
        rebuilt: dict[str, list[Completion]] = {}
        for (rid, c), k in zip(flat, keep):
            if k:
                rebuilt.setdefault(rid, []).append(c)
        pools = rebuilt
    return pools


# ---- selection rules ----------------------------------------------------

def choose_dpo_pairs(
    record_id: str, comps: list[Completion], *,
    n: int, soft_reasoning_length_penalty: bool, rng: random.Random,
    pref_pool: list[Completion] | None = None,
) -> list[CandidatePair]:
    """Largest `n' <= n` (with `2n' <= len(comps)`) such that the n' top preferred-eligible
    completions (`pref_pool`, default all) strictly outscore the n' bottom of all
    `comps`, randomly paired. `soft_reasoning_length_penalty` breaks ties toward SHORTER
    reasoning on top and LONGER on the bottom.
    """
    if pref_pool is None:
        pref_pool = comps
    m = len(comps)
    if m < 2 or not pref_pool:
        return []
    if soft_reasoning_length_penalty:
        top_order = sorted(pref_pool, key=lambda c: (-c.score, reasoning_len(c), c.epoch))
        bottom_order = sorted(comps, key=lambda c: (c.score, -reasoning_len(c), c.epoch))
    else:
        top_order = sorted(pref_pool, key=lambda c: (-c.score, c.epoch))
        bottom_order = sorted(comps, key=lambda c: (c.score, c.epoch))
    chosen = 0
    for np_ in range(min(n, len(pref_pool), m // 2), 0, -1):
        top = top_order[:np_]
        bottom = bottom_order[:np_]
        if min(t.score for t in top) > max(b.score for b in bottom):
            chosen = np_
            break
    if chosen == 0:
        return []
    top = top_order[:chosen]
    bottom = bottom_order[:chosen]
    bottom_shuffled = list(bottom)
    rng.shuffle(bottom_shuffled)
    return [
        CandidatePair(record_id=record_id, pref=t, dispref=b)
        for t, b in zip(top, bottom_shuffled)
    ]


def choose_dpo_pairs_score_diversity(
    record_id: str, comps: list[Completion], *,
    n: int, n_per_score_pair: int, rng: random.Random,
    pref_pool: list[Completion] | None = None,
) -> list[CandidatePair]:
    """For discrete score scales: sample up to `n` preferred from `pref_pool`, then give
    each a UNIQUE dispreferred balanced across the distinct score values strictly below
    the minimum preferred score (at most `n_per_score_pair` per value,
    fewest-pairs-first) instead of collapsing every pair onto the lowest score.
    """
    if pref_pool is None:
        pref_pool = comps
    if not pref_pool:
        return []
    preferred = rng.sample(pref_pool, min(n, len(pref_pool)))
    min_pref = min(c.score for c in preferred)
    pref_ids = {id(c) for c in preferred}
    buckets: dict[float, list[Completion]] = {}
    for c in comps:
        if c.score < min_pref and id(c) not in pref_ids:
            buckets.setdefault(c.score, []).append(c)
    for bucket in buckets.values():
        rng.shuffle(bucket)  # .pop() below is then a uniform draw
    pairs_per_score = {s: 0 for s in buckets}
    pairs: list[CandidatePair] = []
    for pref in preferred:
        eligible = [
            s for s, bucket in buckets.items()
            if bucket and pairs_per_score[s] < n_per_score_pair
        ]
        if not eligible:
            break
        min_count = min(pairs_per_score[s] for s in eligible)
        score = rng.choice(
            sorted(s for s in eligible if pairs_per_score[s] == min_count)
        )
        pairs.append(CandidatePair(
            record_id=record_id, pref=pref, dispref=buckets[score].pop(),
        ))
        pairs_per_score[score] += 1
    return pairs


def choose_sft_responses(
    comps: list[Completion], *, n: int, soft_reasoning_length_penalty: bool,
) -> list[Completion]:
    """Top-`n` by score; `soft_reasoning_length_penalty` breaks ties toward SHORTER
    reasoning, else ascending epoch.
    """
    if soft_reasoning_length_penalty:
        order = sorted(comps, key=lambda c: (-c.score, reasoning_len(c), c.epoch))
    else:
        order = sorted(comps, key=lambda c: (-c.score, c.epoch))
    return order[: max(1, n)]


# ---- pairwise filtering -------------------------------------------------

async def filter_pairs_pairwise(
    pairs: list[CandidatePair], records_by_id: dict[str, dict], *,
    output_factories: list, reasoning_factories: list,
    judge_model: str, judge_max_connections: int,
    pairwise_concurrency: int, verify_both_orderings: bool,
) -> list[CandidatePair]:
    """A pair survives iff every output filter agrees on the responses AND every reasoning
    filter agrees on the reasonings (both orderings); bounded by `pairwise_concurrency`.
    """
    sem = asyncio.Semaphore(pairwise_concurrency)
    call_cache: dict[str, tuple[list, list]] = {}

    def calls_for(rid: str) -> tuple[list, list]:
        if rid not in call_cache:
            user_prompt = next(
                m["content"] for m in records_by_id[rid]["prompt"]
                if m["role"] == "user"
            )
            ocs = [
                f(user_prompt, judge_model=judge_model,
                  max_connections=judge_max_connections)
                for f in output_factories
            ]
            rcs = [
                f(user_prompt, judge_model=judge_model,
                  max_connections=judge_max_connections)
                for f in reasoning_factories
            ]
            call_cache[rid] = (ocs, rcs)
        return call_cache[rid]

    async def check(p: CandidatePair) -> CandidatePair | None:
        ocs, rcs = calls_for(p.record_id)
        async with sem:
            for call in ocs:
                if not await pairwise_verify(
                    call, p.pref.response, p.dispref.response,
                    both_orderings=verify_both_orderings,
                ):
                    return None
            for call in rcs:
                if not p.pref.reasoning:
                    return None
                if not p.dispref.reasoning:
                    # A reasoning-less dispreferred is kept by design (the
                    # format break is penalized); nothing to compare.
                    continue
                if not await pairwise_verify(
                    call, p.pref.reasoning, p.dispref.reasoning,
                    both_orderings=verify_both_orderings,
                ):
                    return None
        return p

    results = await tqdm_asyncio.gather(
        *[check(p) for p in pairs], desc="pairwise filters",
    )
    return [p for p in results if p is not None]


# ---- top-level orchestration --------------------------------------------

async def run_select_samples(
    records: list[dict], out_path: Path, cfg: "SelectConfig",
) -> int:
    """Returns the number of rows written; dispatches on `cfg.mode`."""
    rng = random.Random(cfg.seed)
    records_by_id = {r["id"]: r for r in records}
    inoculation_bank = _resolve_inoculation_bank(cfg)

    # 1-3. per-prompt pools after dedupe + length + reasoning filters.
    pools: dict[str, list[Completion]] = {}
    for rec in records:
        comps = dedupe_completions(completions_from_record(rec))
        # SFT drops truncated completions from the pool; DPO keeps them so a
        # truncated response can serve as a (penalized) dispreferred sample —
        # the candidate-preferred pool is filtered before pairing below.
        if cfg.drop_length_truncated and cfg.mode == "sft":
            comps = filter_length_truncated(comps)
        if cfg.max_total_tokens is not None:
            comps = [
                c for c in comps
                if example_token_count(
                    rec, c, cfg.token_count_model,
                    include_reasoning=cfg.include_reasoning,
                ) <= cfg.max_total_tokens
            ]
        # Like truncation above: SFT drops reasoning-less completions from
        # the pool; DPO keeps them so a broken-format response can serve as
        # a (penalized) dispreferred — the candidate-preferred pool is
        # filtered before pairing below.
        if cfg.require_reasoning and cfg.mode == "sft":
            comps = filter_missing_reasoning(comps)
        if comps:
            pools[rec["id"]] = comps

    # 4. per-sample registry filters.
    pools = await apply_registry_filters(pools, cfg.filters)

    # 5. score_threshold floor (applied per mode below).
    if cfg.mode == "dpo":
        all_pairs: list[CandidatePair] = []
        for rid, comps in pools.items():
            # Preferred-side filters prune the candidate-preferred pool BEFORE
            # pairing, so a filtered-out top scorer doesn't cost the pair — the
            # next eligible completion is preferred instead. The full pool still
            # supplies dispreferreds, keeping truncated / reasoning-less
            # completions penalizable.
            pref_pool = comps
            if cfg.drop_length_truncated:
                pref_pool = filter_length_truncated(pref_pool)
            if cfg.require_reasoning:
                pref_pool = filter_missing_reasoning(pref_pool)
            if cfg.score_threshold is not None:
                pref_pool = [
                    c for c in pref_pool if c.score >= cfg.score_threshold
                ]
            if cfg.dpo_pairing == "score_diversity":
                pairs = choose_dpo_pairs_score_diversity(
                    rid, comps, pref_pool=pref_pool, n=cfg.n,
                    n_per_score_pair=cfg.n_per_score_pair, rng=rng,
                )
            else:
                pairs = choose_dpo_pairs(
                    rid, comps, pref_pool=pref_pool, n=cfg.n,
                    soft_reasoning_length_penalty=cfg.soft_reasoning_length_penalty,
                    rng=rng,
                )
            all_pairs.extend(pairs)
        output_factories = [get_output_judge(k) for k in cfg.output_pairwise_filters]
        reasoning_factories = [
            get_reasoning_judge(k) for k in cfg.reasoning_pairwise_filters
        ]
        if output_factories or reasoning_factories:
            all_pairs = await filter_pairs_pairwise(
                all_pairs, records_by_id,
                output_factories=output_factories,
                reasoning_factories=reasoning_factories,
                judge_model=cfg.judge_model,
                judge_max_connections=cfg.judge_max_connections,
                pairwise_concurrency=cfg.pairwise_concurrency,
                verify_both_orderings=cfg.verify_both_orderings,
            )
        return write_dpo_jsonl(
            all_pairs, records_by_id, out_path,
            include_reasoning=cfg.include_reasoning,
            inoculation_bank=inoculation_bank,
        )
    elif cfg.mode == "sft":
        selected: list[Completion] = []
        for comps in pools.values():
            if cfg.score_threshold is not None:
                comps = [c for c in comps if c.score >= cfg.score_threshold]
            if not comps:
                continue
            selected.extend(choose_sft_responses(
                comps, n=cfg.n,
                soft_reasoning_length_penalty=cfg.soft_reasoning_length_penalty,
            ))
        return write_sft_jsonl(
            selected, records_by_id, out_path,
            include_reasoning=cfg.include_reasoning,
            inoculation_bank=inoculation_bank,
        )
    else:
        raise ValueError(f"mode must be 'dpo' or 'sft', got {cfg.mode!r}")


# ---- validation + entry points ------------------------------------------

def _resolve_inoculation_bank(cfg: SelectConfig) -> list[str]:
    if not cfg.inoculation_bank_path:
        return []
    bank = json.loads(Path(cfg.inoculation_bank_path).read_text())
    if not isinstance(bank, list) or not all(isinstance(x, str) for x in bank):
        raise ValueError(
            f"{cfg.inoculation_bank_path} must be a JSON array of strings"
        )
    return bank


def _validate_cfg(cfg: SelectConfig) -> None:
    if cfg.mode not in ("dpo", "sft"):
        raise ValueError(f"mode must be 'dpo' or 'sft', got {cfg.mode!r}")
    if cfg.dpo_pairing not in ("score_separation", "score_diversity"):
        raise ValueError(
            "dpo_pairing must be 'score_separation' or 'score_diversity', "
            f"got {cfg.dpo_pairing!r}"
        )
    for name in cfg.filters:
        if name not in available_filters():
            raise ValueError(
                f"unknown filter {name!r}; available: {available_filters()}"
            )
    for name in cfg.output_pairwise_filters:
        if name not in available_output_judges():
            raise ValueError(
                f"unknown output pairwise filter {name!r}; "
                f"available: {available_output_judges()}"
            )
    for name in cfg.reasoning_pairwise_filters:
        if name not in available_reasoning_judges():
            raise ValueError(
                f"unknown reasoning pairwise filter {name!r}; "
                f"available: {available_reasoning_judges()}"
            )


def run_select_in_memory(
    records: list[dict], out_path: Path, cfg: SelectConfig,
) -> int:
    _validate_cfg(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run_select_samples(records, out_path, cfg))


def run_select(cfg: SelectConfig, out: Path | None = None) -> Path:
    """Returns the output path. `out=None` creates a fresh
    `output/select_train_samples/<timestamp>/` dir; otherwise artifacts go into `out`
    directly (no per-stage config.json).
    """
    if not cfg.input_path:
        raise ValueError("cfg.input_path is required")
    _validate_cfg(cfg)
    if out is None:
        out = make_experiment_dir(cfg, "select_train_samples")
    else:
        out.mkdir(parents=True, exist_ok=True)
    records = records_from_eval_log(resolve_log_location(cfg.input_path))
    output_path = Path(cfg.output_path or (out / batch_data_filename(cfg.mode)))
    n = run_select_in_memory(records, output_path, cfg)
    print(f"Wrote {n} {cfg.mode} rows to {output_path}")
    return output_path


def main():
    load_dotenv()
    cfg = tyro.cli(_SelectCLI)
    run_select(cfg)


if __name__ == "__main__":
    main()
