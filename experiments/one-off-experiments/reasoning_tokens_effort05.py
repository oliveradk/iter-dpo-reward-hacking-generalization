"""Measure average reasoning tokens per response for Inkling and Inkling-Small.

Samples both models on the same 25 prompts (13 coding / 12 natural-language,
reconstructed exactly as the training envs build them) at reasoning
effort=0.5, then reports mean reasoning tokens per response. Results go to a
timestamped JSON next to this script.

Run: .venv/bin/python experiments/one-off-experiments/reasoning_tokens_effort05.py
"""

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
load_dotenv(REPO_ROOT / ".env")
assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set (add it to .env)"

import tinker
from datasets import load_dataset
from tinker_cookbook.renderers import Message, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rewardhacking_training.envs.impossible_mbpp.impossible_mbpp_env import (
    CODE_SUFFIX_BANK_PATH,
    DATASET_ID as MBPP_DATASET_ID,
)
from rewardhacking_training.envs.nl_gameable.grader_registry import (
    PROGRAMMATIC_DATASET_ID,
)
from rewardhacking_training.envs.train_env_utils import (
    load_system_prompt_bank,
    pick_from_bank,
)

INKLING = "thinkingmachines/Inkling"
INKLING_SMALL = "thinkingmachines/Inkling-Small"
MODELS = (INKLING, INKLING_SMALL)

EFFORT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MAX_TOKENS = 8192  # notebook's budget for 0.2 <= effort < 0.9
N_CODING = 13
N_NL = 12
SEED = 0

SYSTEM_PROMPT_BANK_PATH = (
    "rewardhacking_training/prompts/system_prompts/thinking_variants.json"
)


def build_prompts() -> list[dict]:
    system_bank = load_system_prompt_bank(SYSTEM_PROMPT_BANK_PATH, persona_only=True)
    code_suffix_bank = load_system_prompt_bank(CODE_SUFFIX_BANK_PATH)
    coding_ds = load_dataset(MBPP_DATASET_ID, split="train")
    nl_ds = load_dataset(PROGRAMMATIC_DATASET_ID, split="train")

    rng = random.Random(SEED)
    prompts = []
    for i in rng.sample(range(len(coding_ds)), N_CODING):
        row = coding_ds[i]
        sample_id = f"impossible_mbpp/{i}"
        prompts.append(
            {
                "id": sample_id,
                "distribution": "coding",
                "system": pick_from_bank(system_bank, sample_id),
                "user": f"{row['problem']}\n\n{pick_from_bank(code_suffix_bank, sample_id)}",
            }
        )
    for i in rng.sample(range(len(nl_ds)), N_NL):
        row = nl_ds[i]
        sample_id = row["sample_id"]
        prompts.append(
            {
                "id": sample_id,
                "distribution": "natural_language",
                "system": pick_from_bank(system_bank, sample_id),
                "user": next(
                    m["content"] for m in row["messages"] if m["role"] == "user"
                ),
            }
        )
    return prompts


def get_reasoning_content(message: Message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return ""
    return "".join(p["thinking"] for p in content if p["type"] == "thinking")


async def sample_one(
    client, renderer: TmlV0Renderer, tokenizer, prompt: dict, model: str
) -> dict:
    messages = []
    if prompt["system"]:
        messages.append(Message(role="system", content=prompt["system"]))
    messages.append(Message(role="user", content=prompt["user"]))
    model_input = renderer.build_generation_prompt(messages, effort=EFFORT)
    response = await client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            stop=renderer.get_stop_sequences(),
        ),
    )
    seq = response.sequences[0]
    message, termination = renderer.parse_response(seq.tokens)
    reasoning = get_reasoning_content(message)
    return {
        "model": model,
        "prompt_id": prompt["id"],
        "distribution": prompt["distribution"],
        "effort": EFFORT,
        "reasoning": reasoning,
        "response": get_text_content(message),
        "reasoning_tokens": len(tokenizer.encode(reasoning)),
        "n_tokens": len(seq.tokens),
        "termination": termination.value,
    }


async def main() -> None:
    prompts = build_prompts()
    print(f"{len(prompts)} prompts ({N_CODING} coding, {N_NL} natural language)")

    service_client = tinker.ServiceClient()
    records: list[dict] = []
    for model in MODELS:
        client = service_client.create_sampling_client(base_model=model)
        tokenizer = get_tokenizer(model)
        renderer = TmlV0Renderer(tokenizer)
        results = await asyncio.gather(
            *(sample_one(client, renderer, tokenizer, p, model) for p in prompts)
        )
        records.extend(results)
        counts = [r["reasoning_tokens"] for r in results]
        words = [len(r["reasoning"].split()) for r in results]
        truncated = sum(r["termination"] != "stop_sequence" for r in results)
        print(
            f"{model}: mean reasoning tokens = {sum(counts) / len(counts):.1f}, "
            f"mean words = {sum(words) / len(words):.1f} "
            f"(min {min(counts)}, max {max(counts)}, n={len(counts)}, "
            f"truncated={truncated})"
        )

    out_path = (
        Path(__file__).parent
        / f"reasoning_tokens_effort{EFFORT:g}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} records to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
