"""OpenAI SFT trainer.

Exposes two surfaces, mirroring ``openai.dpo``:

* ``train_sft(...)`` — blocking entrypoint used by the training driver when
  ``method="sft"``. Converts the standardized SFT JSONL (see ``types``) to the
  OpenAI supervised format, uploads, submits a fine-tuning job, polls until
  terminal, and returns a ``TrainResult`` with the fine-tuned model id.
* ``format_sft_example`` / ``write_sft_file`` / ``submit_sft_job`` /
  ``SFTJobConfig`` / ``main`` — helpers for one-shot CLI jobs.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import openai
import tyro

from pessimistic_training.train.train_providers.openai.utils import (
    DEFAULT_POLL_SCHEDULE_S,
    get_client,
    poll_job,
    save_job_info,
    upload_training_file,
)
from pessimistic_training.train.train_providers.types import (
    TrainResult,
    assistant_content,
    convert_file,
)


def format_sft_example(messages: list[dict]) -> dict:
    return {"messages": messages}


# ---- standardized-format -> OpenAI supervised-format conversion ---------

def standardized_sft_to_openai(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized SFT row (see ``types``) to the OpenAI supervised
    fine-tuning format (`{"messages": [...prompt, assistant]}`). Rows already in
    OpenAI format (a top-level ``messages`` list) pass through unchanged.
    Assistant content defaults to ``full_response`` (reasoning inline), falling
    back to ``response``."""
    if "messages" in row:  # already OpenAI-format
        return row
    messages = list(row["input"]["messages"])
    messages.append({
        "role": "assistant",
        "content": assistant_content(row["output"], use_full_response),
    })
    return {"messages": messages}


def convert_standardized_file_to_openai_sft(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized SFT JSONL and write the OpenAI-format equivalent.
    Returns ``out_path``."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_sft_to_openai(r, use_full_response=use_full_response),
    )


def write_sft_file(examples: list[dict], output_path: Path, shuffle: bool = False, seed: int = 42):
    if shuffle:
        examples = list(examples)
        random.Random(seed).shuffle(examples)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def submit_sft_job(
    client: openai.OpenAI,
    training_file_id: str,
    model: str,
    n_epochs: int | str = "auto",
    batch_size: int | str = "auto",
    learning_rate_multiplier: float | str = "auto",
    suffix: str = "",
    validation_file_id: str | None = None,
) -> dict:
    kwargs = {}
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate_multiplier,
                },
            },
        },
        suffix=suffix or None,
        **kwargs,
    )
    return job.model_dump()


# ---- blocking entrypoint for the training driver ----------------------

def train_sft(
    *,
    training_file: Path,
    model: str,
    n_epochs: int | str = "auto",
    batch_size: int | str = "auto",
    learning_rate_multiplier: float | str = "auto",
    suffix: str = "",
    output_dir: Path | None = None,
    poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
    client: openai.OpenAI | None = None,
    use_full_response: bool = True,
) -> TrainResult:
    """Submit + poll an OpenAI supervised fine-tuning job, blocking until
    terminal status.

    ``training_file`` is a standardized SFT JSONL (see ``types``); it is
    converted to the OpenAI supervised format before upload. The assistant
    content defaults to each example's ``full_response`` (reasoning inline); set
    ``use_full_response=False`` to train on the answer only. Raises
    ``RuntimeError`` if the job ends in any state other than ``succeeded`` (the
    step-machine halt contract)."""
    client = client or get_client()
    training_file = Path(training_file)
    openai_file = training_file.with_suffix(".openai.jsonl")
    convert_standardized_file_to_openai_sft(
        training_file, openai_file, use_full_response=use_full_response
    )
    file_id = upload_training_file(client, openai_file)
    job_info = submit_sft_job(
        client, file_id, model,
        n_epochs=n_epochs, batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier, suffix=suffix,
    )
    job_id = job_info["id"]
    print(f"OpenAI SFT job submitted: {job_id}")
    if output_dir is not None:
        save_job_info(job_info, Path(output_dir))

    final = poll_job(
        client, job_id, output_dir=output_dir, poll_schedule_s=poll_schedule_s
    )
    return TrainResult(model=final["fine_tuned_model"], resume_handle=None, info=final)


@dataclass
class SFTJobConfig:
    training_file: str
    model: str = "gpt-4.1-2025-04-14"
    n_epochs: int | str = "auto"
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    suffix: str = ""
    validation_file: str | None = None
    output_dir: str | None = None


def main():
    config = tyro.cli(SFTJobConfig)
    client = get_client()

    output_dir = Path(config.output_dir) if config.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(config), indent=2, default=str))

    file_id = upload_training_file(client, Path(config.training_file))
    val_file_id = upload_training_file(client, Path(config.validation_file)) if config.validation_file else None

    job_info = submit_sft_job(
        client, file_id, config.model,
        n_epochs=config.n_epochs, batch_size=config.batch_size,
        learning_rate_multiplier=config.learning_rate_multiplier,
        suffix=config.suffix, validation_file_id=val_file_id,
    )
    print(f"Job submitted: {job_info['id']}")

    if output_dir:
        save_job_info(job_info, output_dir)

    return job_info


if __name__ == "__main__":
    main()
