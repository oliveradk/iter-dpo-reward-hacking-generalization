"""OpenAI DPO trainer.

Exposes two surfaces:

* ``train_dpo(...)`` — blocking entrypoint used by the training driver. Uploads
  the training file, submits a fine-tuning job, polls until terminal status,
  and returns a ``TrainResult`` carrying the fine-tuned model id.
* ``format_dpo_pair`` / ``write_dpo_file`` / ``submit_dpo_job`` /
  ``DPOJobConfig`` / ``main`` — helpers for one-shot jobs submitted via
  ``python -m`` (CLI).
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


# ---- formatting helpers ------------------------------------------------

def format_dpo_pair(
    user_message: str,
    preferred: str,
    non_preferred: str,
    system_prompt: str | None = None,
) -> dict:
    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_message})
    return format_dpo_pair_messages(input_messages, preferred, non_preferred)


def format_dpo_pair_messages(
    input_messages: list[dict], preferred: str, non_preferred: str,
) -> dict:
    """OpenAI-format DPO row from a prebuilt prompt `input_messages` list and
    the two assistant-content strings."""
    return {
        "input": {"messages": input_messages},
        "preferred_output": [{"role": "assistant", "content": preferred}],
        "non_preferred_output": [{"role": "assistant", "content": non_preferred}],
    }


def write_dpo_file(pairs: list[dict], output_path: Path, shuffle: bool = False, seed: int = 42):
    if shuffle:
        pairs = list(pairs)
        random.Random(seed).shuffle(pairs)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


# ---- standardized-format -> OpenAI-format conversion ------------------

def standardized_pair_to_openai(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized DPO row (see ``types``) to the OpenAI fine-tuning
    DPO format. Rows already in OpenAI format (list-valued ``*_output``) pass
    through unchanged for backward compatibility."""
    pref, dispref = row["preferred_output"], row["non_preferred_output"]
    if isinstance(pref, list):  # already OpenAI-format
        return row
    return format_dpo_pair_messages(
        input_messages=row["input"]["messages"],
        preferred=assistant_content(pref, use_full_response),
        non_preferred=assistant_content(dispref, use_full_response),
    )


def convert_standardized_file_to_openai(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized DPO JSONL and write the OpenAI-format equivalent.
    Returns `out_path`."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_pair_to_openai(r, use_full_response=use_full_response),
    )


# ---- low-level submit (used by the CLI + blocking train_dpo) ----------

def submit_dpo_job(
    client: openai.OpenAI,
    training_file_id: str,
    model: str,
    beta: float = 0.1,
    n_epochs: int = 1,
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
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": beta,
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

def train_dpo(
    *,
    training_file: Path,
    model: str,
    beta: float = 0.1,
    n_epochs: int = 1,
    batch_size: int | str = "auto",
    learning_rate_multiplier: float | str = "auto",
    suffix: str = "",
    output_dir: Path | None = None,
    poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
    client: openai.OpenAI | None = None,
    use_full_response: bool = True,
) -> TrainResult:
    """Submit + poll an OpenAI DPO job, blocking until terminal status.

    ``training_file`` is a standardized DPO JSONL (see ``types``); it is
    converted to the OpenAI fine-tuning format before upload. The assistant
    content defaults to each pair's ``full_response`` (reasoning inline); set
    ``use_full_response=False`` to train on the answer only.

    The job_info / final job payload are written into ``output_dir`` when
    supplied (``job_info.json`` on submit, ``job_final.json`` on completion).
    Raises ``RuntimeError`` if the job ends in any state other than
    ``succeeded``.
    """
    client = client or get_client()
    training_file = Path(training_file)
    openai_file = training_file.with_suffix(".openai.jsonl")
    convert_standardized_file_to_openai(
        training_file, openai_file, use_full_response=use_full_response
    )
    file_id = upload_training_file(client, openai_file)
    job_info = submit_dpo_job(
        client, file_id, model,
        beta=beta, n_epochs=n_epochs, batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier, suffix=suffix,
    )
    job_id = job_info["id"]
    print(f"OpenAI DPO job submitted: {job_id}")
    if output_dir is not None:
        save_job_info(job_info, Path(output_dir))

    final = poll_job(
        client, job_id, output_dir=output_dir, poll_schedule_s=poll_schedule_s
    )
    return TrainResult(model=final["fine_tuned_model"], resume_handle=None, info=final)


# ---- CLI ---------------------------------------------------------------

@dataclass
class DPOJobConfig:
    training_file: str
    model: str = "gpt-4.1-2025-04-14"
    beta: float = 0.1
    n_epochs: int = 1
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    suffix: str = ""
    validation_file: str | None = None
    output_dir: str | None = None
    use_full_response: bool = True
    """Assistant content source when converting the standardized training
    file to OpenAI format: full_response (reasoning inline) vs response only."""


def main():
    config = tyro.cli(DPOJobConfig)
    client = get_client()

    output_dir = Path(config.output_dir) if config.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(config), indent=2, default=str))

    train_path = Path(config.training_file)
    openai_path = train_path.with_suffix(".openai.jsonl")
    convert_standardized_file_to_openai(
        train_path, openai_path, use_full_response=config.use_full_response
    )
    file_id = upload_training_file(client, openai_path)
    val_file_id = upload_training_file(client, Path(config.validation_file)) if config.validation_file else None

    job_info = submit_dpo_job(
        client, file_id, config.model,
        beta=config.beta, n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        learning_rate_multiplier=config.learning_rate_multiplier,
        suffix=config.suffix, validation_file_id=val_file_id,
    )
    print(f"Job submitted: {job_info['id']}")

    if output_dir:
        save_job_info(job_info, output_dir)

    return job_info


if __name__ == "__main__":
    main()
