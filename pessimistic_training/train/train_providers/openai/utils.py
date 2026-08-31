"""Shared utilities for OpenAI fine-tuning jobs (client, upload, blocking poll).

Used by both ``openai.dpo`` and ``openai.sft`` — the only thing they don't
share is the ``method`` block of the submit call and the ``TrainResult``
mapping, so everything else (client, file upload, job-info persistence, and the
blocking poll loop) lives here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv


def get_client() -> openai.OpenAI:
    load_dotenv()
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def upload_training_file(client: openai.OpenAI, file_path: Path) -> str:
    with open(file_path, "rb") as f:
        result = client.files.create(file=f, purpose="fine-tune")
    return result.id


def save_job_info(job_info: dict, output_dir: Path) -> None:
    (Path(output_dir) / "job_info.json").write_text(
        json.dumps(job_info, indent=2, default=str)
    )


DEFAULT_POLL_SCHEDULE_S: tuple[int, ...] = (1200, 2400, 4800, 7200)
"""Successive sleep durations between poll attempts. Once the schedule is
exhausted we keep polling at the last value (7200s = 2h) until terminal."""


def poll_job(
    client: openai.OpenAI,
    job_id: str,
    *,
    output_dir: Path | None = None,
    poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
) -> dict:
    """Block until an OpenAI fine-tuning job reaches a terminal status.

    Walks ``poll_schedule_s`` between retrievals (plateauing at the last value),
    writes ``job_final.json`` into ``output_dir`` on completion when supplied,
    and raises ``RuntimeError`` on any status other than ``succeeded`` (the
    step-machine halt contract). Returns the final ``Job.model_dump()``.
    """
    poll_idx = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"Job {job_id}: {status}")
        if status in ("succeeded", "failed", "cancelled"):
            final = job.model_dump()
            if output_dir is not None:
                (Path(output_dir) / "job_final.json").write_text(
                    json.dumps(final, indent=2, default=str)
                )
            if status != "succeeded":
                raise RuntimeError(f"OpenAI job {job_id} ended with status={status}")
            return final
        wait = poll_schedule_s[min(poll_idx, len(poll_schedule_s) - 1)]
        poll_idx += 1
        print(f"  sleeping {wait}s before next poll")
        time.sleep(wait)
