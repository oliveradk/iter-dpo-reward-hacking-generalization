"""Shared utilities for TogetherAI fine-tuning jobs.

A `get_client` that reads `TOGETHER_API_KEY` from the environment (falling back
to a local `.env`), a training-file uploader, and a `save_job_info` writer.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from together import Together


def get_client() -> Together:
    """Together client; reads TOGETHER_API_KEY from env (or a local .env)."""
    load_dotenv()
    return Together()  # picks up TOGETHER_API_KEY from os.environ


def upload_training_file(client: Together, file_path: Path) -> str:
    """Upload a fine-tuning JSONL and return its file id."""
    result = client.files.upload(file=str(file_path), purpose="fine-tune")
    return result.id


def save_job_info(job_info: dict, output_dir: Path) -> None:
    (Path(output_dir) / "job_info.json").write_text(
        json.dumps(job_info, indent=2, default=str)
    )
