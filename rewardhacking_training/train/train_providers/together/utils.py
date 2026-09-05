from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from together import Together

from rewardhacking_training.train.train_providers.types import TrainResult


def get_client() -> Together:
    load_dotenv()
    return Together()  # picks up TOGETHER_API_KEY from os.environ


def upload_training_file(client: Together, file_path: Path) -> str:
    result = client.files.upload(file=str(file_path), purpose="fine-tune")
    return result.id


def save_job_info(job_info: dict, output_dir: Path) -> None:
    (Path(output_dir) / "job_info.json").write_text(
        json.dumps(job_info, indent=2, default=str)
    )


_TERMINAL_STATES = {"completed", "error", "cancelled"}

DEFAULT_POLL_SCHEDULE_S: tuple[int, ...] = (300, 600, 1200, 2400)
"""Successive sleep durations between poll attempts; plateaus at the last."""


def poll_job(
    client: Together,
    job_id: str,
    *,
    output_dir: Path | None = None,
    poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
) -> TrainResult:
    """Raises `RuntimeError` unless `completed`; result `model` is the job's `x_model_output_name`, `resume_handle` the job id (for `from_checkpoint`)."""
    poll_idx = 0
    while True:
        job = client.fine_tuning.retrieve(job_id)
        status = job.status
        print(f"Job {job_id}: {status}")
        if status in _TERMINAL_STATES:
            final = job.model_dump()
            if output_dir is not None:
                (Path(output_dir) / "job_final.json").write_text(
                    json.dumps(final, indent=2, default=str)
                )
            if status != "completed":
                raise RuntimeError(
                    f"Together job {job_id} ended with status={status}"
                )
            output_model = final.get("x_model_output_name")
            if not output_model:
                raise RuntimeError(
                    f"Together job {job_id} completed but has no "
                    f"x_model_output_name: {final!r}"
                )
            return TrainResult(
                model=output_model,
                resume_handle=job_id,
                info=final,
            )
        wait = poll_schedule_s[min(poll_idx, len(poll_schedule_s) - 1)]
        poll_idx += 1
        print(f"  sleeping {wait}s before next poll")
        time.sleep(wait)


def reattach(
    info: dict, output_dir: Path, poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
) -> TrainResult:
    """Await the job recorded in `job_info.json` instead of submitting a new one."""
    print(f"re-attaching to Together job {info['id']}")
    return poll_job(get_client(), info["id"], output_dir=output_dir, poll_schedule_s=poll_schedule_s)
