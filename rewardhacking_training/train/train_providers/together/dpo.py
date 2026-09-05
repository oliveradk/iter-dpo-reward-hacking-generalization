from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from together import Together

from rewardhacking_training.train.train_providers.together.utils import (
    DEFAULT_POLL_SCHEDULE_S,
    get_client,
    poll_job,
    save_job_info,
    upload_training_file,
)
from rewardhacking_training.train.train_providers.types import (
    TrainResult,
    assistant_content,
    convert_file,
    write_train_result,
)

# ---- standardized-format -> preference-format conversion ---------------

def standardized_pair_to_preference(row: dict, *, use_full_response: bool = True) -> dict:
    """Rows already in preference format (list-valued `*_output`) pass through unchanged."""
    pref, dispref = row["preferred_output"], row["non_preferred_output"]
    if isinstance(pref, list):  # already preference-format
        return row
    return {
        "input": row["input"],
        "preferred_output": [
            {"role": "assistant", "content": assistant_content(pref, use_full_response)}
        ],
        "non_preferred_output": [
            {"role": "assistant", "content": assistant_content(dispref, use_full_response)}
        ],
    }


def convert_standardized_file_to_preference(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    return convert_file(
        in_path, out_path,
        lambda r: standardized_pair_to_preference(r, use_full_response=use_full_response),
    )


def _normalize_batch_size(batch_size: int | str) -> int | str:
    """Together accepts an int or the literal `"max"`; any other string (e.g. OpenAI's `"auto"`) maps to `"max"`."""
    if isinstance(batch_size, int):
        return batch_size
    return "max"


def submit_dpo_job(
    client: Together,
    training_file_id: str,
    model: str | None,
    *,
    beta: float = 0.1,
    n_epochs: int = 1,
    batch_size: int | str = "max",
    learning_rate: float = 1e-5,
    lora: bool = True,
    lora_r: int | None = None,
    lora_alpha: float | None = None,
    lora_trainable_modules: str | None = None,
    rpo_alpha: float | None = None,
    simpo_gamma: float | None = None,
    dpo_normalize_logratios_by_length: bool = False,
    n_checkpoints: int = 1,
    from_checkpoint: str | None = None,
    suffix: str = "",
    wandb_project_name: str | None = None,
    wandb_name: str | None = None,
) -> dict:
    """With `from_checkpoint`, `model` must be None (Together infers the base). `lora_trainable_modules` matters
    for MoE bases: the default `all-linear` attaches LoRA to expert projections some vLLM builds can't serve."""
    kwargs: dict = dict(
        training_file=training_file_id,
        training_method="dpo",
        n_epochs=n_epochs,
        batch_size=_normalize_batch_size(batch_size),
        learning_rate=learning_rate,
        lora=lora,
        dpo_beta=beta,
        dpo_normalize_logratios_by_length=dpo_normalize_logratios_by_length,
        n_checkpoints=n_checkpoints,
    )
    if model is not None:
        kwargs["model"] = model
    if from_checkpoint is not None:
        kwargs["from_checkpoint"] = from_checkpoint
    if lora_r is not None:
        kwargs["lora_r"] = lora_r
    if lora_alpha is not None:
        kwargs["lora_alpha"] = lora_alpha
    if lora_trainable_modules is not None:
        kwargs["lora_trainable_modules"] = lora_trainable_modules
    if rpo_alpha is not None:
        kwargs["rpo_alpha"] = rpo_alpha
    if simpo_gamma is not None:
        kwargs["simpo_gamma"] = simpo_gamma
    if suffix:
        kwargs["suffix"] = suffix
    if wandb_project_name:
        kwargs["wandb_project_name"] = wandb_project_name
        if wandb_name:
            kwargs["wandb_name"] = wandb_name
        # Together doesn't read the env var server-side; without an explicit
        # key the job runs but logs nothing to W&B.
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            kwargs["wandb_api_key"] = api_key

    job = client.fine_tuning.create(**kwargs)
    return job.model_dump()


def train_dpo(
    *,
    training_file: Path,
    model: str | None,
    beta: float = 0.1,
    n_epochs: int = 1,
    batch_size: int | str = "max",
    learning_rate: float = 1e-5,
    lora: bool = True,
    lora_r: int | None = None,
    lora_alpha: float | None = None,
    lora_trainable_modules: str | None = None,
    rpo_alpha: float | None = None,
    simpo_gamma: float | None = None,
    dpo_normalize_logratios_by_length: bool = False,
    n_checkpoints: int = 1,
    from_checkpoint: str | None = None,
    suffix: str = "",
    wandb_project_name: str | None = None,
    wandb_name: str | None = None,
    output_dir: Path | None = None,
    poll_schedule_s: tuple[int, ...] = DEFAULT_POLL_SCHEDULE_S,
    client: Together | None = None,
    use_full_response: bool = True,
) -> TrainResult:
    """Set `from_checkpoint` (and leave `model=None`) to continue from a prior job. Raises `RuntimeError` unless the job ends `completed`."""
    client = client or get_client()
    training_file = Path(training_file)
    together_file = training_file.with_suffix(".together.jsonl")
    # Convert + upload + submit are the deterministic-failure surface (bad
    # format, auth, unservable base, network). The step machine halts only
    # on RuntimeError, so wrap any SDK/IO exception into one — otherwise a
    # submit-time error crashes the whole iterative loop instead of halting
    # the run with a recorded reason.
    try:
        convert_standardized_file_to_preference(
            training_file, together_file, use_full_response=use_full_response
        )
        file_id = upload_training_file(client, together_file)
        job_info = submit_dpo_job(
            client, file_id, model,
            beta=beta, n_epochs=n_epochs, batch_size=batch_size,
            learning_rate=learning_rate, lora=lora, lora_r=lora_r,
            lora_alpha=lora_alpha, lora_trainable_modules=lora_trainable_modules,
            rpo_alpha=rpo_alpha, simpo_gamma=simpo_gamma,
            dpo_normalize_logratios_by_length=dpo_normalize_logratios_by_length,
            n_checkpoints=n_checkpoints, from_checkpoint=from_checkpoint,
            suffix=suffix, wandb_project_name=wandb_project_name,
            wandb_name=wandb_name,
        )
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize to the step-machine contract
        raise RuntimeError(f"Together DPO submission failed: {e}") from e
    job_id = job_info["id"]
    print(f"Together DPO job submitted: {job_id}")
    if output_dir is not None:
        save_job_info(job_info, Path(output_dir))

    return poll_job(
        client, job_id, output_dir=output_dir, poll_schedule_s=poll_schedule_s
    )


# ---- CLI ---------------------------------------------------------------

@dataclass
class TogetherDPOJobConfig:
    training_file: str
    model: str | None = "meta-llama/Llama-3.2-3B-Instruct"
    beta: float = 0.1
    n_epochs: int = 1
    batch_size: int | str = "max"
    learning_rate: float = 1e-5
    lora: bool = True
    lora_r: int | None = None
    lora_alpha: float | None = None
    rpo_alpha: float | None = None
    simpo_gamma: float | None = None
    dpo_normalize_logratios_by_length: bool = False
    n_checkpoints: int = 1
    from_checkpoint: str | None = None
    suffix: str = ""
    wandb_project_name: str | None = None
    wandb_name: str | None = None
    output_dir: str | None = None
    use_full_response: bool = True


def main():
    import tyro
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(TogetherDPOJobConfig)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(cfg), indent=2, default=str)
        )
    result = train_dpo(
        training_file=Path(cfg.training_file),
        model=None if cfg.from_checkpoint else cfg.model,
        beta=cfg.beta,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        lora=cfg.lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        rpo_alpha=cfg.rpo_alpha,
        simpo_gamma=cfg.simpo_gamma,
        dpo_normalize_logratios_by_length=cfg.dpo_normalize_logratios_by_length,
        n_checkpoints=cfg.n_checkpoints,
        from_checkpoint=cfg.from_checkpoint,
        suffix=cfg.suffix,
        wandb_project_name=cfg.wandb_project_name,
        wandb_name=cfg.wandb_name,
        output_dir=output_dir,
        use_full_response=cfg.use_full_response,
    )
    print(
        f"Together DPO done: model={result.model} "
        f"resume_handle={result.resume_handle}"
    )
    if output_dir:
        write_train_result(result, output_dir)
    return result


if __name__ == "__main__":
    main()
