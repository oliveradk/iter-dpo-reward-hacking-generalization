"""Stage 1 driver: run any registered inspect `@task`.

The inspect eval log IS the output: the training envs' `extract_thinking`
solver normalizes inline `<think>`/`<thinking>` tags into a native
`ContentReasoning` block at generation time, so the log needs no further
parsing. `run_generate` returns the final log's location and writes a
local pointer file `<out>/eval_log.json` —
the select stage reads the log via
`rewardhacking_training.select.log_records.records_from_eval_log`.
"""

from __future__ import annotations

import inspect as inspect_module
import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any

import tyro
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry

from rewardhacking_training.constants import EVAL_LOG_FILENAME
from rewardhacking_training.generate.inference_client import (
    InferenceClient,
    InferenceClientConfig,
)
from rewardhacking_training.utils import make_experiment_dir


@dataclass
class ModelConfig:
    """How a model is served for sampling.

    Factors the model-serving identity out of `GenerateConfig` so the
    "which checkpoint" identifier (`GenerateConfig.model` — which varies
    per iteration in the iterative-training loop) stays separate from the
    static serving identity (the inference-client provider + per-provider
    knobs and `model_args`). The iterative-training loop applies one
    `ModelConfig` uniformly across iterations while overriding only `model`."""

    model_args: dict[str, Any] = field(default_factory=dict)
    """Used only when the inference client returns no model_args of its own."""
    inference_client: InferenceClientConfig = field(
        default_factory=InferenceClientConfig
    )
    """Provider + per-provider serving knobs used to resolve a string `model`."""
    include_reasoning: bool = False
    """Set True to keep a native reasoning model's separate trace (tinker only)."""


@dataclass
class GenerateConfig:
    task: str = ""
    """Task spec of the form `pkg.module:task_fn`."""
    model: Any = "openai/gpt-4.1"
    """Inspect model string (resolved via the inference client) or a `Model`."""
    model_config: ModelConfig = field(default_factory=ModelConfig)
    n_samples: int = 5
    system_prompts_path: str | None = None
    task_args: dict[str, str] = field(default_factory=dict)
    """String values are JSON-coerced at run time, so `1536`/`true` arrive typed."""
    log_dir: str | None = None
    """Local path or fsspec URI; None defaults to `<out>/inspect_logs`."""
    max_connections: int | None = 1000
    """Max concurrent model API requests; 1000 is a hard ceiling (the stock
    httpx pool livelocks beyond it)."""
    max_samples: int | None = 2000
    """Set above `max_connections` so scoring overlaps in-flight generation."""
    attempt_timeout: int | None = 480
    """Seconds before a single request attempt is abandoned and retried."""
    limit: int | None = None
    fail_on_error: float | bool = 0.1
    """Float: tolerated fraction of errored samples; bool: abort on first error."""
    retry_on_error: int = 3
    """Times an errored sample is re-run within a single eval."""
    retry_on_failure: bool = True
    """Run `eval_retry` on failed/incomplete samples when the log isn't `"success"`."""
    max_retries_on_failure: int = 1
    retry_max_connections: int | None = 3
    """`max_connections` for the `eval_retry` passes."""


@dataclass
class _GenerateCLI(GenerateConfig):
    # Flatten the nested serving knobs onto the top-level CLI (no
    # `--model-config.*` prefix).
    model_config: Annotated[
        ModelConfig, tyro.conf.OmitArgPrefixes
    ] = field(default_factory=ModelConfig)
    output_dir: str | None = None
    metadata: dict | None = None


def _resolve_task(spec: str):
    mod_path, fn_name = spec.split(":")
    return getattr(import_module(mod_path), fn_name)


def _coerce_task_arg(value: Any) -> Any:
    """CLI task args arrive as strings; JSON-decode the scalar types the task
    signatures actually take (`1536` → int, `true` → bool, quoted strings →
    str) so `task_fn(**task_args)` sees typed values — the same convention as
    `inspect eval -T`. Non-JSON text stays a plain string."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def run_generate(cfg: GenerateConfig, out: Path | None = None) -> str:
    """Run the inspect eval described by `cfg`.

    Returns the final eval log's location — the log is the stage's
    output. Also writes a local
    pointer file `<out>/eval_log.json` (`{"location", "status"}`) so a run
    dir is self-describing (the select stage resolves it via
    `--input-path <run_dir>`). When `out` is None, a fresh
    `output/generate_train/<timestamp>/` dir is created (the legacy CLI
    behavior); when supplied, artifacts go into `out` directly so the
    caller controls the layout (used by the data-generator driver).
    """
    if not cfg.task:
        raise ValueError("cfg.task is required, e.g. 'pkg.module:task_fn'")
    if cfg.max_connections is not None and cfg.max_connections > 1000:
        # The stock provider httpx pool caps at 1000 connections; excess
        # waiters don't just queue — they livelock the client event loop
        # (see GenerateConfig.max_connections and
        # experiments/2026-07-08_client_concurrency_test/results.md).
        raise ValueError(
            f"max_connections={cfg.max_connections} exceeds the stock httpx "
            "pool (1000); requests beyond the pool livelock the client. Pass "
            "an enlarged http_client via model_args before raising this cap."
        )
    if out is None:
        out = make_experiment_dir(cfg, "generate_train")
    else:
        out.mkdir(parents=True, exist_ok=True)
    # An explicit `log_dir` may be a remote fsspec URI — inspect handles
    # those natively, so only mkdir local paths.
    log_dir = cfg.log_dir or str(out / "inspect_logs")
    if "://" not in log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    task_args = {k: _coerce_task_arg(v) for k, v in cfg.task_args.items()}
    if cfg.system_prompts_path and "system_prompts_path" not in task_args:
        task_args["system_prompts_path"] = cfg.system_prompts_path

    task_fn = _resolve_task(cfg.task)
    # Tasks outside the training envs (e.g. the capabilities evals) don't take
    # an `n_samples` epochs param — only pass it where the signature has it.
    if "n_samples" in inspect_module.signature(task_fn).parameters:
        task_args.setdefault("n_samples", cfg.n_samples)
    task_obj = task_fn(**task_args)

    # Resolve `cfg.model` to the `(model, model_args)` pair inspect_ai wants
    # and bring up the serving backend (the modal arm waits for the vLLM server
    # and loads the LoRA adapter). A pre-built `Model` passes through. `end()`
    # tears it down (modal adapter unload) once the eval — including retries —
    # finishes.
    client = InferenceClient(
        cfg.model_config.inference_client,
        include_reasoning=cfg.model_config.include_reasoning,
    )
    model, model_args = client.start(cfg.model)
    try:
        logs = inspect_eval(
            task_obj,
            model=model,
            model_args=model_args or cfg.model_config.model_args or {},
            log_dir=str(log_dir),
            max_connections=cfg.max_connections,
            max_samples=cfg.max_samples,
            attempt_timeout=cfg.attempt_timeout,
            limit=cfg.limit,
            retry_on_error=cfg.retry_on_error,
            fail_on_error=cfg.fail_on_error,
        )
        log = logs[0]

        # A non-"success" status means inspect couldn't complete every sample
        # (e.g. transient provider errors `retry_on_error` didn't recover).
        # `eval_retry` resumes the task in place — completed samples are copied
        # forward, only the failed/incomplete ones re-run — and writes a new
        # combined log we then read from. Old failed logs are left in `log_dir`
        # but nothing downstream globs it; the returned location (recorded in
        # `eval_log.json`) is the final log's.
        if cfg.retry_on_failure:
            attempt = 0
            while log.status != "success" and attempt < cfg.max_retries_on_failure:
                attempt += 1
                print(
                    f"WARN: inspect log status={log.status} "
                    f"(retry {attempt}/{cfg.max_retries_on_failure}) — "
                    f"running eval_retry on {log.location}"
                )
                # `eval_retry` reloads the task to resume it. If the log records
                # a `task_file`, inspect loads that file *standalone*, which
                # breaks the package-relative imports our train_envs tasks use
                # (e.g. `from .language_envs import ...` -> ModuleNotFoundError).
                # The task is already imported/registered in this process
                # (run_generate resolved it via `_resolve_task`), so clear
                # `task_file` to force the registry-name resolution path
                # (`task_registry_name`) instead.
                log.eval.task_file = None
                log = eval_retry(
                    log,
                    log_dir=str(log_dir),
                    max_connections=cfg.retry_max_connections,
                    retry_on_error=cfg.retry_on_error,
                    fail_on_error=cfg.fail_on_error,
                )[0]
            if log.status != "success":
                print(
                    f"WARN: inspect log status={log.status} after {attempt} "
                    f"eval_retry pass(es) — exporting partial samples"
                )
    finally:
        client.end()

    (out / EVAL_LOG_FILENAME).write_text(json.dumps({
        "location": log.location,
        "status": log.status,
    }, indent=2))
    print(f"Eval log: {log.location} (status={log.status})")
    return log.location


def main():
    load_dotenv()
    cfg = tyro.cli(_GenerateCLI)
    run_generate(cfg)


if __name__ == "__main__":
    main()
