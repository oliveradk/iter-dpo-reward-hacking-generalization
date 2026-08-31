"""Stage 1 driver: run any registered inspect `@task`.

The inspect eval log IS the output: the training envs' `extract_thinking`
solver normalizes inline `<think>`/`<thinking>` tags into a native
`ContentReasoning` block at generation time, so the log needs no further
parsing. `run_generate` returns the final log's location (possibly an
`s3://` URI) and writes a local pointer file `<out>/eval_log.json` —
the select stage reads the log via
`rewardhacking_training.select.log_records.records_from_eval_log`.
"""

from __future__ import annotations

import inspect as inspect_module
import json
import os
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


def default_inspect_log_dir(out: Path) -> str:
    """Inspect log dir for the (local) experiment dir `out`.

    When `S3_OUTPUT_ROOT` is set (e.g. `s3://bucket/char-misspecified-rl`),
    return a native `s3://` URI mirroring `out`'s repo-relative path so
    inspect writes logs to S3 directly via fsspec (only the logs go to S3 —
    the rest of the experiment dir stays local). Without `S3_OUTPUT_ROOT`,
    falls back to the local `out/inspect_logs`.

    Uses the logical path (symlinks not resolved) so a symlinked experiment
    dir like `experiments/<name>/output/<sub>` keeps its repo-relative path.
    """
    s3_root = os.environ.get("S3_OUTPUT_ROOT", "").rstrip("/")
    if not s3_root:
        return str(out / "inspect_logs")
    try:
        rel = Path(os.path.abspath(out)).relative_to(os.getcwd())
    except ValueError:
        rel = Path(out.name)
    return f"{s3_root}/{rel.as_posix()}/inspect_logs"


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
    """Forwarded to `inspect_ai.eval(model_args=...)`; used only when the
    inference client returns no model_args of its own."""
    inference_client: InferenceClientConfig = field(
        default_factory=InferenceClientConfig
    )
    """How a string `model` is served for sampling: the provider + per-provider
    knobs (tinker base/renderer, modal server URL/auth + adapter lifecycle).
    `run_generate` builds an `InferenceClient` from this, calls `.start(model)`
    to resolve the `(model, model_args)` inspect uses, and `.end()` afterwards
    (the modal arm waits for the vLLM server, loads/unloads the LoRA adapter)."""
    include_reasoning: bool = False
    """When serving a NATIVE reasoning model (separate reasoning channel),
    include its reasoning trace as a `ContentReasoning` block on the sampled
    message. Only drives the tinker sampling client's `include_reasoning` —
    downstream, the reasoning lands as a `ContentReasoning` block in the log
    either way (natively via this flag, or via the envs' `extract_thinking`
    solver for tag-emitting models), so the select stage doesn't branch on
    it. Default False — most models (openai, vLLM-served) emit inline tags;
    set True to keep a native reasoning model's separate trace (e.g. the
    tinker arm)."""


@dataclass
class GenerateConfig:
    task: str = ""
    """Task spec of the form `pkg.module:task_fn` (e.g.
    `rewardhacking_training.envs.impossible_mbpp.impossible_mbpp_env:impossible_mbpp`)."""
    model: Any = "openai/gpt-4.1"
    """Either an inspect-ai model string or a pre-constructed
    `inspect_ai.model.Model` instance. A string is resolved through
    `model_config.inference_client.InferenceClient.start` per its `provider`
    (openai id, tinker base/`tinker://` URI, Together HF repo id, or a modal
    served-model name); a `Model` instance is passed through unchanged."""
    model_config: ModelConfig = field(default_factory=ModelConfig)
    """Serving + parsing identity for `model`: the inference-client
    provider/knobs, `model_args`, and the reasoning parser. Factored out so the
    iterative-training loop applies one `ModelConfig` across iterations while
    overriding only `model` (the per-iteration checkpoint)."""
    n_samples: int = 5
    system_prompts_path: str | None = None
    task_args: dict[str, str] = field(default_factory=dict)
    """Extra kwargs for the task fn (`--task-args key value key value …`).
    String values are JSON-coerced at run time (`_coerce_task_arg`), so
    `--task-args max_tokens 1536 persona_only true` arrives typed."""
    log_dir: str | None = None
    """Explicit inspect log dir (local path or fsspec URI like `s3://…`).
    Overrides `s3_logs`. When None, defaults to `<out>/inspect_logs` (or its
    S3 mirror when `s3_logs=True`)."""
    s3_logs: bool = True
    """Write inspect logs natively to S3 instead of the local
    `<out>/inspect_logs`. Uses `default_inspect_log_dir`, which
    mirrors the experiment dir's path under `S3_OUTPUT_ROOT` (falls back to
    the local `<out>/inspect_logs` when that env var is unset). Disable with
    `--no-s3-logs` for a fully local run."""
    max_connections: int | None = 1000
    """Forwarded to `inspect_ai.eval(max_connections=...)`: max concurrent
    model API requests. 1000 is a HARD CEILING with the stock provider client:
    inspect's OpenAI-compatible providers use the openai-SDK default httpx
    pool (`max_connections=1000`), and requests queued beyond the pool trigger
    an httpcore waiter-scan CPU collapse (100% CPU, no requests dispatched —
    measured in `experiments/2026-07-08_client_concurrency_test/`). Going
    higher requires passing an enlarged `http_client` through `model_args`;
    `run_generate` refuses values above 1000 until that plumbing exists.
    Modal-side context for the 1000 default: ~5 replicas' worth of load
    (`max_inputs=200` each), safely inside Modal's 2,000-pending-inputs cap
    and the 50-GPU Team quota."""
    max_samples: int | None = 2000
    """Forwarded to `inspect_ai.eval(max_samples=...)`: max samples run in
    parallel. Inspect defaults this to `max_connections`; we set it to 2×
    so sandbox/scoring work on finished requests overlaps with in-flight
    generation rather than throttling concurrency to the connection cap."""
    attempt_timeout: int | None = 480
    """Forwarded to inspect's generate config: seconds before a single
    request ATTEMPT is abandoned and retried (per `retry_on_error`). Without
    it a hung request only dies at the openai client's 600s total timeout —
    measured cost: one straggler out of 5,628 requests held an otherwise-done
    eval for ~9 minutes (848s request; `experiments/2026-07-08_e2e_throughput`
    iter 0). 480s clears the legitimate tail with margin (p99 request time at
    a fully-loaded 1000-connection burst incl. autoscale queuing was ~270s);
    a timed-out attempt just retries against the by-then-warm fleet."""
    limit: int | None = None
    fail_on_error: float | bool = 0.1
    """Forwarded to `inspect_ai.eval(fail_on_error=...)`. A float is the
    tolerated *fraction* of errored samples before the whole task aborts; a
    bool toggles abort-on-first-error. Defaults to tolerating up to 10% so a
    handful of transient provider errors (e.g. a non-retryable HTTP 400 that
    `retry_on_error` won't catch) don't cancel every in-flight sample. The
    errored samples are unscored in the log and skipped by the select-side
    reader (`log_records.records_from_eval_log`)."""
    retry_on_error: int = 3
    """Forwarded to `inspect_ai.eval(retry_on_error=...)` (and the
    `eval_retry` passes). Number of times inspect re-runs an errored sample
    *within* a single eval before giving up on it. The `retry_on_failure`
    pass below is the coarser, log-level recovery that kicks in after this."""
    retry_on_failure: bool = True
    """After the initial eval, if the log status is not `"success"` (some
    samples errored out beyond what `retry_on_error` recovered), call
    `inspect_ai.eval_retry` to re-run only the failed/incomplete samples.
    `eval_retry` *resumes* the task: completed samples are copied forward and
    merged into a fresh combined log — the final log is the one whose
    location `run_generate` returns (and records in `eval_log.json`). Retried
    up to `max_retries_on_failure` times; any samples still errored after
    that are skipped by the select-side reader."""
    max_retries_on_failure: int = 1
    """How many successive `eval_retry` passes to attempt while the log is
    not `"success"`. Each pass only re-runs samples that are still missing."""
    retry_max_connections: int | None = 3
    """`max_connections` for the `eval_retry` passes. Lower than the
    generation default so a flaky endpoint isn't hammered on retry."""


@dataclass
class _GenerateCLI(GenerateConfig):
    # Flatten the nested model-serving knobs onto the top-level CLI
    # (`--provider`, `--base-model`, `--tinker-renderer-name`, `--model-args`,
    # `--reasoning-parser`, …) instead of the `--model-config.*` prefix.
    # `OmitArgPrefixes` cascades into the nested `inference_client`.
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

    Returns the final eval log's location (a local path or an fsspec URI
    like `s3://…`) — the log is the stage's output. Also writes a local
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
    # `log_dir` may be a remote fsspec URI (e.g. `s3://…` when
    # `s3_logs=True`) — inspect handles those natively, so only mkdir
    # local paths.
    if cfg.log_dir:
        log_dir = cfg.log_dir
    elif cfg.s3_logs:
        # Falls back to the local `<out>/inspect_logs` when S3_OUTPUT_ROOT
        # is not set in the environment.
        log_dir = default_inspect_log_dir(out)
    else:
        log_dir = str(out / "inspect_logs")
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
