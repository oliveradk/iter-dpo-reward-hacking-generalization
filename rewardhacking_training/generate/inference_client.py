"""Unified inference-client lifecycle for the generation stage.

`InferenceClient` resolves the `(model, model_args)` pair that
`inspect_ai.eval` consumes and — for the Modal-served providers — manages the
vLLM server readiness + LoRA-adapter load/unload around the eval. It unifies
two things that used to live apart:

* the pure string/`Model` resolution that used to live in a standalone
  inspect-model builder, and
* the Modal vLLM adapter lifecycle (ready → load adapter → unload), using the
  primitives in `modal.modal_utils.inference_utils`,

behind a single `start()` / `end()` interface.

A "provider" selects how the current model is served for sampling:

* `openai`   — a bare/`ft:` OpenAI model id, prefixed with `openai/`.
* `tinker`   — an `inspect_ai.model.Model` built against a
  `tinker.SamplingClient` (resumed from a `tinker://` URI when one is given,
  else created against the base model).
* `together` — an HF repo id served through inspect's native `together/<name>`
  serverless inference provider.
* `modal`    — a vLLM served-model name on the per-model
  `char-vllm-inference-<slug>` web server. `start()` waits for the server, loads the
  current iteration's LoRA adapter when `model` is a `modal-lora:<path>` (else
  serves the base model), points inspect at the lora name via
  `openai-api/modal-vllm/<name>` (setting `MODAL_VLLM_*`), and `end()` unloads
  the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


def wrap_truncated_native_reasoning(output: Any) -> bool:
    """Rewrap a max_tokens-truncated choice from a NATIVE reasoning model
    whose message carries no `ContentReasoning` block: the generation was cut
    off mid-reasoning (the chat template opens the think block implicitly, so
    the renderer's parser finds no closing token and returns the partial trace
    as plain text). The whole text becomes the reasoning block and the answer
    is empty — the standard truncation convention (cf.
    `train_env_utils.split_unclosed_reasoning` for tag-emitting models).

    Mutates the (single-choice) `output`'s message in place — and syncs
    `output.completion`, a stored field, to the now-empty answer — returning
    True when a rewrap happened. No-ops (False) when a reasoning block is
    already present (cut off mid-ANSWER — the partial answer text is kept) or
    the text is empty."""
    from inspect_ai.model import ContentReasoning

    msg = output.choices[0].message
    content = msg.content
    if isinstance(content, list) and any(
        isinstance(c, ContentReasoning) for c in content
    ):
        return False
    text = msg.text
    if not text.strip():
        return False
    msg.content = [ContentReasoning(reasoning=text)]
    output.completion = ""
    return True


def _to_openai_model(name: str) -> str:
    """inspect_ai requires `<api>/<model>`. OpenAI base + fine-tuned names
    come through bare (`gpt-4.1...`, `ft:gpt-4.1...`); prefix `openai/` so
    inspect routes them correctly."""
    if "/" in name:
        return name
    return f"openai/{name}"


@dataclass
class InferenceClientConfig:
    """Static configuration for an `InferenceClient`.

    The per-iteration model identifier itself is NOT here — it is passed to
    `InferenceClient.start(model)` because it changes between calls (a
    `modal-lora:<path>`, a `tinker://` URI, an OpenAI id, …)."""

    provider: Literal["openai", "tinker", "together", "modal"] = "openai"
    base_model: str | None = None
    """Tinker base HF id; on modal, selects the per-model app/URL. Unused for
    openai/together."""
    tinker_model_name: str | None = None
    """Overrides `base_model` for tinker's sampling client when set."""
    tinker_renderer_name: str | None = None
    modal_inference_url: str | None = None
    """Explicit vLLM root URL; None derives it from `base_model` or env."""
    modal_api_key: str | None = None
    """Bearer token; None reads MODAL_VLLM_API_KEY."""
    modal_unload_adapter: bool = True
    modal_ready_timeout_s: int | None = None
    """Seconds; None uses the modal default."""


class InferenceClient:
    """Resolve + manage the generation model for one eval.

    Usage:

        client = InferenceClient(
            cfg.model_config.inference_client,
            include_reasoning=cfg.model_config.include_reasoning,
        )
        model, model_args = client.start(cfg.model)
        try:
            inspect_ai.eval(..., model=model, model_args=model_args)
        finally:
            client.end()
    """

    def __init__(
        self, cfg: InferenceClientConfig, *, include_reasoning: bool = False
    ):
        self.cfg = cfg
        # Whether to keep a native reasoning model's separate trace — on the
        # tinker path this becomes the sampling client's `include_reasoning`.
        # Lives on `ModelConfig.include_reasoning`; passed in here because the
        # client itself only carries the `InferenceClientConfig`.
        self.include_reasoning = include_reasoning
        # Modal lifecycle state captured in start(), consumed in end().
        self._modal_base_url: str | None = None
        self._modal_api_key: str | None = None
        self._modal_served_name: str | None = None
        self._modal_adapter_path: str | None = None

    def start(self, model: Any) -> tuple[Any, dict]:
        """Resolve `(model, model_args)` for `inspect_ai.eval`, connecting to /
        starting the serving backend as needed.

        `model` is the current generation model identifier (an OpenAI id, a
        tinker base HF id / `tinker://` URI, a Together HF repo id, or a modal
        `modal-lora:<path>` / base id). A pre-constructed `inspect_ai.model.
        Model` is returned unchanged."""
        # Already a concrete inspect Model instance — nothing to build.
        if not isinstance(model, str):
            return model, {}

        provider = self.cfg.provider
        if provider == "openai":
            return _to_openai_model(model), {}
        if provider == "together":
            # Generate from `model` (the HF repo id) through Together's stock
            # serverless inference provider.
            name = (
                model.split("/", 1)[1] if model.startswith("together/") else model
            )
            return f"together/{name}", {}
        if provider == "modal":
            return self._start_modal(model)
        if provider == "tinker":
            return self._start_tinker(model)
        raise ValueError(f"unknown provider: {provider!r}")

    def end(self) -> None:
        """Tear down per-eval serving state. A no-op for every provider except
        modal, where it unloads the LoRA adapter loaded in `start()`
        (when `modal_unload_adapter` is set)."""
        if self.cfg.provider != "modal":
            return
        if (
            self._modal_adapter_path
            and self.cfg.modal_unload_adapter
            and self._modal_served_name
        ):
            from rewardhacking_training.modal.modal_utils.inference_utils import (
                unload_adapter,
            )

            unload_adapter(
                self._modal_base_url, self._modal_api_key, self._modal_served_name
            )

    # ---- provider-specific start paths -------------------------------------

    def _start_modal(self, model: str) -> tuple[Any, dict]:
        import os

        from rewardhacking_training.modal.modal_apps.common import (
            BASE_SERVED_MODEL_NAME,
            inference_app_name,
            strip_lora_prefix,
        )
        from rewardhacking_training.modal.modal_utils.inference_utils import (
            DEFAULT_READY_TIMEOUT_S,
            ensure_server_ready,
            get_api_key,
            get_server_base_url,
            load_adapter,
            openai_v1_url,
        )

        cfg = self.cfg
        base_url = get_server_base_url(cfg.modal_inference_url, cfg.base_model or None)
        api_key = get_api_key(cfg.modal_api_key)
        app_name = inference_app_name(cfg.base_model) if cfg.base_model else None
        timeout = (
            cfg.modal_ready_timeout_s
            if cfg.modal_ready_timeout_s is not None
            else DEFAULT_READY_TIMEOUT_S
        )
        ensure_server_ready(base_url, api_key, timeout, app_name=app_name)

        adapter_path = strip_lora_prefix(model) if model else None
        if adapter_path:
            served = load_adapter(
                base_url, api_key, adapter_path,
                app_name=app_name, ready_timeout_s=timeout,
            )
        else:
            served = BASE_SERVED_MODEL_NAME

        # inspect's openai-api provider reads MODAL_VLLM_BASE_URL/_API_KEY for
        # `openai-api/modal-vllm/<name>` (service segment `modal-vllm`).
        os.environ["MODAL_VLLM_BASE_URL"] = openai_v1_url(base_url)
        os.environ["MODAL_VLLM_API_KEY"] = api_key

        self._modal_base_url = base_url
        self._modal_api_key = api_key
        self._modal_served_name = served
        self._modal_adapter_path = adapter_path
        return f"openai-api/modal-vllm/{served}", {}

    def _start_tinker(self, model: str) -> tuple[Any, dict]:
        # Defer imports so the openai/together/modal paths don't pull in tinker.
        import tinker
        from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
        from inspect_ai.model import Model as InspectAIModel
        from tinker_cookbook.eval.inspect_utils import (
            InspectAPIFromTinkerSampling,
        )

        class _TinkerSamplingAPI(InspectAPIFromTinkerSampling):
            """`InspectAPIFromTinkerSampling` hardcodes `stop_reason="stop"` on
            every choice, making max_tokens cut-offs invisible downstream —
            the select stage's `drop_length_truncated` machinery and the
            rows' informational `truncated` flag both key on the stop
            reason. Relabel a single-choice completion that consumed the full
            `max_tokens` budget as `"max_tokens"` (inspect's normalized value
            for a request-cap cut-off). Multi-choice outputs are left alone —
            usage is aggregated across choices so per-choice attribution isn't
            possible; this pipeline always samples one choice per request
            (n_samples parallelism comes from inspect epochs).

            With `include_reasoning` (native reasoning model), a completion
            cut off MID-REASONING has no closing think token, so the
            renderer's parser finds no reasoning part and the partial trace
            lands as answer-channel TEXT (the model's chat template opens the
            think block implicitly — there is no tag in the sampled text to
            recover from). Rewrap such truncated no-reasoning-block choices as
            all-reasoning (`wrap_truncated_native_reasoning`), matching the
            standard truncation convention (reasoning=<tail>, answer=\"\")."""

            async def generate(self, input, tools, tool_choice, config):
                output = await super().generate(input, tools, tool_choice, config)
                if (
                    config.max_tokens
                    and output.usage
                    and len(output.choices) == 1
                    and output.usage.output_tokens >= config.max_tokens
                ):
                    output.choices[0].stop_reason = "max_tokens"
                    if self.include_reasoning:
                        wrap_truncated_native_reasoning(output)
                return output

        cfg = self.cfg
        base = cfg.tinker_model_name or cfg.base_model
        if base is None:
            raise ValueError(
                "tinker provider requires base_model or tinker_model_name"
            )
        service_client = tinker.ServiceClient()
        if model.startswith("tinker://"):
            sampling_client = service_client.create_sampling_client(
                model_path=model,
            )
        else:
            sampling_client = service_client.create_sampling_client(
                base_model=base,
            )
        api = _TinkerSamplingAPI(
            renderer_name=cfg.tinker_renderer_name or "llama3",
            model_name=base,
            sampling_client=sampling_client,
            # For a reasoning model, emit a ContentReasoning block so the trace
            # survives to the eval log (the select-side reader
            # `log_records.records_from_eval_log` reads it). Otherwise the
            # renderer's `<think>` parser misses inline `<thinking>`, so keep
            # tags inline in the completion and let the envs'
            # `extract_thinking` solver normalize them.
            include_reasoning=self.include_reasoning,
        )
        return InspectAIModel(api=api, config=InspectAIGenerateConfig()), {}


# ---- CLI (for shell pipelines, e.g. bash eval loops over checkpoints) ----

def main() -> None:
    """Resolve/serve a model from the shell and print `export` lines.

    Usage:

        # Load a checkpoint's adapter (modal) and capture the inspect routing:
        eval "$(python -m rewardhacking_training.generate.inference_client \
            start modal-lora:adapters/<tag> \
            --provider modal --base-model Qwen/Qwen2.5-32B-Instruct \
            | grep '^export ')"
        inspect eval <task> --model "$INSPECT_EVAL_MODEL" ...

        # Unload it afterwards:
        python -m rewardhacking_training.generate.inference_client \
            stop modal-lora:adapters/<tag> \
            --provider modal --base-model Qwen/Qwen2.5-32B-Instruct

    `start` runs `InferenceClient.start(model)` (waiting for / preparing the
    serving backend, loading the LoRA adapter on the modal paths) and prints
    shell-exportable assignments: `INSPECT_EVAL_MODEL` (the inspect model
    string; inspect reads this env var natively) plus, on the modal paths,
    `MODAL_VLLM_BASE_URL` / `MODAL_VLLM_API_KEY`. Pipe through `grep '^export '`
    before `eval` — the underlying helpers also log progress to stdout.
    Providers that return a non-string `Model` (tinker) are not supported here.
    """
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("command", choices=["start", "stop"])
    ap.add_argument("model", help="model id (e.g. modal-lora:adapters/<tag>, a bare HF/OpenAI id)")
    ap.add_argument("--provider", default="modal",
                    choices=["openai", "together", "modal"])
    ap.add_argument("--base-model", default=None,
                    help="base model (selects the per-model modal app/URL)")
    args = ap.parse_args()

    client = InferenceClient(InferenceClientConfig(
        provider=args.provider,
        base_model=args.base_model,
        modal_unload_adapter=(args.command == "stop"),
    ))
    inspect_model, _model_args = client.start(args.model)
    if not isinstance(inspect_model, str):
        raise SystemExit(f"provider {args.provider!r} is not supported by this CLI")
    if args.command == "stop":
        client.end()
        print(f"unloaded: {args.model}")
        return
    print(f"export INSPECT_EVAL_MODEL={inspect_model}")
    for var in ("MODAL_VLLM_BASE_URL", "MODAL_VLLM_API_KEY"):
        if os.environ.get(var):
            print(f"export {var}={os.environ[var]}")


if __name__ == "__main__":
    main()
