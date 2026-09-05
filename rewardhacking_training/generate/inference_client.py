from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


def wrap_truncated_native_reasoning(output: Any) -> bool:
    """A max_tokens-truncated choice from a NATIVE reasoning model with no
    `ContentReasoning` block was cut mid-reasoning (the chat template opens the think
    block implicitly, so no closing token is found): the whole text becomes reasoning
    and the answer empties. Mutates `output` in place (syncing `output.completion`);
    returns True on rewrap, False if a reasoning block exists or the text is empty.
    """
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
    """inspect_ai needs `<api>/<model>`; bare OpenAI / `ft:` ids get the `openai/` prefix.
    """
    if "/" in name:
        return name
    return f"openai/{name}"


@dataclass
class InferenceClientConfig:
    """The per-iteration model id is NOT here — it is passed to
    `InferenceClient.start(model)` since it changes between calls.
    """

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
    """`start(model)` returns `(model, model_args)` for `inspect_ai.eval`; call `end()` in
    a `finally` to tear down serving state.
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
        """`model` is the per-iteration identifier (OpenAI id, tinker HF id / `tinker://`
        URI, Together repo id, or modal `modal-lora:<path>` / base id); a pre-built
        `Model` is returned unchanged.
        """
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
        """No-op except for modal, where it unloads the adapter loaded in `start()` (when
        `modal_unload_adapter` is set).
        """
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

        from rewardhacking_training.train.train_providers.tinker.renderers import register_renderers

        register_renderers()

        class _TinkerSamplingAPI(InspectAPIFromTinkerSampling):
            """Upstream hardcodes `stop_reason="stop"`, hiding max_tokens cut-offs from the
            select stage; relabel a single-choice completion that used the full budget
            as `"max_tokens"` (multi-choice usage is aggregated, so left alone). With
            `include_reasoning`, a choice cut mid-reasoning lands as plain text — rewrap
            it as all-reasoning via `wrap_truncated_native_reasoning`.
            """

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
    """Print shell-exportable `INSPECT_EVAL_MODEL` (+ `MODAL_VLLM_*` on modal) for `start`,
    or unload the adapter for `stop`. Pipe through `grep '^export '` before `eval` — the
    helpers also log to stdout. Non-string `Model` providers (tinker) are unsupported.
    """
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description="Resolve/serve a model from the shell and print `export` lines.")
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
