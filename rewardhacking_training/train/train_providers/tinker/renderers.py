"""Custom tinker-cookbook renderers, resolved by name through the cookbook's public
registry (`register_renderer`), which both halves of the loop consult: the sampling
client (`InspectAPIFromTinkerSampling`) and the DPO/SFT dataset builders.

Call `register_renderers()` before `get_renderer` runs in a process; it is idempotent.
"""

from __future__ import annotations

from tinker_cookbook.renderers import is_renderer_registered, register_renderer
from tinker_cookbook.renderers.base import Message, RenderContext, RenderedMessage, TextPart
from tinker_cookbook.renderers.qwen3 import Qwen3InstructRenderer

QWEN3_INSTRUCT_THINKING = "qwen3_instruct_thinking"


class Qwen3InstructThinkingTagRenderer(Qwen3InstructRenderer):
    """`qwen3_instruct` whose trained reasoning is spelled `<thinking>...</thinking>`.

    The stock renderer writes a ThinkingPart as the Qwen `<think>` special token, while the
    non-reasoning Instruct-2507 models are *prompted* for `<thinking>` tags (the literal
    `<think>` token is out of distribution for them). Rendering the trained completion with
    the same plain-text tag keeps sampling and training consistent. Everything else (headers,
    tool calls, `parse_response`) is inherited; a sampled `<thinking>` block is plain text to
    the parser, so it stays inline for the envs' `extract_thinking` solver.
    """

    open_tag = "<thinking>"
    close_tag = "</thinking>"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        content = message["content"]
        if isinstance(content, list):
            parts = content
            if self.strip_thinking_from_history and message["role"] == "assistant" and not ctx.is_last:
                parts = [p for p in parts if p["type"] != "thinking"]
            parts = [
                TextPart(type="text", text=f"{self.open_tag}{p['thinking']}{self.close_tag}")
                if p["type"] == "thinking"
                else p
                for p in parts
            ]
            message = {**message, "content": parts}
        return super().render_message(message, ctx)


def register_renderers() -> None:
    if not is_renderer_registered(QWEN3_INSTRUCT_THINKING):
        register_renderer(
            QWEN3_INSTRUCT_THINKING,
            lambda tokenizer, image_processor=None: Qwen3InstructThinkingTagRenderer(tokenizer),
        )
