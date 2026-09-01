"""Jupyter (ipywidgets) viewer for DPO pair files.

Notebook counterpart of ``utils.browse_dpo``: prompt at top, dispreferred on
the left, preferred on the right, with the same two view modes ("full" shows
the stored ``full_response`` verbatim, "split" renders
``<think>reasoning</think>response`` from the separate fields) and the same
yellow marker on a side that was cut off at the generation token cap.

Usage in a notebook cell::

    from utils.dpo_widget import dpo_pairs_widget
    dpo_pairs_widget("path/to/dpo_data.jsonl")
"""

from __future__ import annotations

import html
from pathlib import Path

import ipywidgets as widgets

from utils.browse_dpo import extract_parts, load_examples

_BOX_STYLE = (
    "font-family: var(--jp-code-font-family, monospace); font-size: 12px;"
    "white-space: pre-wrap; word-break: break-word;"
    "border: 1px solid var(--jp-border-color2, #ccc); border-radius: 4px;"
    "padding: 8px; margin: 2px; overflow-y: auto;"
)


def _pane(title: str, color: str, text: str, truncated: bool, max_height: str) -> str:
    warn = (
        "<div style='color:#b58900;font-weight:bold'>&#9888; TRUNCATED (max_tokens)</div>"
        if truncated
        else ""
    )
    return (
        f"<div style='{_BOX_STYLE} max-height:{max_height};'>"
        f"<div style='color:{color};font-weight:bold'>{title}</div>{warn}"
        f"{html.escape(text)}</div>"
    )


def dpo_pairs_widget(
    source: str | Path | list[dict],
    start: int = 0,
    prompt_height: str = "220px",
    pair_height: str = "420px",
) -> widgets.Widget:
    """Interactive side-by-side viewer for DPO pairs.

    ``source`` is a standardized DPO JSONL path (as written by
    ``rewardhacking_training.select``) or an already-loaded list of examples.
    Returns the widget; display it by making it the last expression of a cell.
    """
    if isinstance(source, (str, Path)):
        label = str(source)
        examples = load_examples(str(source))
    else:
        label = f"{len(source)} in-memory examples"
        examples = source
    if not examples:
        return widgets.HTML("<i>No DPO examples found.</i>")

    idx = widgets.BoundedIntText(
        value=min(start, len(examples) - 1),
        min=0,
        max=len(examples) - 1,
        description="pair",
        layout=widgets.Layout(width="160px"),
    )
    prev_btn = widgets.Button(description="◀ prev", layout=widgets.Layout(width="80px"))
    next_btn = widgets.Button(description="next ▶", layout=widgets.Layout(width="80px"))
    # Native-reasoning rows store no inline full_response — start those in
    # split view so the panes aren't empty.
    first_pref = examples[0].get("preferred_output")
    has_full = isinstance(first_pref, dict) and bool(first_pref.get("full_response"))
    mode = widgets.ToggleButtons(
        options=["full", "split"],
        value="full" if has_full else "split",
        layout=widgets.Layout(width="auto"),
        style={"button_width": "60px"},
    )
    counter = widgets.HTML()
    prompt_html = widgets.HTML()
    left_html = widgets.HTML(layout=widgets.Layout(width="50%"))
    right_html = widgets.HTML(layout=widgets.Layout(width="50%"))

    def render(*_):
        prompt, (pref, pref_trunc), (dispref, dispref_trunc) = extract_parts(
            examples[idx.value], mode.value
        )
        counter.value = (
            f"&nbsp;<b>[{idx.value + 1}/{len(examples)}]</b>&nbsp;"
            f"<span style='color:#888'>{html.escape(label)}</span>"
        )
        prompt_html.value = _pane("PROMPT", "#268bd2", prompt, False, prompt_height)
        left_html.value = _pane("DISPREFERRED", "#dc322f", dispref, dispref_trunc, pair_height)
        right_html.value = _pane("PREFERRED", "#859900", pref, pref_trunc, pair_height)

    prev_btn.on_click(lambda _: setattr(idx, "value", max(idx.value - 1, 0)))
    next_btn.on_click(lambda _: setattr(idx, "value", min(idx.value + 1, len(examples) - 1)))
    idx.observe(render, names="value")
    mode.observe(render, names="value")
    render()

    controls = widgets.HBox([prev_btn, next_btn, idx, mode, counter])
    return widgets.VBox([controls, prompt_html, widgets.HBox([left_html, right_html])])
