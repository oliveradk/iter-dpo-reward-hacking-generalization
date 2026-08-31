"""Client-side helpers for the Modal inference server: server URL/auth
resolution and dynamic LoRA load/unload. The generation-stage adapter
lifecycle is driven by `generate.inference_client.InferenceClient` (the modal
arm), which uses these primitives.
"""
