"""Modal apps: TRL DPO training (`modal_dpo`, app `char-dpo-train-<slug>`),
TRL SFT training (`modal_sft`, app `char-sft-train-<slug>`), and vLLM inference
(`modal_inference`, app `char-vllm-inference-<slug>`). Deployed via
`modal deploy`; see specs/MODAL_INTEGRATION.md. The DPO + SFT training apps
share the `char-dpo` volume and the single inference app (a LoRA adapter is
served the same way regardless of how it was trained).

Shared constants (volume name, app names, paths) live in `common.py` so the
client side (`trainers/modal_dpo.py`, `trainers/modal_sft.py`,
`modal_utils/`) can import them without pulling in the app modules.
"""
