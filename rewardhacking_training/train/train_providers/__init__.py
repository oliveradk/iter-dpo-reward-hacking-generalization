"""Training backends for the iterative-training pipeline (DPO or SFT).

One subpackage per provider (``openai/``, ``together/``, ``tinker/``,
``modal/``), each with a ``dpo``/``sft`` module and a per-provider ``utils``.
``types`` is the ONLY module shared across providers: the ``TrainResult`` return
type, the standardized DPO/SFT row format, and the small helpers each backend
uses to read that format. Each backend exposes a blocking ``train_dpo`` and/or
``train_sft`` entrypoint that runs a single job to completion and returns a
``TrainResult`` describing how the next iteration should refer to the model.

DPO backends (``train_dpo``):

* ``openai.dpo`` — uploads the training file, submits an OpenAI fine-tuning
  job, polls until terminal status, returns the fine-tuned model id (stateless:
  ``resume_handle=None``).
* ``tinker.dpo`` — wraps ``tinker_cookbook.preference.train_dpo`` (itself
  blocking) and parses the final checkpoint record.
* ``together.dpo`` — converts to the Together/OpenAI preference format,
  uploads, submits a ``training_method="dpo"`` job, polls until terminal, and
  returns ``model=x_model_output_name`` + ``resume_handle=<job id>``.
* ``modal.dpo`` — runs TRL (``train_dpo``) or ms-swift/Megatron
  (``train_dpo_swift``) LoRA training on the per-model Modal app; returns
  ``model=modal-lora:<path>`` + ``resume_handle=<adapter path>`` (LoRA-on-base
  continuation).

SFT backends (``train_sft``):

* ``openai.sft`` — OpenAI supervised fine-tuning (analog of ``openai.dpo``).
* ``modal.sft`` — TRL (``train_sft``) / ms-swift (``train_sft_swift``) LoRA
  training on the per-model Modal app (analog of ``modal.dpo``).

The selection stage (``rewardhacking_training.select``) writes the standardized
DPO/SFT JSONL directly via ``types.write_standardized_file`` (no dependency on
any provider trainer module); each backend converts that file to its native
format at train time.
"""

from rewardhacking_training.train.train_providers.types import TrainResult

__all__ = ["TrainResult"]
