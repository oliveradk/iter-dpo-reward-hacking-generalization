"""The iterative-training state machine (DPO or SFT).

`iterative_training.py` holds the user-facing config + CLI
(`IterativeTrainingConfig` + `GeneratorSpec`); `state.py` / `step.py` are the
resumable state machine; `generate_and_select.py` is the per-iteration
generate → select phase (each generator samples a fixed prompt slice of its
dataset via a cross-iteration cursor and runs `generate_and_select`); `train.py`
is the per-iteration `train` phase (`do_train` builds this iteration's
`TrainConfig` and calls the standalone `pessimistic_training.train.train.run_train`).
"""
