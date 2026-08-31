"""Shared experiment pipeline helpers.

Extracts the per-experiment boilerplate that was being copied between
`experiments/*` dirs for the standard pipeline:

1. iterative DPO (run via `rewardhacking_training.iterative_training`)
2. training-curve extraction/plots from the run dirs
   (`training_curves`, `plot_training_curves`)
3. eval batteries over checkpoints, each resumable per (checkpoint, cell):
   - OOD specification-gaming evals (`run_specgaming_evals`):
     impossible_apps / short_gameable / toy_reward, with and without the
     env-specific don't-spec-game suffix, optionally under an eval-time
     inoculation block
   - misalignment evals (`run_misalignment_evals`): frame_colleague, betley,
     alignment_questions, goals, monitor_disruption
   - the TRAIN distribution re-run as an eval (`run_train_distribution`):
     the impossible_mbpp / nl_gameable training envs over the full dataset,
     n_samples=1, under per-condition system-prompt banks (with / without the
     trained-in inoculation persona)
4. plotting CLIs consuming those layouts (`plot_specgaming`,
   `plot_misalignment`, `plot_train_distribution`), each accepting an
   arbitrary number of checkpoints per eval.

Log-layout convention written by the runners (and expected by the plotters):
``<output_dir>/<checkpoint_label>/<cell>/*.eval`` where cells are named
``<env>_<instr>[_inoc_<name>]`` (specgaming),
``mis_<eval>[_deploy][_inoc_<name>]`` (misalignment) and
``<env>_<condition>`` (train distribution; a generate-stage run dir with an
``eval_log.json`` pointer rather than a bare log dir).
"""
