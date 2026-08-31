# Data-generator config files

A *training data generator* bundles the two pipeline stages (generate ->
select training samples) into one config. Generators are loaded from JSON
config files via `load_generator(path)` — there is no in-process registry.

A config file is a partial **overlay** over an optional `base`:

```jsonc
{
  "name": "nl_gameable_negative_ojudge",   // optional; defaults to the base's name
  "base": "nl_gameable.base.json",         // optional; path relative to THIS file, else cwd
  "generate": { /* GenerateConfig fields to override */ },
  "select":   { /* SelectConfig fields to override */ }
}
```

Resolution (see `../base.py`):

- `base` is loaded first (recursively). With no `base`, the overlay starts
  from `TrainingDataGenerator()` defaults — so a complete spec works too.
- `generate` / `select` are **deep-merged** onto the base's stage configs, so
  you only restate the fields you change. `task_args` and the nested
  `inference_client` merge key-by-key instead of being clobbered.
- The select stage is a single `SelectConfig`; `select.mode`
  (`"dpo"`/`"sft"`) picks the per-prompt rule. (The iterative-training runner's
  run-level `--method` overrides `mode`, so the same generator works for both.)

## Bases shipped here

Each base sets only its `task`, system-prompt bank, and `select.mode`
(`"dpo"`). All other selection knobs (`n`, filters, inoculation,
`score_threshold`, …) default and are set per-overlay.

| file | env | notes |
|------|-----|-------|
| `impossible_mbpp.base.json` | impossible_mbpp | thinking-variants bank |
| `nl_gameable.base.json` | nl_gameable | thinking-variants bank |
| `codecontests.base.json` | codecontests | fixed `system_prompt_key="no_hints"`, no bank solver |

## Worked overlays

Put your own overlays anywhere (e.g. inside an experiment dir) and point the
runner at the path.

Negative user-message inoculation + the nl_gameable pairwise output filter:

```json
{
  "name": "nl_gameable_negative_ojudge",
  "base": "nl_gameable.base.json",
  "select": {
    "inoculation_bank_path": "rewardhacking_training/prompts/inoculation_prompts/nl_gameable_negative.json",
    "output_pairwise_filters": ["nl_gameable"]
  }
}
```

impossible_mbpp with the reasoning-format (CoT) pairwise filter + a full-hack
preferred-score floor:

```json
{
  "name": "impossible_mbpp_negative_reasoning_filter_passall",
  "base": "impossible_mbpp.base.json",
  "select": {
    "inoculation_bank_path": "rewardhacking_training/prompts/inoculation_prompts/impossible_mbpp_negative.json",
    "reasoning_pairwise_filters": ["impossible_mbpp"],
    "score_threshold": 1.0
  }
}
```

System-prompt inoculation block at generation time (rides into the data) +
persona-only bank prompt for a native reasoning model:

```json
{
  "name": "nl_gameable_persona_dont_hack",
  "base": "nl_gameable.base.json",
  "generate": {
    "task_args": { "persona_only": true, "inoculation": "dont_hack" }
  }
}
```

An SFT generator off the same base (flip `mode`; `n` is the top-n kept per
prompt):

```json
{
  "name": "nl_gameable_sft",
  "base": "nl_gameable.base.json",
  "select": {
    "mode": "sft",
    "n": 2,
    "score_threshold": 0.9
  }
}
```

## Running

```bash
# end-to-end single generator
python -m rewardhacking_training.generate_and_select.generate_and_select \
    rewardhacking_training/data_generators/configs/nl_gameable.base.json --generate.n-samples 8

# iterative training (paths in the --generator list; per-gen n_prompts/np,
# epoch_fraction/frac overrides apply; completions-per-prompt is generate.n_samples)
python -m rewardhacking_training.iterative_training.iterative_training \
    --run-name v2 --base-model gpt-4.1-2025-04-14 \
    --generator path/to/impossible_mbpp_negative.json:np=80 path/to/nl_gameable_negative.json:frac=0.25
```
