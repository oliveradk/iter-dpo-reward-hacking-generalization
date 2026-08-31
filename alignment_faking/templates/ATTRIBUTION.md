# Attribution

The jinja templates in this directory are vendored from
[safety-research/open-source-alignment-faking](https://github.com/safety-research/open-source-alignment-faking)
(MIT License, Copyright (c) 2025 Safety Research), at upstream paths:

| file here                                  | upstream path                                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `helpful-only_prompt-only_cot.jinja2`      | `prompts/system_prompts/helpful-only_prompt-only_cot.jinja2`                                   |
| `reasoning_traces.jinja2`                  | `prompts/system_prompts/examples/reasoning_traces.jinja2`                                      |
| `free_tier_cot.jinja2`                     | `prompts/instruction_formats/free_tier_formats/cot.jinja2`                                     |
| `paid_tier_cot.jinja2`                     | `prompts/instruction_formats/paid_tier_formats/cot.jinja2`                                     |
| `strong_reject.jinja2`                     | `prompts/classifiers/helpful_only_classifiers/strong_reject.jinja2`                            |
| `deceptive_alignment_majority_vote_v1.jinja` | `prompts/classifiers/scratchpad_classifiers/new_classifiers/deceptive_alignment_majority_vote_v1.jinja` |
| `self-preservation_majority_vote_v1.jinja` | `prompts/classifiers/scratchpad_classifiers/new_classifiers/self-preservation_majority_vote_v1.jinja` |
| `partial_compliance_v1.jinja`              | `prompts/classifiers/scratchpad_classifiers/new_classifiers/partial_compliance_v1.jinja`       |

Files are byte-identical to upstream commit `07929d2`, except
`helpful-only_prompt-only_cot.jinja2`, whose `{% include %}` path is rewritten
to point at the flat `reasoning_traces.jinja2` in this directory.

The classifier orchestration/parsing logic ported from that repo's
`src/pipeline/evals.py` lives in `../classifiers.py`.
