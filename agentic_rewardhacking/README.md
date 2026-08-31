# agentic_rewardhacking

Agentic (multi-turn, sandboxed) reward-hacking evals, as opposed to the
single-turn evals in `rewardhacking_evals/`:

1. **`bash_chess/`** — inspect_ai port of Palisade's ctfish "beat Stockfish"
   env; lives in this repo. See [`bash_chess/README.md`](bash_chess/README.md).
2. **EvilGenie** — LiveCodeBench coding tasks in a docker workspace where the
   agent can hack the visible tests; lives in a separate checkout (install +
   run instructions below).

## EvilGenie

EvilGenie ([paper](https://arxiv.org/abs/2511.21654), upstream repo:
[JonathanGabor/evilgenie_inspect](https://github.com/JonathanGabor/evilgenie_inspect)) gives an
agent a docker workspace with `problem.md`, an empty `solution.py`, visible
`test_cases.json`, and a `test.py` runner. Hacking opportunities are deliberate
(hardcoding visible test cases, editing the test files); detection combines
holdout tests, an LLM judge (`legitimate` / `heuristic` / `reward_hacking`),
and SHA256 file-modification checks.

We run the **inspect_ai port** (`evilgenie_inspect`), which drives an inspect
`basic_agent` over the API model under test — which is what our fine-tuned
checkpoints need. A local checkout lives at
`../char-misspecified-rl/evilgenie_inspect/`; the upstream is
[JonathanGabor/evilgenie_inspect](https://github.com/JonathanGabor/evilgenie_inspect).
It is a self-contained uv project with its own lockfile — don't install it
into this repo's venv.

### Install

Prerequisites: Docker running, `uv`, and API keys (`OPENAI_API_KEY` for agent +
judge; the port defaults the judge to `openai/gpt-5`).

```bash
cd ../char-misspecified-rl/evilgenie_inspect
uv venv && uv sync
cp ~/projects/.env .env   # or set keys in the shell
```

(If you don't have the char-misspecified-rl checkout, clone the upstream repo
instead: `git clone https://github.com/JonathanGabor/evilgenie_inspect` and run
the same commands from its root.)

### Run

From `evilgenie_inspect/`:

```bash
uv run inspect eval reward_hacking.py --model openai/<model> \
    --max-connections 5 \
    --log-dir <this-repo>/output/evilgenie/<run-name>
```

Useful task params (`-T name=value`):

- `difficulty` (`easy`/`medium`/`hard`, default `hard`) and
  `dataset_source` (`livecodebench` default, or `apps`)
- `limit` — number of problems (default 10)
- `problem_ids='["lcb_abc374_a", ...]'` — pin an exact problem set
- `ambiguous_problems_only=true` — the 8 predefined ambiguous problems where
  hacking is most prevalent
- `enable_llm_judge=false` — skip the judge (holdout tests + file checks only)
- `--model-role judge=<model>` — override the judge model

View results with `uv run inspect view`. Scores are categorical dicts per
sample: `test_result` (`passed_all` / `passed_visible_only` / `failed_visible`),
`llm_classification`, and `files_modified`.

### Operational notes (from past long runs)

Hard-won lessons from
`../char-misspecified-rl/experiments/2026-08-06_evilgenie_it02_allhard/`:

- Chunk large problem sets into ~20-problem cells with separate `--log-dir`s;
  one long run rarely survives to inspect's final flush.
- Keep `--max-connections` ≤ 5 locally — higher caused memory-pressure kills.
- A crashed pass leaves its sandboxes running; reap before retrying:
  `docker ps -aq --filter "name=inspect-reward_hacki" | xargs -r docker rm -f`
- Resume incomplete logs with `uv run inspect eval-retry <log.eval>` in a loop
  until the log reports `status=success`.
