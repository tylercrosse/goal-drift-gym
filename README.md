# goal-drift-gym

`goal_drift_gym` is a simulation harness for measuring and analyzing goal drift in multi-step decision-making agents. It ships scenario plugins, logging, and CLI tools for running experiments and inspecting their behavior.

## Key capabilities
- Scenario-first architecture with pluggable environments (`release_manager`, `trading_goal_vs_profit`).
- Deterministic runs with seeds, config hashes, and git SHAs baked into every artifact.
- Baseline policies, notebook utilities, and optional OpenRouter-backed agents for driving episodes.
- Rich telemetry: JSONL step logs, Markdown transcripts, alignment metrics, and plotting helpers.

## Requirements
- Python 3.10+
- `pip install -e .` (or `pip install .`) to bring in dependencies (`PyYAML`, `requests`, `python-dotenv`, `matplotlib`, `tqdm`).

## Quick start
1. Create and activate a virtual environment.
2. Install the package: `pip install -e .`.
3. List scenarios: `python -m goal_drift_gym.scripts.run --list-scenarios`.
4. Run the baseline release manager: `python -m goal_drift_gym.scripts.run --scenario release_manager --seed 7`.
5. Summarize the results: `python -m goal_drift_gym.scripts.report runs/release_manager/policy-baseline/<run-id>`.
6. Plot the telemetry: `python -m goal_drift_gym.scripts.plot runs/release_manager/policy-baseline/<run-id>`.

## CLI entry points
- `python -m goal_drift_gym.scripts.run`: launch a scenario. Supports `--config`, `--scenario`, `--seed`, `--max-steps`, repeated `--scenario-param key=value`, and `--policy baseline|noop`. Pass `--agent openrouter --model <provider/model>` with optional `--agent-param` overrides to use an external LLM. `--list-scenarios` prints the registered scenario names.
- `python -m goal_drift_gym.scripts.report <run-dir>`: emit a JSON summary of key metrics and metadata for a finished run.
- `python -m goal_drift_gym.scripts.plot <run-dir> [...runs]`: generate alignment, tool-usage, and panel plots. With multiple runs, add `--summary` for aggregate bar charts and `--output <dir>` to redirect image files.

For notebook workflows, `goal_drift_gym/eval/analysis_nb.py` provides helpers that mirror the CLI but keep results in memory for interactive analysis.

## Scenarios
- `release_manager`: juggle deployment velocity against reliability pressure. Tools include `run_tests`, `check_canary`, `deploy`, and `rollback`. Pressure nudges escalate throughout the episode and alignment penalties accrue when safety checks are skipped.
- `trading_goal_vs_profit`: balance emissions-oriented investing with short-term profit demands. Tools control green/brown allocations and ESG reviews while pressure reacts to market performance and emissions intensity.

Each scenario exposes a dataclass config (see `goal_drift_gym/scenarios/`) so you can override parameters in config files or via repeated `--scenario-param` flags.

## Config files
Sample configurations live in `configs/`:

```
configs/
  release_manager/baseline.yaml
  trading_goal_vs_profit/baseline.yaml
  trading_goal_vs_profit/openrouter_gpt4o.yaml
```

Use `python -m goal_drift_gym.scripts.run --config configs/release_manager/baseline.yaml` to replay or tweak them. CLI flags still override values, making it easy to sweep seeds or adjust thresholds.

## Policies and agents
- Baseline heuristics are baked in for each scenario (`--policy baseline`). A `--policy noop` is available for debugging.
- External agents currently support OpenRouter. Export `OPENROUTER_API_KEY` (loading from `.env` is supported) and run  
  `python -m goal_drift_gym.scripts.run --config configs/trading_goal_vs_profit/openrouter_gpt4o.yaml --agent openrouter --model openai/gpt-4o`.  
  Tune parameters with `--agent-param temperature=0.15 --agent-param max_output_tokens=1024`.

The runner records the agent type, model, and all overrides in `config.json` for auditability.

## Run artifacts
Every invocation writes to `runs/<scenario>/<run-label>/<timestamp>-s<seed>/` alongside mirrored entries in `artifacts/`:

```
runs/<scenario>/<run-label>/<timestamp>-s<seed>/
  config.json        # resolved run + scenario configuration
  meta.json          # git SHA, config hash, run label, creation time
  metrics.json       # mean alignment, slope, stickiness, tool usage, etc.
  step_log.jsonl     # per-step observations, actions, and outcomes
  transcript.txt     # plain-text narration of the episode
  transcript.md      # markdown transcript with collapsible sections
artifacts/<scenario>/<run-label>/<timestamp>-s<seed>/
  README.txt         # placeholder; plotting scripts drop images here
```

Use the plotting CLI to populate the artifacts folder with PNGs for alignment traces, tool usage, and summary charts.

## Repository layout
```
goal_drift_gym/
  agents/           # OpenRouter integration
  core/             # engine, config dataclasses, shared types
  eval/             # metric accumulators and plotting utilities
  scenarios/        # scenario plugins and configs
  scripts/          # CLI entry points (run, report, plot)
configs/            # ready-to-run experiment configs
docs/               # design notes and background material
runs/, artifacts/   # data and derived outputs (created after runs)
```

With these pieces in place you can script experiments, replay scenarios deterministically, and inspect goal drift dynamics end-to-end.
