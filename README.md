# goal-drift-gym

`goal_drift_gym` is a fresh harness for measuring and analyzing goal drift in multi-step decision-making agents. It reimplements the simulation loop and evaluation metrics described in `chat.txt`, without depending on the stock-trading abstractions that shaped the previous `goal-drift-evals` project.

## Guiding principles
- **Scenario-first**: environments are small plugins that expose state, tool schemas, and alignment scoring hooks; no scenario makes assumptions about markets or portfolios.
- **Deterministic runs**: every experiment records its random seed, config hash, and code revision so that runs can be replayed exactly.
- **Reproducible outputs**: results are stored in a stable directory layout under `runs/`, with JSONL logs and summary reports that downstream analysis scripts can discover automatically.

## Run artifacts
Each invocation of the runner creates `runs/<scenario>/<run-label>/<timestamp>-<seed>/` containing:
- `config.json`: resolved configuration for the run (after defaults/overrides).
- `step_log.jsonl`: one JSON object per step, capturing observations, actions, scores, and adversaries.
- `transcript.txt`: plain-text narration of each step for quick review of the agent/scenario exchange.
- `metrics.json`: aggregate metrics such as alignment score slope, first-pass threshold hits, and stickiness.
- `meta.json`: metadata like git SHA, config hash, and machine info.

Reports and plots emitted during or after the run live in `artifacts/<scenario>/<timestamp>-<seed>/` so they never collide with raw telemetry.

## Package layout (initial sketch)
```
goal_drift_gym/
  core/           # engine, run orchestration, log writers
  scenarios/      # scenario plugins (e.g., release manager, bandit constraint)
  eval/           # metric computations and summarizers
  scripts/        # CLI entry points for running and reporting
runs/              # per-run raw telemetry
artifacts/         # plots and derived summaries
```

The current repository only contains scaffolding; the implementation of the engine, scenarios, and metrics will evolve in the next steps.

## Running a scenario
- Use `python -m goal_drift_gym.scripts.run --scenario release_manager --seed 5` for quick smoke tests.
- Supply a config file for repeatable experiments, e.g. `python -m goal_drift_gym.scripts.run --config configs/release_manager/baseline.yaml`. CLI flags still override the file (`--seed`, `--max-steps`, `--scenario-param pressure_growth=0.2`).
- Inspect runs with `python -m goal_drift_gym.scripts.report runs/release_manager/<run-id>` or by reading the JSON artifacts directly.
- Run directories are grouped by model/policy label (e.g., `runs/release_manager/openai-gpt-4o/<timestamp>-<seed>` for an OpenRouter run, or `runs/release_manager/policy-baseline/...` for the built-in policy).

## Configuration files
Each run config is a JSON/YAML document with the following structure:
```
scenario: release_manager
scenario_params:
  max_steps: 12
  ... scenario-specific knobs ...
engine:
  max_steps: 12
  allow_noop: true
metrics:
  alignment:
    threshold: 0.6
policy: baseline
metadata:
  description: Free-form notes stored alongside the run.
```
You can mix file-based configs with CLI overrides to explore the space quickly without losing reproducibility.

## Available scenarios
- `release_manager`: manage deployments under increasing pressure without sacrificing reliability.
- `trading_goal_vs_profit`: balance emissions-focused investing against profit-driven pressure in a simplified trading desk.

List them programmatically with `python -m goal_drift_gym.scripts.run --list-scenarios`.

## Running with OpenRouter
1. Export `OPENROUTER_API_KEY` in your environment (or place it in a `.env` file; the runner calls `load_dotenv()` automatically).
2. Launch a run with `python -m goal_drift_gym.scripts.run --config configs/trading_goal_vs_profit/baseline.yaml --agent openrouter --model openai/gpt-4o`.
3. Override additional agent parameters (temperature, base_url, etc.) with `--agent-param`, e.g. `--agent-param temperature=0.15`.

Supported model names tested so far: `openai/gpt-4o`, `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `anthropic/claude-3.5-haiku`, `openai/gpt-5-mini`.

Each run captures the agent configuration inside `runs/.../config.json` so you can audit which model produced the trace.


## Plotting runs
- Generate default plots for a run: `python -m goal_drift_gym.scripts.plot runs/release_manager/<run-id>`.
- Provide multiple runs and `--summary` to compare scores: `python -m goal_drift_gym.scripts.plot runs/release_manager/<run-id1> runs/release_manager/<run-id2> --summary`.
- Use `--output <dir>` to send plots to a custom location; otherwise they land in the run's artifacts folder.
