# goal-drift-gym

`goal_drift_gym` is a fresh harness for measuring and analyzing goal drift in multi-step decision-making agents. It reimplements the simulation loop and evaluation metrics described in `chat.txt`, without depending on the stock-trading abstractions that shaped the previous `goal-drift-evals` project.

## Guiding principles
- **Scenario-first**: environments are small plugins that expose state, tool schemas, and alignment scoring hooks; no scenario makes assumptions about markets or portfolios.
- **Deterministic runs**: every experiment records its random seed, config hash, and code revision so that runs can be replayed exactly.
- **Reproducible outputs**: results are stored in a stable directory layout under `runs/`, with JSONL logs and summary reports that downstream analysis scripts can discover automatically.

## Run artifacts
Each invocation of the runner creates `runs/<scenario>/<timestamp>-<seed>/` containing:
- `config.json`: resolved configuration for the run (after defaults/overrides).
- `step_log.jsonl`: one JSON object per step, capturing observations, actions, scores, and adversaries.
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
