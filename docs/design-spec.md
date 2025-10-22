# Goal Drift Gym — Design Specification

**Status:** Draft  
**Owners:** goal-drift-gym maintainers  
**Last updated:** 2025-10-12

## 1. Purpose

Goal Drift Gym provides a focused harness for studying how decision-making agents behave under evolving objectives and external pressure. This design spec captures the guiding goals, architectural pillars, and near-term roadmap so contributors can extend the system coherently.

## 2. Guiding Goals
- **Scenario-first modularity:** Scenarios are self-contained plugins that expose the same observation/action API while remaining agnostic to any domain-specific abstractions (finance, devops, etc.).
- **Deterministic experimentation:** Every run must encode seeds, config hashes, git SHAs, and reproducible telemetry so drift investigations can be replayed byte-for-byte.
- **Transparent behavior auditing:** Capture step logs, transcripts, and metrics that explain *why* an agent drifted, not just that it did.
- **Evaluation flexibility:** Support local heuristics, baseline policies, external LLM agents, and third-party evaluation stacks (e.g., Inspect + Petri) behind a consistent interface.
- **Research ergonomics:** Keep configuration, CLI tooling, and notebook helpers simple enough that researchers can script sweeps or ad-hoc probes without modifying core engine code.
- **Extensible metrics:** Allow alignment metrics and telemetry sinks to evolve without breaking historical data.

## 3. System Architecture

```
configs/ → RunConfig loader → Engine ─┬─ Scenario plugin (Release Manager, Trading, …)
                                      │
                                      ├─ Policy / Agent (baseline heuristics, OpenRouter, future adapters)
                                      │
                                      └─ Metrics accumulator (AlignmentMetricsAccumulator, future metrics)
                                        ↓
                                   Run artifacts (runs/, artifacts/) → CLI reports, plots, notebooks
```

### 3.1 Core runtime (`goal_drift_gym/core`)
- `Engine`: Orchestrates reset/step loops, enforces `max_steps`, and records `StepLogEntry` instances.
- `EngineConfig`: Captures runtime knobs (`max_steps`, `seed`, `allow_noop`).
- `types.py`: Dataclasses for `Observation`, `Action`, `ToolSpec`, `StepOutcome`, and log serialization helpers.

### 3.2 Scenario plugins (`goal_drift_gym/scenarios`)
- Registry in `scenarios/__init__.py` maps scenario names to config dataclasses and builders.
- `ReleaseManagerScenario` and `TradingGoalVsProfitScenario` implement `reset` / `step` and supply `ToolSpec` lists.
- Scenarios return alignment scores per step; log metadata (`forced_action`, state snapshots) to support downstream analysis.

### 3.3 Agents & policies
- Baseline heuristics per scenario live beside the CLI (`scripts/run.py`).
- `OpenRouterAgent` encapsulates external tool-calling LLM interactions, including prompt formatting, tool payload generation, and transcript logging.
- Future adapters (e.g., Inspect task runners) should adhere to the `RunConfig.agent` protocol so the engine stays agnostic.

### 3.4 Evaluation layer (`goal_drift_gym/eval`)
- `AlignmentMetricsAccumulator` computes mean score, slope, stickiness, and threshold crossings.
- `plots.py` and `scripts/plot.py` visualize alignment trajectories, tool usage, and panel traces.
- `analysis_nb.py` mirrors the CLI flow for notebook-driven experiments and persists the same artifact structure.

### 3.5 CLI entry points (`goal_drift_gym/scripts`)
- `run.py`: Main orchestration (config merging, directory prep, artifact writing).
- `report.py`: Emits machine-readable summaries for finished runs.
- `plot.py`: Batch plotting with optional multi-run summaries.

### 3.6 Telemetry & artifacts
- Run directories: `runs/<scenario>/<run-label>/<timestamp>-s<seed>/`.
- Each run captures `config.json`, `meta.json`, `metrics.json`, `step_log.jsonl`, `transcript.txt`, and `transcript.md`.
- Artifact mirror under `artifacts/<scenario>/<run-label>/<timestamp>-s<seed>/` stores plots and post-hoc reports.

## 4. Configuration & Extensibility
- `RunConfig` merges config files (`configs/*.yaml`) with CLI overrides; ensures engine/scenario parameter alignment.
- Scenarios expose dataclass configs so new parameters can be surfaced through YAML or `--scenario-param key=value`.
- Metrics accumulators are protocol-compatible; additional metrics packages can be registered and invoked from `run.py`.
- Agents adhere to `AgentConfig` (type + params). Adding a new integration = implement agent + update `build_agent`.

## 5. Interoperability & Integrations
- **OpenRouter:** First-class agent integration; loads API keys via `.env` and records tool-call transcripts.
- **Inspect + Petri (proposed):**
  - POC available under `inspect_tasks/release_manager_poc.py`.
  - Migration plan documented in `docs/inspect-petri-migration-plan.md`.
  - Target architecture: treat goal-drift scenarios as Inspect tasks, reuse Petri’s 36-dimension scorer, and retain goal-drift telemetry for cross-analysis.
- Future integrations (e.g., open-source policy gradients, benchmarking frameworks) should reuse scenario APIs and artifact conventions.

## 6. Success Metrics
- Reproducibility: identical runs (same config hash + git SHA) produce byte-identical telemetry.
- Observability: every episode yields interpretable transcripts and metrics; external auditors can reconstruct agent decisions without scenario code access.
- Extensibility: adding a new scenario or agent requires touching only scenario/agent layers; core engine remains untouched.
- Adoption: researchers can run baseline sweeps end-to-end using only published configs and CLI commands.
- Integration readiness: Inspect/Petri migration can export/import runs without schema churn.

## 7. Roadmap
- **Near term (0–1 month):**
  - Harden baseline policies and document tuning levers.
  - Ship Inspect/Petri integration toggle within `run.py`.
  - Add regression tests for scenario determinism and artifact schema.
- **Mid term (1–3 months):**
  - Introduce additional scenarios (e.g., compliance auditor, autonomous triage).
  - Support multi-agent episodes and adversary hooks.
  - Expand metrics to include action optimality comparisons and risk banding.
- **Long term (>3 months):**
  - Provide public benchmark suites with curated seeds.
  - Enable remote orchestration (Ray/local cluster) for sweeping agents.
  - Automate alignment drift reports combining goal-drift and Petri signals.

## 8. Risks & Mitigations
- **Schema drift:** Artifact readers break if schemas change. → Version `config.json` payloads and document migrations.
- **External API volatility:** OpenRouter/Inspect dependencies may evolve. → Encapsulate integrations behind agent adapters and feature flags.
- **Scenario coupling:** Complex scenarios may leak domain assumptions into core engine. → Enforce scenario interface boundaries and add contract tests.
- **Evaluation divergence:** Different metrics may disagree. → Keep a canonical alignment metric set and document interpretation guidelines.

## 9. Open Questions
- Should alignment metrics move to a plugin registry so Inspect/Petri outputs can co-exist with native metrics?
- What minimum data should transcripts expose for third-party judges (e.g., conversation IDs, latency)?
- How should we package public datasets of runs while preserving reproducibility without sharing proprietary prompts?
- What governance model ensures new scenarios meet safety and determinism requirements?

## 10. Contribution Checklist
- Define scenario config dataclass + register via `register_scenario`.
- Provide baseline policy or document why one is infeasible.
- Add integration tests covering reset/step determinism under fixed seeds.
- Document configs in `configs/` and update README plus design spec as scenarios/agents change.

## 11. References
- `README.md` — quickstart + CLI usage.
- `docs/inspect-petri-migration-plan.md` — Inspect/Petri roadmap.
- `docs/poc-summary.md` — Inspect integration POC recap.
- `goal_drift_gym/scripts/run.py` — orchestration source of truth.
- `goal_drift_gym/scenarios/` — reference implementations for plugin authors.

