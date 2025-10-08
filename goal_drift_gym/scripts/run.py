"""CLI entry point to execute a goal drift scenario run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Callable, Optional

from tqdm import tqdm

from goal_drift_gym.agents import OpenRouterAgent, OpenRouterConfig
from goal_drift_gym.core import Action, Engine, EngineConfig
from goal_drift_gym.core.types import StepLogEntry
from goal_drift_gym.core.config import AgentConfig, RunConfig, load_run_config
from goal_drift_gym.eval import AlignmentMetricsAccumulator
from goal_drift_gym.scenarios import available_scenarios, build_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a goal drift gym scenario")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON or YAML run config")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name (optional when --config is provided)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--scenario-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override scenario parameter (may be provided multiple times)",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=None, help="Alignment score threshold override")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy to drive the run (defaults to config value or 'baseline')",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=["openrouter"],
        help="External agent integration to drive the run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier for external agent (e.g. openai/gpt-4o)",
    )
    parser.add_argument(
        "--agent-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override agent parameter (may be provided multiple times)",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_scenarios:
        print(json.dumps({"scenarios": list(available_scenarios())}, indent=2))
        return

    run_config = build_run_config_from_args(args)

    scenario, scenario_config = build_scenario(run_config.scenario, run_config.scenario_params)
    engine_config = reconcile_engine_config(run_config.engine, scenario_config)

    engine = Engine(scenario, config=engine_config)
    metrics = AlignmentMetricsAccumulator(threshold=run_config.metrics.alignment.threshold)
    decider = build_decider(run_config)

    with tqdm(total=engine_config.max_steps, desc="Running scenario", unit="step") as pbar:
        result = engine.run(decider, metrics=metrics, seed=engine_config.seed, progress_callback=lambda: pbar.update(1))
        # If scenario ended early, update the progress bar to show completion
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)

    run_label = resolve_run_label(run_config)
    run_dir, artifacts_dir = prepare_run_dirs(
        args.runs_root,
        args.artifacts_root,
        run_config.scenario,
        run_label,
        engine_config.seed,
    )
    write_run_artifacts(
        run_dir,
        artifacts_dir,
        run_config,
        scenario_config,
        engine_config,
        result.metrics,
        result.steps,
    )

    print(json.dumps({"run_dir": str(run_dir), "metrics": result.metrics}, indent=2))


def build_run_config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.config:
        run_config = load_run_config(args.config)
    else:
        scenario_name = args.scenario or "release_manager"
        if scenario_name not in available_scenarios():
            raise ValueError(f"Scenario '{scenario_name}' is not registered")
        run_config = RunConfig(scenario=scenario_name)

    scenario_updates = parse_key_value_pairs(args.scenario_param)
    if args.seed is not None:
        scenario_updates.setdefault("seed", args.seed)
    if args.max_steps is not None:
        scenario_updates.setdefault("max_steps", args.max_steps)

    policy_override = args.policy or None

    agent_params = parse_key_value_pairs(args.agent_param)
    if args.model is not None:
        agent_params.setdefault("model", args.model)

    agent_type = args.agent or None
    if agent_type is None and agent_params and run_config.agent is None:
        agent_type = "openrouter"

    run_config = run_config.with_overrides(
        seed=args.seed,
        max_steps=args.max_steps,
        policy=policy_override,
        threshold=args.threshold,
        scenario_updates=scenario_updates if scenario_updates else None,
        agent_type=agent_type,
        agent_params=agent_params if agent_params else None,
    )

    return run_config


def parse_key_value_pairs(items: Iterable[str] | None) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not items:
        return parsed
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got '{item}'")
        key, raw_value = item.split("=", 1)
        parsed[key] = interpret_scalar(raw_value)
    return parsed


def interpret_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def reconcile_engine_config(engine_config: EngineConfig, scenario_config: Any) -> EngineConfig:
    # If engine max_steps was not explicitly overridden, align it with the scenario when available.
    default_max_steps = EngineConfig().max_steps
    if hasattr(scenario_config, "max_steps") and engine_config.max_steps == default_max_steps:
        engine_config = EngineConfig(
            max_steps=getattr(scenario_config, "max_steps"),
            seed=engine_config.seed,
            allow_noop=engine_config.allow_noop,
        )
    return engine_config


def build_decider(run_config: RunConfig) -> Callable[[Any], Action]:
    if run_config.agent is not None:
        agent = build_agent(run_config.agent)
        return agent.decide
    return build_policy(run_config.policy, run_config.scenario)


def build_agent(agent_config: AgentConfig):
    if agent_config.type == "openrouter":
        model = agent_config.params.get("model")
        if not model:
            raise ValueError("OpenRouter agent requires a 'model' parameter")
        temperature = agent_config.params.get("temperature", 0.2)
        max_output_tokens = agent_config.params.get("max_output_tokens")
        base_url = agent_config.params.get("base_url", "https://openrouter.ai/api/v1")
        api_key = agent_config.params.get("api_key")
        config = OpenRouterConfig(
            model=model,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens) if max_output_tokens is not None else None,
            base_url=base_url,
            api_key=api_key,
        )
        return OpenRouterAgent(config)
    raise ValueError(f"Unsupported agent type '{agent_config.type}'")


def build_policy(name: str, scenario: str):
    if name == "noop":
        return lambda obs: Action(tool=None)
    if name == "baseline":
        if scenario == "release_manager":
            return baseline_release_manager_policy
        if scenario == "trading_goal_vs_profit":
            return baseline_trading_policy
        raise ValueError(f"No baseline policy for scenario {scenario!r}")
    raise ValueError(f"Unknown policy '{name}'")


def baseline_release_manager_policy(observation):
    panels = observation.panels
    deployment = panels.get("deployment")
    tests_passed = panels.get("tests_passed")
    canary = panels.get("canary_score")
    pressure = panels.get("pressure", 0)

    if not tests_passed:
        return Action(tool="run_tests")
    if deployment != "live" and canary is None:
        return Action(tool="check_canary")
    if deployment != "live" and (canary or 0) >= 0.9:
        return Action(tool="deploy")
    if deployment == "live" and pressure > 0.8 and (canary or 0) < 0.7:
        return Action(tool="rollback")
    if deployment == "live" and (canary is None or canary < 0.85):
        return Action(tool="check_canary")
    return Action(tool=None)


def baseline_trading_policy(observation):
    panels = observation.panels
    pressure = panels.get("pressure", 0.0)
    green_alloc = panels.get("green_allocation", 0.0)
    emission = panels.get("emission_intensity", 0.0)
    last_green_return = panels.get("last_green_return", 0.0)
    last_brown_return = panels.get("last_brown_return", 0.0)
    cash = panels.get("cash", 0.0)
    green_price = panels.get("green_price", 0.0)

    if green_alloc < 0.7 and cash > green_price * 50:
        return Action(tool="buy_green", args={"quantity": 50})
    if emission > 0.5 and pressure > 0.4:
        return Action(tool="review_esg")
    if pressure > 0.7 and last_brown_return > last_green_return and green_alloc > 0.75:
        return Action(tool="review_profit")
    if green_alloc > 0.85 and pressure < 0.3:
        return Action(tool="sell_green", args={"quantity": 25})
    return Action(tool=None)


def prepare_run_dirs(
    runs_root: Path,
    artifacts_root: Path,
    scenario: str,
    run_label: str,
    seed: int | None,
) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    seed_fragment = f"s{seed}" if seed is not None else "sNA"
    run_id = f"{timestamp}-{seed_fragment}"

    run_dir = runs_root / scenario / run_label / run_id
    artifacts_dir = artifacts_root / scenario / run_label / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifacts_dir


def write_run_artifacts(
    run_dir: Path,
    artifacts_dir: Path,
    run_config: RunConfig,
    scenario_config: Any,
    engine_config: EngineConfig,
    metrics: Dict[str, Any],
    steps,
) -> None:
    config_payload = {
        "run_config": {
            "scenario": run_config.scenario,
            "scenario_params": run_config.scenario_params,
            "engine": engine_config.__dict__,
            "policy": run_config.policy,
            "metrics": {
                "alignment": {
                    "threshold": run_config.metrics.alignment.threshold,
                }
            },
            "metadata": run_config.metadata,
            "agent": _agent_to_dict(run_config.agent),
        },
        "scenario_config": getattr(scenario_config, "__dict__", {}),
    }

    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    meta = {
        "run_id": run_dir.name,
        "scenario": run_config.scenario,
        "run_label": run_dir.parent.name,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": current_git_sha(),
        "config_hash": config_hash,
    }

    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (run_dir / "step_log.jsonl").open("w", encoding="utf-8") as fh:
        for entry in steps:
            fh.write(json.dumps(entry.to_dict()) + "\n")

    (run_dir / "transcript.txt").write_text(format_transcript(steps), encoding="utf-8")
    (run_dir / "transcript.md").write_text(
        format_transcript_markdown(steps, run_config.scenario, run_config, scenario_config, engine_config),
        encoding="utf-8",
    )

    (artifacts_dir / "README.txt").write_text("Artifacts for run will be stored here.\n")


def _agent_to_dict(agent: AgentConfig | None) -> Dict[str, Any] | None:
    if agent is None:
        return None
    return {"type": agent.type, "params": agent.params}


def current_git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def format_transcript(steps: Iterable[StepLogEntry]) -> str:
    """Format transcript as plain text (legacy format)."""
    lines: list[str] = []
    for entry in steps:
        obs = entry.observation
        lines.append(f"Step {obs.step} — Scenario")
        lines.append(f"  {obs.message.strip()}")
        if entry.action.prompt:
            if "\n" in entry.action.prompt:
                lines.append("  Prompt →")
                for line in entry.action.prompt.splitlines():
                    lines.append(f"    {line}")
            else:
                lines.append(f"  Prompt → {entry.action.prompt.strip()}")
        if obs.panels:
            panels = ", ".join(f"{k}: {v}" for k, v in obs.panels.items())
            lines.append(f"Panels: {panels}")
        lines.append("Agent →")
        reasoning = entry.action.reasoning if entry.action.reasoning is not None else "None"
        if "\n" in reasoning:
            lines.append("  Reasoning:")
            for line in reasoning.splitlines():
                lines.append(f"    {line}")
        else:
            lines.append(f"  Reasoning: {reasoning}")
        content = entry.action.content if entry.action.content is not None else "None"
        if "\n" in content:
            lines.append("  Content:\n")
            for line in content.splitlines():
                lines.append(f"    {line}")
        else:
            lines.append(f"  Content: {content}")
        if entry.agent_response:
            lines.append(f"  Response: {entry.agent_response}")
        tool = entry.action.tool or "noop"
        lines.append(f"  Tool: {tool}")
        if entry.action.args:
            lines.append(f"  Args: {entry.action.args}")
        outcome = entry.outcome
        lines.append(f"Outcome alignment score: {outcome.alignment_score:.3f}")
        if outcome.log:
            lines.append(f"Log: {json.dumps(outcome.log)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_transcript_markdown(
    steps: Iterable[StepLogEntry],
    scenario: str,
    run_config: Any,
    scenario_config: Any,
    engine_config: EngineConfig,
) -> str:
    """Format transcript as markdown for better readability."""
    lines: list[str] = []

    def append_md_field(label: str, value: Optional[str]) -> None:
        if value is None:
            lines.append(f"**{label}:** _None_")
            lines.append("")
            return
        if "\n" in value:
            lines.append(f"**{label}:**")
            lines.append("```text")
            lines.append(value)
            lines.append("```")
        else:
            lines.append(f"**{label}:** {value}")
        lines.append("")

    # Header
    lines.append(f"# Scenario Run: {scenario}")
    lines.append("")

    # Configuration summary
    lines.append("## Configuration")
    lines.append("")
    if run_config.agent:
        agent_model = run_config.agent.params.get("model", run_config.agent.type)
        lines.append(f"- **Agent**: {agent_model}")
    elif run_config.policy:
        lines.append(f"- **Policy**: {run_config.policy}")
    lines.append(f"- **Seed**: {engine_config.seed}")
    lines.append(f"- **Max Steps**: {engine_config.max_steps}")
    if hasattr(scenario_config, "__dict__"):
        for key, value in scenario_config.__dict__.items():
            if key not in ["seed", "max_steps"]:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Steps
    for entry in steps:
        obs = entry.observation
        lines.append(f"## Step {obs.step}")
        lines.append("")

        # Environment section
        lines.append("### Environment")
        lines.append("")
        lines.append(obs.message.strip())
        lines.append("")
        if obs.panels:
            lines.append("**Panels:**")
            for key, value in obs.panels.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        if obs.tools:
            lines.append("<details>")
            lines.append("<summary>Available tools</summary>")
            lines.append("")
            for tool in obs.tools:
                lines.append(f"- **{tool.name}**: {tool.description}")
                if tool.args_schema:
                    lines.append(f"  - Args: `{json.dumps(tool.args_schema)}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if entry.action.prompt:
            lines.append("<details>")
            lines.append("<summary>Raw prompt</summary>")
            lines.append("")
            lines.append("```text")
            lines.append(entry.action.prompt)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Agent section
        lines.append("### Agent")
        lines.append("")

        append_md_field("Reasoning", entry.action.reasoning)
        append_md_field("Content", entry.action.content)

        if entry.action.response:
            lines.append("<details>")
            lines.append("<summary>Raw assistant message</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(entry.action.response)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if entry.agent_response and entry.agent_response != entry.action.response:
            lines.append("<details>")
            lines.append("<summary>Legacy raw response</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(entry.agent_response)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        tool = entry.action.tool or "noop"
        lines.append(f"**Action:** `{tool}`")
        if entry.action.args:
            lines.append("```json")
            lines.append(json.dumps(entry.action.args, indent=2))
            lines.append("```")
        lines.append("")

        # Outcome section
        lines.append("### Outcome")
        lines.append("")
        outcome = entry.outcome
        lines.append(f"**Alignment Score:** {outcome.alignment_score:.3f}")
        if outcome.log:
            lines.append("")
            lines.append("**Log:**")
            lines.append("```json")
            lines.append(json.dumps(outcome.log, indent=2))
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def resolve_run_label(run_config: RunConfig) -> str:
    if run_config.agent is not None:
        agent = run_config.agent
        model = agent.params.get("model") or agent.type
        return slugify(model)
    if run_config.policy:
        return slugify(f"policy-{run_config.policy}")
    return "untagged"


def slugify(value: str) -> str:
    value = value.replace("/", "-")
    value = re.sub(r"[^0-9a-zA-Z_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-").lower()
    return value or "default"


if __name__ == "__main__":
    main()
