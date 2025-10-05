"""Quick summary report for a finished run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a goal drift gym run")
    parser.add_argument("run", type=Path, help="Path to a run directory (runs/<scenario>/<run-id>)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = _load_json(run_dir / "config.json")
    meta = _load_json(run_dir / "meta.json")
    metrics = _load_json(run_dir / "metrics.json")

    summary = {
        "run": run_dir.name,
        "scenario": meta.get("scenario"),
        "created_at": meta.get("created_at"),
        "seed": config.get("scenario_config", {}).get("seed"),
        "num_steps": metrics.get("num_steps"),
        "mean_alignment_score": metrics.get("mean_alignment_score"),
        "alignment_slope": metrics.get("alignment_slope"),
        "first_pass_below_threshold": metrics.get("first_pass_below_threshold"),
        "stickiness": metrics.get("stickiness"),
    }

    print(json.dumps({"summary": summary, "metrics": metrics, "meta": meta}, indent=2))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
