#!/usr/bin/env python3
"""Evaluate locked BACRA V14 models without mutating training artifacts.

The default sequence evaluates sealed 40 mm macroblocks, the preregistered
trajectory suite, and then writes the final operational/scientific summary.
Every stage uses create-once directories, so an existing evaluation cannot be
silently replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from run_bacra_v14_omega200_atlas import (
    SOURCE_ROOT,
    _resolve_project_path,
    _strict_json,
    load_protocol_config,
    project_root_from,
    stage_spatial_evaluation,
    stage_summary,
    stage_trajectory_evaluation,
)


EVALUATION_STAGES = {
    "spatial_evaluation": stage_spatial_evaluation,
    "trajectory_evaluation": stage_trajectory_evaluation,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml"),
    )
    parser.add_argument(
        "--preset", choices=("smoke", "pilot", "formal"), default="smoke"
    )
    parser.add_argument(
        "--stage", choices=("all", *EVALUATION_STAGES), default="all"
    )
    parser.add_argument("--output-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_protocol_config(args.config, args.preset)
    project_root = project_root_from(SOURCE_ROOT)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _resolve_project_path(project_root, config["output_root"])
    )
    selected = tuple(EVALUATION_STAGES) if args.stage == "all" else (args.stage,)
    reports = {}
    for stage_name in selected:
        reports[stage_name] = EVALUATION_STAGES[stage_name](
            config, project_root, output_root
        )
        if not bool(reports[stage_name].get("gate_pass", False)):
            break
    print(json.dumps(_strict_json(reports), sort_keys=True, indent=2))
    return 0 if reports and all(
        bool(report.get("gate_pass", False)) for report in reports.values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
