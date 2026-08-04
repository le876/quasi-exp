#!/usr/bin/env python3
"""Train BACRA V14 Students from an already materialized workspace dataset.

This is a narrow resume entrypoint.  It never regenerates Teacher labels and
refuses to overwrite an existing ``07_student`` stage.
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
    stage_student,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml"),
    )
    parser.add_argument(
        "--preset", choices=("smoke", "pilot", "formal"), default="smoke"
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
    report = stage_student(config, project_root, output_root)
    print(json.dumps(_strict_json(report), sort_keys=True, indent=2))
    return 0 if bool(report.get("gate_pass", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
