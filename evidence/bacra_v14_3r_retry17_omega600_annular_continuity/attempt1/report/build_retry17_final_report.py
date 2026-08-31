#!/usr/bin/env python3
"""Build an isolated final Retry17 advisor-facing diagnostic report.

The reviewed Phase-B builder already contains the complete, read-only evidence
replay: deterministic point-cloud sampling, authoritative FK replay, and the
four held-out trajectory paths.  This small wrapper executes that logic in a
new report directory after binding it to the current Retry17 source worktree.
It intentionally writes only derived report material beside this file.
"""
from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
PHASE_B_BUILDER = HERE.parent / "phase_b" / "build_phase_b_report.py"
CURRENT_SOURCE = "/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-3r-retry17-continuity"
OLD_SOURCE = "/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-3r-retry17-continuity-exec5"


def main() -> None:
    if not PHASE_B_BUILDER.is_file():
        raise FileNotFoundError(PHASE_B_BUILDER)
    source = PHASE_B_BUILDER.read_text(encoding="utf-8")
    source = source.replace(OLD_SOURCE, CURRENT_SOURCE)
    source = source.replace(
        'HTML_NAME = "retry17_phase_b_advisor_dataset_report.html"',
        'HTML_NAME = "retry17_attempt1_final_advisor_dataset_report.html"',
    )
    source = source.replace(
        "retry17 Phase B 数据集与轨迹诊断报告",
        "retry17 attempt1 最终数据集与轨迹诊断报告",
    )
    source = source.replace(
        "retry17 正式 attempt1：Phase B 环状连续性数据集与冻结后轨迹",
        "retry17 正式 attempt1：最终环状连续性数据集与冻结后轨迹",
    )
    source = source.replace(
        "report/phase_b/build_phase_b_report.py",
        "report/final/build_retry17_final_report.py",
    )
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(PHASE_B_BUILDER), "exec"), namespace)


if __name__ == "__main__":
    main()
