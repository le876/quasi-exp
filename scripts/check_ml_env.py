#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _parse_packages(raw: str) -> list[str]:
    out: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if name:
            out.append(name)
    return out


def _check_packages(names: list[str]) -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    missing: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
            missing.append(name)
    return versions, missing


def _check_tensorflow_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "tensorflow_version": None,
        "gpu_count": 0,
        "gpus": [],
        "error": None,
    }
    try:
        import tensorflow as tf  # type: ignore

        info["tensorflow_version"] = getattr(tf, "__version__", "unknown")
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_count"] = len(gpus)
        info["gpus"] = [str(g) for g in gpus]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-packages",
        default="numpy,pandas,sklearn,pyarrow",
        help="Comma separated package names that must be importable.",
    )
    parser.add_argument(
        "--check-tf-gpu",
        action="store_true",
        help="Probe tensorflow GPU visibility and include result in output.",
    )
    parser.add_argument(
        "--require-tf-gpu",
        action="store_true",
        help="Fail if tensorflow GPU is not available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    required = _parse_packages(args.require_packages)
    versions, missing = _check_packages(required)

    tf_info: dict[str, Any] | None = None
    if args.check_tf_gpu or args.require_tf_gpu:
        tf_info = _check_tensorflow_gpu()

    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "required_packages": required,
        "package_versions": versions,
        "missing_packages": missing,
        "tensorflow_gpu": tf_info,
        "ok": True,
    }

    errors: list[str] = []
    if missing:
        errors.append(f"missing packages: {', '.join(missing)}")
    if args.require_tf_gpu:
        if tf_info is None:
            errors.append("tensorflow gpu probe was not executed")
        else:
            if tf_info.get("error"):
                errors.append(f"tensorflow probe error: {tf_info['error']}")
            if int(tf_info.get("gpu_count", 0)) < 1:
                errors.append("tensorflow gpu_count < 1")

    payload["errors"] = errors
    payload["ok"] = len(errors) == 0

    output = json.dumps(payload, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
