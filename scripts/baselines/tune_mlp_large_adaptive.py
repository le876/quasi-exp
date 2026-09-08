#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for writing params YAML") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@dataclass
class TrialMetrics:
    trial_id: int
    train_theta_mae_deg: float
    train_tension_mae_n: float
    val_theta_mae_deg: float
    val_tension_mae_n: float
    test_theta_mae_deg: float
    test_tension_mae_n: float
    test_score: float
    fit_time_s: float


class AdaptiveController:
    def __init__(self) -> None:
        self.hidden_grid = [
            (224, 112, 56, 28),
            (256, 128, 64, 32),
            (288, 144, 72, 36),
            (320, 160, 80, 40),
            (352, 176, 88, 44),
        ]
        self.alpha_grid = [3e-8, 1e-7, 3e-7, 1e-6, 3e-6]
        self.lr_grid = [6e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3]
        self.batch_grid = [512, 768, 1024, 1536]
        self.max_iter_grid = [600, 800, 1000, 1200]
        self.n_iter_grid = [30, 50, 70]
        self.tol_grid = [3e-6, 1e-6, 3e-7]

        self.base_indices = {
            "hidden": 1,
            "alpha": 1,
            "lr": 2,
            "batch": 2,
            "max_iter": 1,
            "n_iter": 1,
            "tol": 1,
        }
        self.last_move: tuple[str, int] | None = None

    def _clip(self, knob: str, idx: int) -> int:
        bounds = {
            "hidden": len(self.hidden_grid),
            "alpha": len(self.alpha_grid),
            "lr": len(self.lr_grid),
            "batch": len(self.batch_grid),
            "max_iter": len(self.max_iter_grid),
            "n_iter": len(self.n_iter_grid),
            "tol": len(self.tol_grid),
        }
        return min(max(idx, 0), bounds[knob] - 1)

    def _apply_move(self, indices: dict[str, int], move: tuple[str, int]) -> dict[str, int]:
        knob, delta = move
        out = dict(indices)
        out[knob] = self._clip(knob, out[knob] + int(delta))
        return out

    def _regime(self, metrics: TrialMetrics) -> str:
        theta_gap = metrics.val_theta_mae_deg - metrics.train_theta_mae_deg
        tension_gap = metrics.val_tension_mae_n - metrics.train_tension_mae_n
        if theta_gap > 0.45 or tension_gap > 18.0:
            return "overfit"
        if theta_gap < 0.20 and tension_gap < 8.0:
            return "underfit"
        return "balanced"

    def _candidate_moves(self, regime: str) -> list[tuple[str, int]]:
        if regime == "overfit":
            return [
                ("alpha", +1),
                ("hidden", -1),
                ("lr", -1),
                ("batch", +1),
                ("max_iter", -1),
                ("n_iter", -1),
                ("tol", +1),
            ]
        if regime == "underfit":
            return [
                ("hidden", +1),
                ("max_iter", +1),
                ("alpha", -1),
                ("lr", +1),
                ("batch", -1),
                ("n_iter", +1),
                ("tol", -1),
            ]
        return [
            ("lr", -1),
            ("alpha", -1),
            ("batch", -1),
            ("hidden", +1),
            ("max_iter", +1),
            ("tol", -1),
        ]

    def _signature(self, indices: dict[str, int]) -> tuple[int, ...]:
        return (
            indices["hidden"],
            indices["alpha"],
            indices["lr"],
            indices["batch"],
            indices["max_iter"],
            indices["n_iter"],
            indices["tol"],
        )

    def propose(
        self,
        current_indices: dict[str, int],
        best_indices: dict[str, int],
        previous_metrics: TrialMetrics | None,
        previous_improved: bool,
        seen: set[tuple[int, ...]],
    ) -> tuple[dict[str, int], str, tuple[str, int]]:
        if previous_metrics is None:
            first_move = ("max_iter", +1)
            candidate = self._apply_move(current_indices, first_move)
            return candidate, "warmup", first_move

        regime = self._regime(previous_metrics)
        if previous_improved and self.last_move is not None:
            repeated = self._apply_move(best_indices, self.last_move)
            if self._signature(repeated) not in seen:
                return repeated, f"{regime}:repeat", self.last_move

        for move in self._candidate_moves(regime):
            candidate = self._apply_move(best_indices, move)
            if self._signature(candidate) not in seen:
                self.last_move = move
                return candidate, regime, move

        # Fallback to the closest baseline variant not seen.
        for knob in ["max_iter", "alpha", "lr", "batch", "hidden", "n_iter", "tol"]:
            for delta in (-1, +1):
                candidate = self._apply_move(best_indices, (knob, delta))
                if self._signature(candidate) not in seen:
                    self.last_move = (knob, delta)
                    return candidate, "fallback", (knob, delta)

        # Last resort: return best itself.
        return dict(best_indices), "stagnation", ("none", 0)

    def params_from_indices(self, indices: dict[str, int]) -> dict[str, Any]:
        return {
            "mlp_large": {
                "hidden_layer_sizes": list(self.hidden_grid[indices["hidden"]]),
                "alpha": float(self.alpha_grid[indices["alpha"]]),
                "learning_rate_init": float(self.lr_grid[indices["lr"]]),
                "batch_size": int(self.batch_grid[indices["batch"]]),
                "max_iter": int(self.max_iter_grid[indices["max_iter"]]),
                "n_iter_no_change": int(self.n_iter_grid[indices["n_iter"]]),
                "tol": float(self.tol_grid[indices["tol"]]),
            }
        }


def _run_trial(
    python_executable: str,
    dataset: Path,
    robot_config: Path,
    split_file: Path,
    out_dir: Path,
    params_file: Path,
    seed: int,
    trial_tag: str,
) -> TrialMetrics:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-u",
        "scripts/baselines/run_baselines.py",
        "--dataset",
        str(dataset),
        "--robot-config",
        str(robot_config),
        "--split-file",
        str(split_file),
        "--out-dir",
        str(out_dir),
        "--models",
        "mlp_large",
        "--params-file",
        str(params_file),
        "--summary-split",
        "test",
        "--eval-splits",
        "train,val,test",
        "--save-curves",
        "--save-preds",
        "2000",
        "--save-preds-splits",
        "val,test",
        "--seed",
        str(seed),
        "--tag",
        trial_tag,
        "--skip-fk",
    ]

    log_path = out_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"trial failed with exit code {return_code}: {trial_tag}")

    metrics_path = out_dir / "mlp_large" / "metrics.json"
    payload = _read_json(metrics_path)
    block = payload["metrics"]
    return TrialMetrics(
        trial_id=int(trial_tag.split("_")[-1]),
        train_theta_mae_deg=float(block["train"]["theta_mae_deg"]),
        train_tension_mae_n=float(block["train"]["tension_mae_n"]),
        val_theta_mae_deg=float(block["val"]["theta_mae_deg"]),
        val_tension_mae_n=float(block["val"]["tension_mae_n"]),
        test_theta_mae_deg=float(block["test"]["theta_mae_deg"]),
        test_tension_mae_n=float(block["test"]["tension_mae_n"]),
        test_score=float(block["test"]["score_val_like"]),
        fit_time_s=float(payload["fit_time_s"]),
    )


def _summary_markdown(
    dataset: Path,
    split_file: Path,
    history: list[dict[str, Any]],
    best_trial: dict[str, Any],
    baseline_trial: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# MLP-Large 20-Round Adaptive Tuning Summary")
    lines.append("")
    lines.append(f"- dataset: `{dataset}`")
    lines.append(f"- split: `{split_file}`")
    lines.append(f"- rounds: {len(history)}")
    lines.append("")
    lines.append(
        "- baseline(trial_{trial}): theta={theta:.4f} deg, tension={tension:.2f} N, score={score:.6f}".format(
            trial=baseline_trial["trial"],
            theta=baseline_trial["test_theta_mae_deg"],
            tension=baseline_trial["test_tension_mae_n"],
            score=baseline_trial["test_score"],
        )
    )
    lines.append(
        "- best(trial_{trial}): theta={theta:.4f} deg, tension={tension:.2f} N, score={score:.6f}".format(
            trial=best_trial["trial"],
            theta=best_trial["test_theta_mae_deg"],
            tension=best_trial["test_tension_mae_n"],
            score=best_trial["test_score"],
        )
    )
    lines.append(
        "- delta vs baseline: d_theta={d_theta:.4f} deg, d_tension={d_tension:.2f} N".format(
            d_theta=best_trial["test_theta_mae_deg"] - baseline_trial["test_theta_mae_deg"],
            d_tension=best_trial["test_tension_mae_n"] - baseline_trial["test_tension_mae_n"],
        )
    )
    lines.append("")
    lines.append("| trial | theta_test(deg) | tension_test(N) | score | fit(s) | move | regime | improved |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for row in history:
        lines.append(
            "| {trial:02d} | {theta:.4f} | {tension:.2f} | {score:.6f} | {fit:.2f} | {move} | {regime} | {improved} |".format(
                trial=row["trial"],
                theta=row["test_theta_mae_deg"],
                tension=row["test_tension_mae_n"],
                score=row["test_score"],
                fit=row["fit_time_s"],
                move=row["move"],
                regime=row["regime"],
                improved="yes" if row["improved"] else "no",
            )
        )
    return "\n".join(lines) + "\n"


def _plot_curves(history: list[dict[str, Any]], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(item["trial"]) for item in history]
    theta_values = [float(item["test_theta_mae_deg"]) for item in history]
    tension_values = [float(item["test_tension_mae_n"]) for item in history]
    scores = [float(item["test_score"]) for item in history]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(rounds, theta_values, marker="o")
    axes[0].set_title("test theta MAE (deg)")
    axes[0].set_xlabel("trial")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rounds, tension_values, marker="o", color="tab:orange")
    axes[1].set_title("test tension MAE (N)")
    axes[1].set_xlabel("trial")
    axes[1].grid(alpha=0.3)

    axes[2].plot(rounds, scores, marker="o", color="tab:green")
    axes[2].set_title("test score")
    axes[2].set_xlabel("trial")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260207)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    params_dir = args.out_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    controller = AdaptiveController()
    current_indices = dict(controller.base_indices)
    best_indices = dict(controller.base_indices)
    seen_signatures: set[tuple[int, ...]] = {controller._signature(current_indices)}

    history: list[dict[str, Any]] = []
    previous_metrics: TrialMetrics | None = None
    previous_improved = False
    best_score = math.inf

    for round_number in range(1, int(args.rounds) + 1):
        if round_number > 1:
            candidate_indices, regime, move = controller.propose(
                current_indices=current_indices,
                best_indices=best_indices,
                previous_metrics=previous_metrics,
                previous_improved=previous_improved,
                seen=seen_signatures,
            )
            current_indices = candidate_indices
            seen_signatures.add(controller._signature(current_indices))
        else:
            regime = "baseline"
            move = ("none", 0)

        trial_name = f"trial_{round_number:02d}"
        trial_dir = args.out_root / trial_name
        params_file = params_dir / f"{trial_name}.yaml"
        params_payload = controller.params_from_indices(current_indices)
        _write_yaml(params_file, params_payload)

        print(
            f"\n=== ROUND {round_number:02d}/{args.rounds} | regime={regime} | move={move[0]}:{move[1]} ===",
            flush=True,
        )
        start_time = time.time()
        metrics = _run_trial(
            python_executable=sys.executable,
            dataset=args.dataset,
            robot_config=args.robot_config,
            split_file=args.split_file,
            out_dir=trial_dir,
            params_file=params_file,
            seed=int(args.seed),
            trial_tag=trial_name,
        )
        elapsed = time.time() - start_time

        improved = metrics.test_score < best_score - 1e-6
        if improved:
            best_score = metrics.test_score
            best_indices = dict(current_indices)
        previous_improved = improved
        previous_metrics = metrics

        row = {
            "trial": round_number,
            "trial_name": trial_name,
            "regime": regime,
            "move": f"{move[0]}:{move[1]}",
            "params_file": str(params_file),
            "params": params_payload["mlp_large"],
            "train_theta_mae_deg": metrics.train_theta_mae_deg,
            "train_tension_mae_n": metrics.train_tension_mae_n,
            "val_theta_mae_deg": metrics.val_theta_mae_deg,
            "val_tension_mae_n": metrics.val_tension_mae_n,
            "test_theta_mae_deg": metrics.test_theta_mae_deg,
            "test_tension_mae_n": metrics.test_tension_mae_n,
            "test_score": metrics.test_score,
            "fit_time_s": metrics.fit_time_s,
            "wall_time_s": float(elapsed),
            "improved": bool(improved),
        }
        history.append(row)
        with (args.out_root / "history.jsonl").open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            "ROUND {round_id:02d} RESULT: theta={theta:.4f} deg, tension={tension:.2f} N, score={score:.6f}, improved={improved}".format(
                round_id=round_number,
                theta=metrics.test_theta_mae_deg,
                tension=metrics.test_tension_mae_n,
                score=metrics.test_score,
                improved=improved,
            ),
            flush=True,
        )

    baseline = history[0]
    best_row = min(history, key=lambda item: float(item["test_score"]))

    summary = _summary_markdown(
        dataset=args.dataset,
        split_file=args.split_file,
        history=history,
        best_trial=best_row,
        baseline_trial=baseline,
    )
    (args.out_root / "SUMMARY.md").write_text(summary, encoding="utf-8")
    _plot_curves(history, args.out_root / "trial_curve.png")
    _write_yaml(args.out_root / "best_params.yaml", {"mlp_large": best_row["params"]})
    _write_json(args.out_root / "summary.json", {"best": best_row, "baseline": baseline, "rounds": history})

    print("\n=== DONE ===", flush=True)
    print(f"best trial: {best_row['trial_name']}", flush=True)
    print(f"best theta_mae_deg: {best_row['test_theta_mae_deg']:.4f}", flush=True)
    print(f"best tension_mae_n: {best_row['test_tension_mae_n']:.2f}", flush=True)
    print(f"summary: {args.out_root / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
