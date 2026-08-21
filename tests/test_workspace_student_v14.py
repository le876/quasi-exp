from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_inverse import InverseQuery, WorkspaceInverse
from quasi_exp.teacher.workspace_student import (
    PREVIOUS_BETA_COLUMNS,
    WorkspaceStudentModels,
    WorkspaceStudentTrainingConfig,
    build_workspace_student_models,
    build_workspace_student_inverse,
    evaluate_workspace_student,
    load_workspace_student_models,
    save_workspace_student_models,
    student_feature_columns,
    train_workspace_student,
    validate_workspace_student_frame,
)


def _geometry() -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.ones(31, dtype=np.float32),
        p_end_local_m=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        theta_sign=-1.0,
        beta_bounds_rad=np.deg2rad(np.tile([[-10.0, 10.0]], (6, 1))).astype(np.float32),
    )


def _frame(*, kind: str = "static", primary: bool = True) -> pd.DataFrame:
    rows = []
    for index in range(4):
        row = {
            "record_id": f"row_{index}",
            "kind": kind,
            "split_role": "train_core" if index < 2 else "validation",
            "chart_id": "chart_a" if index % 2 == 0 else "chart_b",
            "is_primary": primary,
            "sample_weight": float(index + 1),
            "x_m": 1.0 + index * 0.001,
            "y_m": 0.001 * index,
            "z_m": -0.001 * index,
        }
        row.update({column: 0.001 * (index + joint) for joint, column in enumerate(BETA_COLUMNS)})
        row.update({column: 0.0 for column in JACOBIAN_COLUMNS})
        if kind == "stateful":
            row.update({column: 0.0 for column in PREVIOUS_BETA_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_schema_is_mode_specific_positive_and_unexpanded() -> None:
    static = _frame()
    checked = validate_workspace_student_frame(static, mode=RepresentationMode.XYZ_GLOBAL)
    assert len(checked) == len(static)
    assert student_feature_columns(RepresentationMode.XYZ_GLOBAL) == XYZ_COLUMNS
    assert student_feature_columns(RepresentationMode.XYZ_ROUTER_EXPERTS) == XYZ_COLUMNS
    assert student_feature_columns(RepresentationMode.STATEFUL_ROUTER_EXPERTS) == (
        *XYZ_COLUMNS,
        *PREVIOUS_BETA_COLUMNS,
    )

    duplicated = pd.concat([static, static.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_workspace_student_frame(duplicated, mode=RepresentationMode.XYZ_GLOBAL)

    negative_weight = static.copy()
    negative_weight.loc[0, "sample_weight"] = 0.0
    with pytest.raises(ValueError, match="sample_weight must be positive"):
        validate_workspace_student_frame(negative_weight, mode=RepresentationMode.XYZ_GLOBAL)

    stateful = _frame(kind="stateful", primary=False)
    validate_workspace_student_frame(stateful, mode=RepresentationMode.STATEFUL_ROUTER_EXPERTS)


def test_bounded_models_use_xyz_or_stateful_features_and_six_beta_outputs() -> None:
    pytest.importorskip("tensorflow")
    geometry = _geometry()
    static = _frame()
    global_models = build_workspace_student_models(
        static, mode=RepresentationMode.XYZ_GLOBAL, geometry=geometry, hidden_units=(4,)
    )
    assert global_models.global_model.input_shape == (None, 3)
    assert global_models.global_model.output_shape == (None, 6)
    output = np.asarray(global_models.global_model(np.zeros((2, 3), dtype=np.float32)))
    assert np.all(output >= geometry.beta_bounds_rad[:, 0])
    assert np.all(output <= geometry.beta_bounds_rad[:, 1])

    stateful = _frame(kind="stateful", primary=False)
    stateful_models = build_workspace_student_models(
        stateful,
        mode=RepresentationMode.STATEFUL_ROUTER_EXPERTS,
        geometry=geometry,
        hidden_units=(4,),
    )
    assert stateful_models.router_model.input_shape == (None, 9)
    assert all(model.input_shape == (None, 9) for model in stateful_models.expert_models.values())


def test_public_inverse_abstains_for_nonstitchable_static_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class ConstantModel:
        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=float).reshape(1, -1)

        def __call__(self, features: np.ndarray, training: bool = False) -> np.ndarray:
            del training
            return np.repeat(self.values, len(features), axis=0)

    def simple_fk(beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3] + values[:, 3:]

    import quasi_exp.teacher.workspace_student as student_module

    monkeypatch.setattr(student_module, "_geometry_fk", lambda _geometry: simple_fk)
    models = WorkspaceStudentModels(
        mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
        chart_ids=("chart_a", "chart_b"),
        router_model=ConstantModel([0.6, 0.4]),
        expert_models={
            "chart_a": ConstantModel([0, 0, 0, 0, 0, 0]),
            "chart_b": ConstantModel(np.deg2rad([2, 0, 0, -2, 0, 0])),
        },
    )
    inverse = build_workspace_student_inverse(models, geometry=_geometry())
    assert isinstance(inverse, WorkspaceInverse)

    evaluation = evaluate_workspace_student(
        inverse, InverseQuery(xyz_m=np.zeros((1, 3)))
    )

    assert evaluation.accepted_count == 0
    assert evaluation.prediction.reasons == ("nonstitchable_static_ambiguity",)


def test_tiny_static_train_uses_public_interfaces_without_padding() -> None:
    pytest.importorskip("tensorflow")
    frame = _frame()
    result = train_workspace_student(
        frame.iloc[:2],
        frame.iloc[2:],
        mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_geometry(),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=(4,), max_steps=1, validation_interval=1, patience_intervals=1
        ),
    )

    assert result.train_row_count == 2
    assert result.validation_row_count == 2
    assert set(result.history["model_id"]) == {"global"}
    evaluation = evaluate_workspace_student(
        result.inverse, InverseQuery(xyz_m=np.asarray([[1.0, 0.0, 0.0]])))
    assert evaluation.accepted_count + evaluation.abstained_count == 1


def test_model_manifest_round_trip_preserves_router_chart_order(tmp_path) -> None:
    pytest.importorskip("tensorflow")
    models = build_workspace_student_models(
        _frame(),
        mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
        geometry=_geometry(),
        hidden_units=(5,),
        router_hidden_units=(3,),
    )

    manifest = save_workspace_student_models(models, tmp_path / "models")
    loaded = load_workspace_student_models(tmp_path / "models")

    assert manifest["chart_ids"] == ["chart_a", "chart_b"]
    assert loaded.chart_ids == models.chart_ids
    assert loaded.router_model.input_shape == models.router_model.input_shape
    assert set(loaded.expert_models) == set(models.expert_models)
