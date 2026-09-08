from __future__ import annotations

import numpy as np

from quasi_exp.teacher.ellipsoidal_shell import (
    EllipsoidSpec,
    build_shell_mesh,
    covered_cells,
    icosphere,
    plane_section,
    radial_cell_volumes,
    sample_shell_cells,
    shell_coverage_report,
    triangular_prism_volume,
)


def _spec() -> EllipsoidSpec:
    angle = 0.37
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return EllipsoidSpec(
        center_m=np.asarray((0.75, -0.10, 0.05)),
        semiaxes_m=np.asarray((0.30, 0.20, 0.12)),
        rotation=rotation,
    )


def test_icosphere_registered_counts_and_ellipsoid_geometry() -> None:
    for subdivisions, expected in ((1, (42, 80)), (3, (642, 1280)), (4, (2562, 5120))):
        vertices, faces = icosphere(subdivisions)
        assert vertices.shape == (expected[0], 3)
        assert faces.shape == (expected[1], 3)
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1.0e-12)

    mesh = build_shell_mesh(_spec(), subdivisions=1)
    local = (mesh.vertices_m - mesh.spec.center_m) @ mesh.spec.rotation
    np.testing.assert_allclose(
        np.sum(np.square(local / mesh.spec.semiaxes_m), axis=1),
        1.0,
        atol=1.0e-12,
    )
    assert np.all(np.sum(mesh.normals * (mesh.vertices_m - mesh.spec.center_m), axis=1) > 0.0)
    assert all(tuple(sorted(row)) == row for row in mesh.vertex_neighbors)


def test_analytical_plane_section_satisfies_plane_and_ellipsoid() -> None:
    spec = _spec()
    normal = np.asarray((0.3, -0.4, 0.5))
    normal /= np.linalg.norm(normal)
    offset = float(normal @ spec.center_m + 0.03)
    section = plane_section(spec, normal, offset)
    assert section is not None
    points = section.points(np.linspace(0.0, 2.0 * np.pi, 97, endpoint=False))
    np.testing.assert_allclose(points @ normal, offset, atol=1.0e-12)
    local = (points - spec.center_m) @ spec.rotation
    np.testing.assert_allclose(
        np.sum(np.square(local / spec.semiaxes_m), axis=1),
        1.0,
        atol=2.0e-12,
    )
    assert section.semiaxes_m[0] >= section.semiaxes_m[1] > 0.0
    assert plane_section(spec, normal, float(normal @ spec.center_m + np.linalg.norm(spec.linear_map.T @ normal))) is None


def test_prism_volume_matches_area_times_height_for_parallel_prism() -> None:
    lower = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)))
    upper = lower + np.asarray((0.0, 0.0, 0.4))
    assert np.isclose(triangular_prism_volume(lower, upper), 0.5 * 2.0 * 3.0 * 0.4)


def test_same_chart_cell_coverage_and_physical_thickness() -> None:
    mesh = build_shell_mesh(_spec(), subdivisions=0)
    levels = np.asarray((-0.02, -0.01, 0.0, 0.01, 0.02))
    charts = np.full((len(levels), len(mesh.vertices_m)), "chart_A", dtype=object)
    mask, primary = covered_cells(mesh, levels, charts)
    assert mask.all()
    assert set(primary.reshape(-1)) == {"chart_A"}
    report = shell_coverage_report(mesh, levels, charts)
    assert np.isclose(report.surface_area_ratio, 1.0)
    assert np.isclose(report.volume_ratio, 1.0)
    np.testing.assert_allclose(report.symmetric_half_thickness_m, 0.02)
    assert np.isclose(report.thickness_area_weighted_quantiles_m["p10"], 0.02)

    # Losing one +20 mm vertex removes its incident outer cells and makes its
    # symmetric physical half-thickness exactly 10 mm.
    charts[-1, 0] = None
    report = shell_coverage_report(mesh, levels, charts)
    assert report.volume_ratio < 1.0
    assert np.isclose(report.symmetric_half_thickness_m[0], 0.01)


def test_volume_weighted_shell_sampling_is_deterministic_and_inside_cells() -> None:
    mesh = build_shell_mesh(_spec(), subdivisions=0)
    levels = np.asarray((-0.01, 0.0, 0.01))
    volumes = radial_cell_volumes(mesh, levels)
    assert volumes.shape == (2, 20)
    assert np.all(volumes > 0.0)
    mask = np.zeros_like(volumes, dtype=bool)
    mask[1, :2] = True
    first = sample_shell_cells(mesh, levels, mask, 4000, seed=20260820)
    second = sample_shell_cells(mesh, levels, mask, 4000, seed=20260820)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert set(first["face_id"]) == {0, 1}
    assert set(first["radial_interval_id"]) == {1}
    assert np.all(first["rho_m"] >= 0.0)
    assert np.all(first["rho_m"] <= 0.01)
    np.testing.assert_allclose(first["surface_barycentric"].sum(axis=1), 1.0, atol=1.0e-12)
    assert np.all(first["surface_barycentric"] >= 0.0)
