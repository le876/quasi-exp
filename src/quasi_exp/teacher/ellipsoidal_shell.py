"""Geometry and sampling primitives for the BACRA V13 shell dataset.

The module deliberately stops at the geometry/coverage seam.  IK solving and
canonical chart construction remain owned by the existing BACRA modules.  A
shell is represented by a triangulated centre surface and physical normal
offsets in metres, so the reported thickness is an end-effector-space
quantity rather than a parameter-space surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EllipsoidSpec:
    """A rotated ellipsoid ``center + rotation @ (semiaxes * unit)``."""

    center_m: np.ndarray
    semiaxes_m: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float).reshape(3)
        semiaxes = np.asarray(self.semiaxes_m, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        if not np.isfinite(center).all() or not np.isfinite(semiaxes).all():
            raise ValueError("ellipsoid center and semiaxes must be finite")
        if np.any(semiaxes <= 0.0):
            raise ValueError("ellipsoid semiaxes must be positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
            raise ValueError("ellipsoid rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
            raise ValueError("ellipsoid rotation must be proper")
        object.__setattr__(self, "center_m", center.copy())
        object.__setattr__(self, "semiaxes_m", semiaxes.copy())
        object.__setattr__(self, "rotation", rotation.copy())

    @property
    def linear_map(self) -> np.ndarray:
        return self.rotation @ np.diag(self.semiaxes_m)

    def surface_points(self, unit_vectors: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit_vectors, dtype=float).reshape(-1, 3)
        if not np.isfinite(unit).all():
            raise ValueError("unit vectors must be finite")
        norms = np.linalg.norm(unit, axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-8):
            raise ValueError("surface parameters must have unit norm")
        return self.center_m + unit @ self.linear_map.T

    def outward_normals(self, unit_vectors: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit_vectors, dtype=float).reshape(-1, 3)
        local_gradient = unit / self.semiaxes_m.reshape(1, 3)
        world = local_gradient @ self.rotation.T
        return world / np.linalg.norm(world, axis=1, keepdims=True)


@dataclass(frozen=True)
class ShellMesh:
    """Triangulated ellipsoid centre surface with physical geometry."""

    spec: EllipsoidSpec
    unit_vertices: np.ndarray
    faces: np.ndarray
    vertices_m: np.ndarray
    normals: np.ndarray
    face_areas_m2: np.ndarray
    vertex_neighbors: tuple[tuple[int, ...], ...]

    @property
    def surface_area_m2(self) -> float:
        return float(np.sum(self.face_areas_m2))

    def offset_vertices(self, rho_m: float | np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_m, dtype=float)
        if rho.ndim == 0:
            return self.vertices_m + float(rho) * self.normals
        rho = rho.reshape(len(self.vertices_m), 1)
        return self.vertices_m + rho * self.normals


@dataclass(frozen=True)
class PlaneSection:
    """Analytical ellipse made by intersecting a plane and an ellipsoid."""

    center_m: np.ndarray
    axes_world: np.ndarray
    semiaxes_m: np.ndarray
    plane_normal: np.ndarray
    plane_offset_m: float

    def points(self, phase_rad: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase_rad, dtype=float).reshape(-1)
        circle = np.column_stack((np.cos(phase), np.sin(phase)))
        return self.center_m + circle @ (
            self.axes_world * self.semiaxes_m.reshape(1, 2)
        ).T


@dataclass(frozen=True)
class ShellCoverageReport:
    surface_area_ratio: float
    volume_ratio: float
    surface_area_total_m2: float
    surface_area_covered_m2: float
    shell_volume_total_m3: float
    shell_volume_covered_m3: float
    symmetric_half_thickness_m: np.ndarray
    thickness_area_weighted_quantiles_m: Mapping[str, float]


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic unit-sphere vertices and triangular faces."""
    if int(subdivisions) != subdivisions or subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    vertices, faces = _icosahedron()
    rows = [row.copy() for row in vertices]
    for _ in range(int(subdivisions)):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(left: int, right: int) -> int:
            key = (min(left, right), max(left, right))
            if key not in midpoint_cache:
                value = rows[left] + rows[right]
                value /= np.linalg.norm(value)
                midpoint_cache[key] = len(rows)
                rows.append(value)
            return midpoint_cache[key]

        refined: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(int(a), int(b)), midpoint(int(b), int(c)), midpoint(int(c), int(a))
            refined.extend(((int(a), ab, ca), (int(b), bc, ab), (int(c), ca, bc), (ab, bc, ca)))
        faces = np.asarray(refined, dtype=np.int64)
    return np.asarray(rows, dtype=float), faces


def build_shell_mesh(spec: EllipsoidSpec, subdivisions: int) -> ShellMesh:
    unit, faces = icosphere(subdivisions)
    vertices = spec.surface_points(unit)
    normals = spec.outward_normals(unit)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if np.any(areas <= 0.0):
        raise ValueError("shell mesh contains degenerate faces")
    neighbors: list[set[int]] = [set() for _ in vertices]
    for a, b, c in faces:
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return ShellMesh(
        spec=spec,
        unit_vertices=unit,
        faces=faces,
        vertices_m=vertices,
        normals=normals,
        face_areas_m2=areas,
        vertex_neighbors=tuple(tuple(sorted(values)) for values in neighbors),
    )


def plane_section(
    spec: EllipsoidSpec,
    plane_normal: np.ndarray,
    plane_offset_m: float,
) -> PlaneSection | None:
    """Return the exact planar ellipse, or ``None`` for no/tangent section."""
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0.0:
        raise ValueError("plane normal must be finite and non-zero")
    normal /= np.linalg.norm(normal)
    offset = float(plane_offset_m)
    if not np.isfinite(offset):
        raise ValueError("plane offset must be finite")
    transform = spec.linear_map
    p = transform.T @ normal
    p_norm = float(np.linalg.norm(p))
    signed = offset - float(normal @ spec.center_m)
    ratio = signed / p_norm
    if abs(ratio) >= 1.0 - 1.0e-14:
        return None
    unit_center = signed * p / (p_norm * p_norm)
    _u, _s, vt = np.linalg.svd(p.reshape(1, 3), full_matrices=True)
    null_basis = vt[1:].T
    section_map = transform @ null_basis * math.sqrt(max(0.0, 1.0 - ratio * ratio))
    axes_world, singular, _ = np.linalg.svd(section_map, full_matrices=False)
    order = np.argsort(-singular)
    axes_world = axes_world[:, order]
    singular = singular[order]
    if np.dot(np.cross(axes_world[:, 0], axes_world[:, 1]), normal) < 0.0:
        axes_world[:, 1] *= -1.0
    return PlaneSection(
        center_m=spec.center_m + transform @ unit_center,
        axes_world=axes_world,
        semiaxes_m=singular,
        plane_normal=normal,
        plane_offset_m=offset,
    )


def tetrahedron_volume(vertices_m: np.ndarray) -> float:
    vertices = np.asarray(vertices_m, dtype=float).reshape(4, 3)
    return float(abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0)


_PRISM_TETRAHEDRA = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)), dtype=np.int64)


def triangular_prism_tetrahedra(
    lower_vertices_m: np.ndarray,
    upper_vertices_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower_vertices_m, dtype=float).reshape(3, 3)
    upper = np.asarray(upper_vertices_m, dtype=float).reshape(3, 3)
    vertices = np.vstack((lower, upper))
    tetrahedra = vertices[_PRISM_TETRAHEDRA]
    volumes = np.asarray([tetrahedron_volume(value) for value in tetrahedra])
    return tetrahedra, volumes


def triangular_prism_volume(lower_vertices_m: np.ndarray, upper_vertices_m: np.ndarray) -> float:
    """Exact volume of the piecewise-linear triangular prism."""
    _tetrahedra, volumes = triangular_prism_tetrahedra(lower_vertices_m, upper_vertices_m)
    return float(np.sum(volumes))


def radial_cell_volumes(mesh: ShellMesh, radial_levels_m: Sequence[float]) -> np.ndarray:
    levels = np.asarray(radial_levels_m, dtype=float).reshape(-1)
    if len(levels) < 2 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0.0):
        raise ValueError("radial levels must be finite, strictly increasing, and contain at least two values")
    result = np.empty((len(levels) - 1, len(mesh.faces)), dtype=float)
    for radial_index, (lower_rho, upper_rho) in enumerate(zip(levels[:-1], levels[1:])):
        lower = mesh.offset_vertices(float(lower_rho))
        upper = mesh.offset_vertices(float(upper_rho))
        for face_id, face in enumerate(mesh.faces):
            result[radial_index, face_id] = triangular_prism_volume(lower[face], upper[face])
    return result


def _chart_set(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset((value,)) if value else frozenset()
    return frozenset(str(item) for item in value if str(item))


def covered_cells(
    mesh: ShellMesh,
    radial_levels_m: Sequence[float],
    vertex_level_charts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell acceptance and deterministic primary chart.

    A cell is covered only when one chart is present at all six vertices of
    the triangular prism.  This is the discrete same-chart closure used by the
    V13 volume and sampling gates.
    """
    levels = np.asarray(radial_levels_m, dtype=float).reshape(-1)
    charts = np.asarray(vertex_level_charts, dtype=object)
    if charts.shape != (len(levels), len(mesh.vertices_m)):
        raise ValueError("vertex_level_charts must have shape (radial_levels, vertices)")
    mask = np.zeros((len(levels) - 1, len(mesh.faces)), dtype=bool)
    primary = np.full(mask.shape, "", dtype=object)
    for radial_index in range(len(levels) - 1):
        for face_id, face in enumerate(mesh.faces):
            common: set[str] | None = None
            for level_index in (radial_index, radial_index + 1):
                for vertex_id in face:
                    values = set(_chart_set(charts[level_index, int(vertex_id)]))
                    common = values if common is None else common & values
            if common:
                mask[radial_index, face_id] = True
                primary[radial_index, face_id] = sorted(common)[0]
    return mask, primary


def symmetric_half_thickness(
    radial_levels_m: Sequence[float],
    vertex_level_charts: np.ndarray,
) -> np.ndarray:
    """Largest consecutive, same-chart symmetric reach around rho=0."""
    levels = np.asarray(radial_levels_m, dtype=float).reshape(-1)
    charts = np.asarray(vertex_level_charts, dtype=object)
    if charts.ndim != 2 or charts.shape[0] != len(levels):
        raise ValueError("vertex_level_charts must align with radial levels")
    center_indices = np.flatnonzero(np.isclose(levels, 0.0, atol=1.0e-12))
    if len(center_indices) != 1:
        raise ValueError("radial levels must contain exactly one zero level")
    center = int(center_indices[0])
    result = np.zeros(charts.shape[1], dtype=float)
    positive = levels[levels > 0.0]
    for vertex_id in range(charts.shape[1]):
        center_charts = _chart_set(charts[center, vertex_id])
        best = 0.0
        for reach in positive:
            negative_indices = np.flatnonzero(np.isclose(levels, -reach, atol=1.0e-12))
            positive_indices = np.flatnonzero(np.isclose(levels, reach, atol=1.0e-12))
            if len(negative_indices) != 1 or len(positive_indices) != 1:
                continue
            common = center_charts & _chart_set(charts[int(negative_indices[0]), vertex_id]) & _chart_set(charts[int(positive_indices[0]), vertex_id])
            if not common:
                break
            best = float(reach)
        result[vertex_id] = best
    return result


def _vertex_area_weights(mesh: ShellMesh) -> np.ndarray:
    weights = np.zeros(len(mesh.vertices_m), dtype=float)
    for face, area in zip(mesh.faces, mesh.face_areas_m2):
        weights[face] += float(area) / 3.0
    return weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    rows = np.asarray(values, dtype=float).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    requested = np.asarray(tuple(quantiles), dtype=float)
    if len(rows) != len(mass) or not len(rows) or np.any(mass < 0.0) or np.sum(mass) <= 0.0:
        raise ValueError("values and non-negative positive-total weights must align")
    if np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    order = np.argsort(rows, kind="stable")
    cumulative = np.cumsum(mass[order])
    targets = requested * cumulative[-1]
    return np.interp(targets, cumulative, rows[order], left=rows[order[0]], right=rows[order[-1]])


def shell_coverage_report(
    mesh: ShellMesh,
    radial_levels_m: Sequence[float],
    vertex_level_charts: np.ndarray,
) -> ShellCoverageReport:
    levels = np.asarray(radial_levels_m, dtype=float).reshape(-1)
    mask, _primary = covered_cells(mesh, levels, vertex_level_charts)
    volumes = radial_cell_volumes(mesh, levels)
    center = int(np.flatnonzero(np.isclose(levels, 0.0, atol=1.0e-12))[0])
    charts = np.asarray(vertex_level_charts, dtype=object)
    face_covered = np.zeros(len(mesh.faces), dtype=bool)
    for face_id, face in enumerate(mesh.faces):
        common: set[str] | None = None
        for vertex_id in face:
            values = set(_chart_set(charts[center, int(vertex_id)]))
            common = values if common is None else common & values
        face_covered[face_id] = bool(common)
    thickness = symmetric_half_thickness(levels, charts)
    quantiles = weighted_quantile(thickness, _vertex_area_weights(mesh), (0.0, 0.1, 0.5, 0.9))
    area_total = mesh.surface_area_m2
    area_covered = float(np.sum(mesh.face_areas_m2[face_covered]))
    volume_total = float(np.sum(volumes))
    volume_covered = float(np.sum(volumes[mask]))
    return ShellCoverageReport(
        surface_area_ratio=area_covered / area_total,
        volume_ratio=volume_covered / volume_total,
        surface_area_total_m2=area_total,
        surface_area_covered_m2=area_covered,
        shell_volume_total_m3=volume_total,
        shell_volume_covered_m3=volume_covered,
        symmetric_half_thickness_m=thickness,
        thickness_area_weighted_quantiles_m={
            name: float(value) for name, value in zip(("min", "p10", "p50", "p90"), quantiles)
        },
    )


def sample_shell_cells(
    mesh: ShellMesh,
    radial_levels_m: Sequence[float],
    accepted_cell_mask: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    stratified: bool = True,
) -> Mapping[str, np.ndarray]:
    """Uniformly sample the accepted piecewise-linear shell volume.

    Cells and their three tetrahedra are selected by exact physical volume;
    tetrahedral barycentric coordinates are Dirichlet(1,1,1,1).  This avoids
    the trajectory-density bias that V13 is intended to remove.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    levels = np.asarray(radial_levels_m, dtype=float).reshape(-1)
    mask = np.asarray(accepted_cell_mask, dtype=bool)
    expected = (len(levels) - 1, len(mesh.faces))
    if mask.shape != expected:
        raise ValueError(f"accepted_cell_mask must have shape {expected}")
    cell_ids = np.argwhere(mask)
    if not len(cell_ids):
        raise ValueError("at least one shell cell must be accepted")
    tetra_rows: list[np.ndarray] = []
    tetra_meta: list[tuple[int, int, int]] = []
    tetra_volume: list[float] = []
    tetra_cell: list[int] = []
    cell_volume: list[float] = []
    for cell_ordinal, (radial_index, face_id) in enumerate(cell_ids):
        face = mesh.faces[int(face_id)]
        lower = mesh.offset_vertices(float(levels[int(radial_index)]))[face]
        upper = mesh.offset_vertices(float(levels[int(radial_index) + 1]))[face]
        tetrahedra, volumes = triangular_prism_tetrahedra(lower, upper)
        cell_volume.append(float(np.sum(volumes)))
        for tetra_id, (tetrahedron, volume) in enumerate(zip(tetrahedra, volumes)):
            if volume > 0.0:
                tetra_rows.append(tetrahedron)
                tetra_meta.append((int(radial_index), int(face_id), int(tetra_id)))
                tetra_volume.append(float(volume))
                tetra_cell.append(int(cell_ordinal))
    rng = np.random.default_rng(int(seed))
    weights = np.asarray(tetra_volume, dtype=float)
    if stratified:
        cell_weights = np.asarray(cell_volume, dtype=float)
        expected_count = int(sample_count) * cell_weights / cell_weights.sum()
        cell_count = np.floor(expected_count).astype(np.int64)
        remainder = int(sample_count) - int(cell_count.sum())
        if remainder:
            fractional = expected_count - cell_count
            order = np.lexsort((np.arange(len(fractional)), -fractional))
            cell_count[order[:remainder]] += 1
        selected_parts: list[np.ndarray] = []
        tetra_cell_array = np.asarray(tetra_cell, dtype=np.int64)
        for cell_ordinal, count in enumerate(cell_count):
            if not count:
                continue
            available = np.flatnonzero(tetra_cell_array == cell_ordinal)
            local_weights = weights[available]
            selected_parts.append(
                rng.choice(
                    available,
                    size=int(count),
                    replace=True,
                    p=local_weights / local_weights.sum(),
                )
            )
        selected = np.concatenate(selected_parts)
        selected = selected[rng.permutation(len(selected))]
    else:
        selected = rng.choice(len(weights), size=int(sample_count), replace=True, p=weights / weights.sum())
    exponential = -np.log(np.maximum(rng.random((int(sample_count), 4)), np.finfo(float).tiny))
    bary4 = exponential / exponential.sum(axis=1, keepdims=True)
    xyz = np.empty((int(sample_count), 3), dtype=float)
    face_id = np.empty(int(sample_count), dtype=np.int64)
    radial_index = np.empty(int(sample_count), dtype=np.int64)
    tetra_id = np.empty(int(sample_count), dtype=np.int64)
    surface_bary = np.zeros((int(sample_count), 3), dtype=float)
    prism_weights = np.zeros((int(sample_count), 6), dtype=float)
    rho = np.empty(int(sample_count), dtype=float)
    for row_id, selection in enumerate(selected):
        xyz[row_id] = bary4[row_id] @ tetra_rows[int(selection)]
        r_index, f_id, t_id = tetra_meta[int(selection)]
        radial_index[row_id], face_id[row_id], tetra_id[row_id] = r_index, f_id, t_id
        prism_vertex = _PRISM_TETRAHEDRA[t_id]
        six_weights = np.zeros(6, dtype=float)
        six_weights[prism_vertex] = bary4[row_id]
        prism_weights[row_id] = six_weights
        surface_bary[row_id] = six_weights[:3] + six_weights[3:]
        upper_fraction = float(np.sum(six_weights[3:]))
        rho[row_id] = (1.0 - upper_fraction) * levels[r_index] + upper_fraction * levels[r_index + 1]
    return {
        "xyz_m": xyz,
        "face_id": face_id,
        "radial_interval_id": radial_index,
        "tetrahedron_id": tetra_id,
        "surface_barycentric": surface_bary,
        "prism_vertex_weights": prism_weights,
        "rho_m": rho,
    }
