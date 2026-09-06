import tempfile

import numpy as np
import torch

from simulation_toolkit.substrate_generator.myelin import generate_inner_fibers
from simulation_toolkit.utils.common_utils import (
    load_substrate_geometry,
    map_matrix_to_list_numpy,
    save_data_array_to_pickle,
    save_myelinated_substrate_to_pickle,
)


def test_generate_inner_fibers_preserves_beaded_radius_profile():
    outer = np.array([
        [0.0, 0.0, -5.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, 5.0, 1.0, 1.0],
    ], dtype=np.float32)

    inner = generate_inner_fibers(outer, g_ratio=0.7, inner_sphere_spacing_ratio=0.5)

    assert inner.shape[1] == 6
    assert np.isclose(inner[0, 3], 0.7)
    assert np.isclose(inner[-1, 3], 0.7)
    assert np.all(inner[:, 3] > 0)
    assert np.all(inner[:, 3] <= 1.4 + 1e-5)
    assert np.all(inner[:, 4] == 1.0)


def test_load_substrate_geometry_supports_myelinated_and_legacy_formats():
    outer = np.array([
        [0.0, 0.0, -5.0, 1.0, 1.0],
        [0.0, 0.0, 5.0, 1.0, 1.0],
    ], dtype=np.float32)
    inner = generate_inner_fibers(outer, g_ratio=0.7, inner_sphere_spacing_ratio=0.5)

    with tempfile.NamedTemporaryFile(suffix=".pkl") as myelin_file:
        save_myelinated_substrate_to_pickle(myelin_file.name, outer, inner, 10.0, 0.7, 0.5)
        loaded = load_substrate_geometry(myelin_file.name)
        assert loaded.is_myelinated
        assert loaded.box_length == 10.0
        assert loaded.g_ratio == 0.7
        assert loaded.outer_fibers.shape == outer.shape
        assert loaded.inner_fibers.shape == inner.shape

    with tempfile.NamedTemporaryFile(suffix=".pkl") as legacy_file:
        save_data_array_to_pickle(legacy_file.name, outer, 10.0)
        loaded = load_substrate_geometry(legacy_file.name)
        assert not loaded.is_myelinated
        assert loaded.inner_fibers is None
        assert loaded.outer_fibers.shape == outer.shape


def test_myelinated_geometry_grouping_uses_fiber_id_column_not_sphere_id():
    inner = torch.tensor([
        [0.0, 0.0, -5.0, 0.7, 1.0, 10.0],
        [0.0, 0.0, 5.0, 0.7, 1.0, 11.0],
        [1.0, 0.0, -5.0, 0.8, 2.0, 12.0],
        [1.0, 0.0, 5.0, 0.8, 2.0, 13.0],
    ], dtype=torch.float32)

    fiber_list = map_matrix_to_list_numpy(inner)

    assert len(fiber_list) == 2
    assert fiber_list[0].shape[0] == 2
    assert fiber_list[1].shape[0] == 2
    assert np.all(fiber_list[0][:, 4] == 1.0)
    assert np.all(fiber_list[1][:, 4] == 2.0)


def test_generate_inner_fibers_filters_pbc_duplicate_rows():
    """Inner generation must split the augmented outer at z=+L/2 boundaries
    and only treat the canonical segment (start anchor in box) as a fiber
    polyline. PBC-image segments must not be turned into extra inner
    polylines (which previously caused horizontal beam artefacts).

    The new contract also requires inner to mirror outer's PBC image map
    one-to-one, so we verify the same (axis, shift) duplicates appear in
    the returned inner array."""
    box_length = 4.0
    canonical = np.array([
        [0.5, 0.0, -2.0, 0.5, 1.0],
        [0.5, 0.0, 0.0, 0.5, 1.0],
        [0.5, 0.0, 2.0, 0.5, 1.0],
    ], dtype=np.float32)
    pbc_x_image = canonical.copy()
    pbc_x_image[:, 0] += box_length
    pbc_y_image = canonical.copy()
    pbc_y_image[:, 1] -= box_length
    outer = np.vstack([canonical, pbc_x_image, pbc_y_image])

    inner = generate_inner_fibers(
        outer,
        g_ratio=0.7,
        inner_sphere_spacing_ratio=0.5,
        box_length=box_length,
    )

    half = box_length / 2
    canonical_mask = (np.abs(inner[:, 0]) <= half + 1e-4) & (np.abs(inner[:, 1]) <= half + 1e-4)
    canonical_inner = inner[canonical_mask]
    image_inner = inner[~canonical_mask]

    # Canonical inner anchors match outer canonical anchors exactly.
    canonical_inner_sorted = canonical_inner[np.argsort(canonical_inner[:, 5])]
    assert np.allclose(canonical_inner_sorted[0, :3], canonical[0, :3])
    assert np.allclose(canonical_inner_sorted[-1, :3], canonical[-1, :3])

    # Inner mirrors outer's two PBC images: +x shift and -y shift.
    shifts_x = image_inner[:, 0] - canonical_inner_sorted[
        np.searchsorted(canonical_inner_sorted[:, 5], image_inner[:, 5]), 0
    ]
    # The image rows include the +x copies (shift +L, y unchanged) and the
    # -y copies (x unchanged, shift -L).
    pos_x_count = np.sum(np.isclose(image_inner[:, 0] - 0.5, box_length, atol=1e-4))
    neg_y_count = np.sum(np.isclose(image_inner[:, 1] - 0.0, -box_length, atol=1e-4))
    assert pos_x_count == canonical_inner.shape[0]
    assert neg_y_count == canonical_inner.shape[0]


def test_generate_inner_fibers_endpoints_match_outer_endpoints():
    """Inner first/last sphere centers must equal the outer first/last
    centers exactly so the optimizer's anchored start/end points are
    preserved (purple-end == red-start in PBC pairing)."""
    box_length = 4.0
    half = box_length / 2
    outer = np.array([
        # fiber_id 1: anchored at (0.5, -0.3) on the +/-L/2 walls
        [0.5, -0.3, -half, 0.5, 1.0],
        [0.4, -0.2, 0.0, 0.5, 1.0],
        [0.5, -0.3, half, 0.5, 1.0],
        # fiber_id 2: anchored at (0.5, -0.3) too, swapped z (PBC partner)
        [0.5, -0.3, -half, 0.5, 2.0],
        [0.6, -0.4, 0.0, 0.5, 2.0],
        [0.5, -0.3, half, 0.5, 2.0],
    ], dtype=np.float32)

    inner = generate_inner_fibers(
        outer,
        g_ratio=0.7,
        inner_sphere_spacing_ratio=0.5,
        box_length=box_length,
    )

    for fid in (1.0, 2.0):
        outer_rows = outer[outer[:, 4] == fid]
        inner_rows = inner[inner[:, 4] == fid]
        inner_rows = inner_rows[np.argsort(inner_rows[:, 5])]
        assert np.allclose(inner_rows[0, :3], outer_rows[0, :3], atol=1e-5)
        assert np.allclose(inner_rows[-1, :3], outer_rows[-1, :3], atol=1e-5)
        assert np.isclose(inner_rows[0, 2], -half)
        assert np.isclose(inner_rows[-1, 2], half)

    # Inner end of fiber_id=1 (at z=+L/2) must share x,y with inner start of
    # fiber_id=2 (at z=-L/2), matching the outer PBC pairing.
    inner_1_end = inner[inner[:, 4] == 1.0]
    inner_1_end = inner_1_end[np.argmax(inner_1_end[:, 2])][:2]
    inner_2_start = inner[inner[:, 4] == 2.0]
    inner_2_start = inner_2_start[np.argmin(inner_2_start[:, 2])][:2]
    assert np.allclose(inner_1_end, inner_2_start, atol=1e-5)


def test_generate_inner_fibers_emits_pbc_image_duplicates_like_outer():
    """When the augmented outer carries a PBC-image segment for a fiber,
    the saved inner array must include a corresponding PBC-image segment
    shifted by the same (axis, +/-L) offset. This guarantees inner and
    outer share the same storage format and per-fiber image count."""
    box_length = 4.0
    half = box_length / 2
    # Canonical outer: x=1.7 (touches +x wall: 1.7+0.5=2.2 > L/2 - L/5).
    canonical = np.array([
        [1.7, 0.0, -half, 0.5, 1.0],
        [1.7, 0.0, 0.0, 0.5, 1.0],
        [1.7, 0.0, half, 0.5, 1.0],
    ], dtype=np.float32)
    # Augmented outer: include the -L x-shifted image, as the optimizer
    # would after torch_optimizer_wrapPBC.
    pbc_x_image = canonical.copy()
    pbc_x_image[:, 0] -= box_length
    outer = np.vstack([canonical, pbc_x_image])

    inner = generate_inner_fibers(
        outer,
        g_ratio=0.7,
        inner_sphere_spacing_ratio=0.5,
        box_length=box_length,
    )

    canonical_mask = (np.abs(inner[:, 0]) <= half + 1e-4) & (np.abs(inner[:, 1]) <= half + 1e-4)
    canonical_inner = inner[canonical_mask]
    duplicates = inner[~canonical_mask]
    assert canonical_inner.shape[0] > 0
    assert duplicates.shape[0] > 0
    # Duplicates must be canonical rows shifted by exactly -box_length on x.
    assert np.all(np.isclose(duplicates[:, 0] - canonical_inner[:, 0], -box_length, atol=1e-5))
    # Radii and fiber_ids must be preserved across duplication.
    assert np.allclose(np.sort(duplicates[:, 3]), np.sort(canonical_inner[:, 3]))
    assert np.array_equal(np.unique(duplicates[:, 4]), np.unique(canonical_inner[:, 4]))


def test_generate_inner_fibers_ignores_interior_graze_at_z_boundary():
    """A single fiber must not be split into two segments when an interior
    sphere drifts to within tolerance of +Lz/2 during optimization.

    Endpoints are masked at exactly +/-Lz/2, but interior spheres move; on
    anisotropic thin-z substrates an interior sphere near the end can land at
    e.g. z=+Lz/2 + 6.7e-5. The segment splitter must only close at a genuine
    terminus (last row, or next row starts a new polyline at z=-Lz/2), so the
    fiber stays one canonical polyline and the endpoint-consistency check does
    not compare mismatched segments (regression for the K=7 VF=0.6 failure).
    """
    box_length = 100.0
    box_length_z = 45.0
    half_z = box_length_z / 2
    fid = 585.0
    r = 0.84
    # One fiber: start at -Lz/2, an interior sphere grazing just above +Lz/2,
    # then the true masked end at exactly +Lz/2.
    outer = np.array([
        [5.0, 5.0, -half_z, r, fid],
        [7.0, 4.0, 0.0, r, fid],
        [9.0, 3.0, half_z + 6.7e-5, r, fid],  # interior graze near +Lz/2
        [9.5, 2.9, half_z - 1e-3, r, fid],
        [10.6, 2.8, half_z, r, fid],           # true masked end
    ], dtype=np.float32)

    inner = generate_inner_fibers(
        outer,
        g_ratio=0.7,
        inner_sphere_spacing_ratio=0.5,
        box_length=box_length,
        box_length_z=box_length_z,
    )

    inner_rows = inner[inner[:, 4] == fid]
    inner_rows = inner_rows[np.argsort(inner_rows[:, 5])]
    # Endpoints must match the single fiber's true start/end (at +/-Lz/2).
    assert np.allclose(inner_rows[0, :3], outer[0, :3], atol=1e-4)
    assert np.allclose(inner_rows[-1, :3], outer[-1, :3], atol=1e-4)
    assert np.isclose(inner_rows[0, 2], -half_z, atol=1e-4)
    assert np.isclose(inner_rows[-1, 2], half_z, atol=1e-4)
