import numpy as np
import torch


def _as_numpy(fibers):
    if isinstance(fibers, torch.Tensor):
        return fibers.detach().cpu().numpy()
    return np.asarray(fibers)


def _fiber_id_column(fibers):
    if fibers.shape[1] < 5:
        raise ValueError("Fiber geometry must have at least x, y, z, radius, fiber_id columns")
    return 4


def _sort_fiber_rows(fiber):
    if fiber.shape[1] >= 6:
        return fiber[np.argsort(fiber[:, 5])]
    # Fallback: sort by z so the polyline goes from -L/2 to +L/2.
    return fiber[np.argsort(fiber[:, 2])]


def _split_into_segments(spheres, box_length_z):
    """Split the augmented outer array into per-fiber polyline segments.

    Mirrors ``utils.common_utils.split_matrix_to_list``: each polyline runs from
    z=-Lz/2 to z=+Lz/2 exactly once (the optimizer anchors fibre ends at
    z = +/- Lz/2), so a row at z=+Lz/2 that is the last row or is immediately
    followed by a new polyline start (z=-Lz/2) closes one segment. A z=+Lz/2 row
    that is *not* a terminus (an interior sphere that drifted to within tol of
    +Lz/2 during optimization) is ignored so a single fibre is not falsely split.
    ``box_length_z`` is the *z*-extent (Lz), which is decoupled from the in-plane
    box (Lx=Ly) for anisotropic thin-z substrates, so the split threshold must
    use Lz, not the in-plane box.
    """
    if box_length_z is None or box_length_z <= 0 or len(spheres) == 0:
        return [spheres] if len(spheres) > 0 else []
    half = box_length_z / 2
    n = len(spheres)
    segments = []
    start_idx = 0
    for i in range(n):
        if not np.isclose(spheres[i, 2], half, atol=1e-4):
            continue
        # Only close the segment at a genuine polyline terminus: either the last
        # row, or the next row starts a new polyline at z=-Lz/2. Endpoints are
        # masked at exactly +/-Lz/2 by Init2D, but interior spheres move during
        # optimization and can drift to within tol of +Lz/2; without this guard
        # such a graze would falsely split one fiber into two segments sharing a
        # fiber_id (its next row continues the same fiber, not at -Lz/2).
        is_last = i == n - 1
        next_is_start = (not is_last) and np.isclose(spheres[i + 1, 2], -half, atol=1e-4)
        if is_last or next_is_start:
            segments.append(spheres[start_idx:i + 1])
            start_idx = i + 1
    if start_idx < len(spheres):
        segments.append(spheres[start_idx:])
    return [s for s in segments if len(s) >= 2]


def _is_canonical_segment(segment, box_length, tol=1e-4):
    """A segment is canonical iff its **start anchor** (row at z=-L/2) lies
    in the canonical box (x, y in [-L/2, L/2]).

    Init2D anchors every fiber's start at a 2D-packed in-box (xy_0) but may
    store the end as the "unwrapped" partner anchor (xy_L + L) when the
    shortest red/purple line crosses a wall. So requiring BOTH anchors in
    box would silently drop every wall-crossing fiber. PBC-image segments
    appended by ``torch_optimizer_wrapPBC`` have their *whole* fibre shifted
    by +/- L, so their start anchor is also outside the canonical box and
    they remain correctly rejected.

    The rule yields exactly one canonical segment per fiber_id; subsequent
    ``_add_pbc_images`` reconstructs the wall-crossing image in lock-step
    with the outer array.
    """
    if box_length is None or box_length <= 0:
        return True
    half = box_length / 2 + tol
    return abs(segment[0, 0]) <= half and abs(segment[0, 1]) <= half


def _interpolate_polyline(points, radii, cumulative_length, sample_length):
    if sample_length <= 0:
        return points[0].copy(), float(radii[0])
    if sample_length >= cumulative_length[-1]:
        return points[-1].copy(), float(radii[-1])

    segment_idx = np.searchsorted(cumulative_length, sample_length, side="right") - 1
    segment_idx = min(max(segment_idx, 0), len(points) - 2)
    segment_start = cumulative_length[segment_idx]
    segment_end = cumulative_length[segment_idx + 1]
    segment_length = segment_end - segment_start
    if segment_length <= 0:
        return points[segment_idx].copy(), float(radii[segment_idx])

    t = (sample_length - segment_start) / segment_length
    point = points[segment_idx] + t * (points[segment_idx + 1] - points[segment_idx])
    radius = radii[segment_idx] + t * (radii[segment_idx + 1] - radii[segment_idx])
    return point, float(radius)


def generate_inner_fibers(outer_fibers, g_ratio=0.7, inner_sphere_spacing_ratio=0.5, box_length=None, box_length_z=None):
    """
    Generate inner axonal membrane spheres from optimized outer membrane spheres.

    Behaviour:
      * PBC-duplicate rows (x or y outside the canonical box) are filtered out
        first, so each fiber_id reduces to a single canonical polyline running
        from z = -L/2 to z = +L/2.
      * The first inner sphere is placed exactly at the outer fiber's start
        point and the last inner sphere exactly at the outer fiber's end
        point, so inner endpoints match the outer endpoints anchored by the
        optimizer's mask_starts_ends. Together with the start/end pairing
        produced by Init2D this guarantees that for two fiber_ids that form
        the two PBC halves of the same physical axon, the inner end of one
        half coincides (in x, y) with the inner start of the other half.
      * Interior inner spheres are sampled along the polyline at spacing
        inner_sphere_spacing_ratio * local_inner_radius. Samples that fall
        within half a step of the end anchor are dropped to avoid stacking
        with the explicit end anchor.
    """
    if not (0 < g_ratio < 1):
        raise ValueError("g_ratio must be between 0 and 1")
    if inner_sphere_spacing_ratio <= 0:
        raise ValueError("inner_sphere_spacing_ratio must be positive")

    outer = _as_numpy(outer_fibers).astype(np.float32, copy=False)
    fid_col = _fiber_id_column(outer)
    # z-extent (Lz) drives per-fiber segment splitting and endpoint checks. It
    # is decoupled from the in-plane box (Lx=Ly) for anisotropic thin-z
    # substrates; fall back to the in-plane box only for legacy isotropic calls.
    if box_length_z is None:
        box_length_z = box_length
    inner_rows = []
    next_sphere_id = 0
    canonical_outer_segments = []

    # The augmented outer array contains canonical fibres plus PBC-image
    # duplicates produced by ``torch_optimizer_wrapPBC``. Split at z = +L/2
    # boundaries so each chunk is one polyline; canonical chunks have their
    # start/end anchors in the box, image chunks have anchors shifted by
    # +/- L. Working on whole segments avoids row-level mixing of canonical
    # and image points (which previously caused horizontal "beam" artefacts
    # whenever a fibre's interior x, y crossed a wall).
    for segment in _split_into_segments(outer, box_length_z):
        if not _is_canonical_segment(segment, box_length):
            continue
        canonical_outer_segments.append(segment)

        fiber_id = float(segment[0, fid_col])
        points = segment[:, :3].astype(np.float64)
        radii = segment[:, 3].astype(np.float64)

        if segment.shape[0] == 1:
            inner_rows.append([
                points[0, 0], points[0, 1], points[0, 2],
                g_ratio * radii[0], fiber_id, next_sphere_id,
            ])
            next_sphere_id += 1
            continue

        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = cumulative_length[-1]
        if total_length <= 0:
            continue

        # Anchor: first inner sphere at outer start point.
        first_inner_radius = g_ratio * float(radii[0])
        inner_rows.append([
            points[0, 0], points[0, 1], points[0, 2],
            first_inner_radius, fiber_id, next_sphere_id,
        ])
        next_sphere_id += 1

        # Interior samples.
        last_inner_radius = g_ratio * float(radii[-1])
        end_drop_tol = 0.5 * inner_sphere_spacing_ratio * last_inner_radius

        step = inner_sphere_spacing_ratio * first_inner_radius
        if step <= 0:
            raise ValueError("Computed non-positive inner sphere spacing")
        sample_length = step
        while sample_length < total_length - end_drop_tol:
            point, outer_radius = _interpolate_polyline(
                points, radii, cumulative_length, sample_length
            )
            inner_radius = g_ratio * outer_radius
            inner_rows.append([
                point[0], point[1], point[2], inner_radius, fiber_id, next_sphere_id,
            ])
            next_sphere_id += 1
            step = inner_sphere_spacing_ratio * inner_radius
            if step <= 0:
                raise ValueError("Computed non-positive inner sphere spacing")
            sample_length += step

        # Anchor: last inner sphere at outer end point.
        inner_rows.append([
            points[-1, 0], points[-1, 1], points[-1, 2],
            last_inner_radius, fiber_id, next_sphere_id,
        ])
        next_sphere_id += 1

    if not inner_rows:
        return np.zeros((0, 6), dtype=np.float32)
    inner_array = np.asarray(inner_rows, dtype=np.float32)

    _assert_endpoint_consistency(canonical_outer_segments, inner_array, box_length_z, fid_col)

    if box_length is not None and box_length > 0:
        # Mirror outer's PBC duplication map exactly. Recomputing it from
        # inner radii (g_ratio * outer_r) would miss wall-crossings because
        # inner spheres are smaller than outer; that produced 12 missing
        # inner image segments on the 20-axon test. Cloning outer's per-fid
        # (axis, shift) map keeps inner and outer one-to-one.
        inner_array = _replicate_outer_pbc_images(
            inner_array, outer, box_length, box_length_z, fid_col
        )
    return inner_array


def _replicate_outer_pbc_images(canonical_inner, augmented_outer, box_length, box_length_z, fid_col):
    """Duplicate canonical inner using the same per-fiber (axis, shift)
    pattern recorded in the augmented outer array.

    For each fiber_id, we compare each outer segment's start anchor to the
    canonical outer segment's start anchor; the difference (rounded to the
    nearest integer multiple of box_length along x and y) is the PBC shift
    that must also be applied to the corresponding canonical inner segment.
    """
    if canonical_inner.shape[0] == 0:
        return canonical_inner

    half = box_length / 2
    # Group outer segments by fid. Segment splitting is on the z-axis (Lz).
    outer_segments_by_fid = {}
    for seg in _split_into_segments(augmented_outer, box_length_z):
        fid = float(seg[0, fid_col])
        outer_segments_by_fid.setdefault(fid, []).append(seg)

    # Group canonical inner rows by fid (preserves storage order).
    inner_by_fid = {}
    for idx, row in enumerate(canonical_inner):
        fid = float(row[fid_col])
        inner_by_fid.setdefault(fid, []).append(idx)

    blocks = [canonical_inner]
    for fid, segs in outer_segments_by_fid.items():
        if fid not in inner_by_fid:
            continue
        # Identify canonical outer segment (start anchor in box).
        canon_seg = None
        for s in segs:
            if abs(s[0, 0]) <= half + 1e-4 and abs(s[0, 1]) <= half + 1e-4:
                canon_seg = s
                break
        if canon_seg is None:
            continue
        canon_xy = canon_seg[0, :2]
        inner_idx = np.asarray(inner_by_fid[fid], dtype=np.int64)
        inner_block = canonical_inner[inner_idx]
        seen_shifts = set()
        for s in segs:
            shift_x = int(round((s[0, 0] - canon_xy[0]) / box_length))
            shift_y = int(round((s[0, 1] - canon_xy[1]) / box_length))
            if (shift_x, shift_y) == (0, 0):
                continue
            if (shift_x, shift_y) in seen_shifts:
                continue
            seen_shifts.add((shift_x, shift_y))
            shifted = inner_block.copy()
            shifted[:, 0] += shift_x * box_length
            shifted[:, 1] += shift_y * box_length
            blocks.append(shifted)

    return np.vstack(blocks).astype(canonical_inner.dtype, copy=False)


def _add_pbc_images(spheres, box_length, fid_col, tol=None):
    """Mirror ``GeometricOptimization.torch_optimizer_wrapPBC`` in numpy.

    For every fiber_id whose canonical rows touch the +/-x or +/-y wall
    (within ``tol``), append duplicate rows of the entire fiber shifted by
    +/-box_length along that axis. This makes the inner array's storage
    format match the outer array's, so simulation seeders can treat both
    arrays identically without re-wrapping at load time.

    Note: this matches the optimizer's behaviour, which duplicates by whole
    fiber_id (not just the boundary-crossing rows) using ``torch.isin``.
    """
    if box_length is None or box_length <= 0 or spheres.shape[0] == 0:
        return spheres
    if tol is None:
        tol = box_length / 5

    half = box_length / 2
    augmented = spheres.copy()

    for axis in (0, 1):  # x then y
        x = augmented[:, axis]
        r = augmented[:, 3]
        fid = augmented[:, fid_col]

        low_mask = (x - r) < (-half + tol)
        # Match the optimizer's wall criterion: L - x - r < L/2 + tol  <=>  x + r > L/2 - tol
        high_mask = (x + r) > (half - tol)

        low_fids = np.unique(fid[low_mask]) if np.any(low_mask) else np.array([], dtype=fid.dtype)
        high_fids = np.unique(fid[high_mask]) if np.any(high_mask) else np.array([], dtype=fid.dtype)

        new_blocks = [augmented]
        if low_fids.size > 0:
            block = augmented[np.isin(fid, low_fids)].copy()
            block[:, axis] += box_length
            new_blocks.append(block)
        if high_fids.size > 0:
            block = augmented[np.isin(fid, high_fids)].copy()
            block[:, axis] -= box_length
            new_blocks.append(block)
        augmented = np.vstack(new_blocks)

    return augmented.astype(spheres.dtype, copy=False)


def _assert_endpoint_consistency(canonical_outer_segments, inner, box_length_z, fid_col, tol=1e-4):
    """Sanity check: per canonical outer segment, the inner sub-array sharing
    the same fiber_id must have first/last sphere centers equal to the
    segment's start/end anchors. When ``box_length_z`` (Lz) is provided each
    anchor must sit on |z| = Lz/2.

    ``inner`` is the canonical inner array (before PBC duplication), so its
    rows are grouped one-to-one with ``canonical_outer_segments`` by fiber_id.
    """
    half = None if box_length_z is None else box_length_z / 2
    for outer_segment in canonical_outer_segments:
        fiber_id = float(outer_segment[0, fid_col])
        inner_rows = _sort_fiber_rows(inner[inner[:, fid_col] == fiber_id])
        if outer_segment.shape[0] == 0 or inner_rows.shape[0] == 0:
            continue
        for label, o_idx, i_idx in (("start", 0, 0), ("end", -1, -1)):
            o_xyz = outer_segment[o_idx, :3]
            i_xyz = inner_rows[i_idx, :3]
            if not np.allclose(o_xyz, i_xyz, atol=tol):
                raise AssertionError(
                    f"Inner {label} of fiber_id={fiber_id} {i_xyz} does not "
                    f"match outer {label} {o_xyz}."
                )
            if half is not None:
                expected_z = -half if label == "start" else half
                if not np.isclose(o_xyz[2], expected_z, atol=tol):
                    raise AssertionError(
                        f"Outer {label} of fiber_id={fiber_id} has z={o_xyz[2]}, "
                        f"expected {expected_z}."
                    )