import json
import tempfile

import numpy as np

import simulation_toolkit.toolkit_params as config_params
from simulation_toolkit.utils import along_fiber_plot
from simulation_toolkit.utils.common_utils import (
    save_data_array_to_pickle,
    save_myelinated_substrate_to_pickle,
)


def _two_segment_geometry(box_length=2.0):
    half = box_length / 2
    canonical = np.array([
        [0.0, 0.0, -half, 0.5, 1.0],
        [0.0, 0.0, 0.0, 0.5, 1.0],
        [0.0, 0.0, half, 0.5, 1.0],
    ], dtype=np.float32)
    pbc_image = canonical.copy()
    pbc_image[:, 0] += box_length
    return np.vstack([canonical, pbc_image])


def test_collect_stored_axons_from_pickle_keeps_all_stored_segments():
    box_length = 2.0
    outer = _two_segment_geometry(box_length)

    with tempfile.NamedTemporaryFile(suffix=".pkl") as substrate_file:
        save_data_array_to_pickle(substrate_file.name, outer, box_length)
        collection = along_fiber_plot.collect_stored_axons_from_substrate_pickle(substrate_file.name)

    assert collection["box_length"] == box_length
    assert collection["component_segment_counts"] == {"outer": 2}
    assert len(collection["components"]["outer"]) == 2
    assert np.allclose(collection["components"]["outer"][1][:, 0], outer[3:, 0])


def test_effective_diameter_metric_summary_uses_requested_moments():
    radii = np.array([1.0, 2.0], dtype=np.float64)
    summary = along_fiber_plot._metric_summary(radii)

    sum_r2 = np.sum(radii ** 2)
    assert np.isclose(summary["d_eff_p3_q2_um"], 2 * np.sum(radii ** 3) / sum_r2)
    assert np.isclose(summary["r_app_wide_pulse_um"], (np.sum(radii ** 6) / sum_r2) ** 0.25)
    assert np.isclose(summary["d_app_wide_pulse_um"], 2 * (np.sum(radii ** 6) / sum_r2) ** 0.25)
    assert np.isclose(summary["r_app_internal_um"], np.sqrt(np.sum(radii ** 4) / sum_r2))
    assert np.isclose(summary["d_app_internal_um"], 2 * np.sqrt(np.sum(radii ** 4) / sum_r2))
    assert set(summary["moments"].keys()) == {"r1", "r2", "r3", "r4", "r6"}


def test_save_effective_axon_diameter_stats_from_myelinated_pickle_writes_outer_and_inner_json():
    box_length = 2.0
    outer = _two_segment_geometry(box_length)
    inner = outer.copy()
    inner[:, 3] *= 0.7
    sphere_id = np.arange(inner.shape[0], dtype=np.float32).reshape(-1, 1)
    inner = np.hstack([inner, sphere_id]).astype(np.float32)

    with tempfile.TemporaryDirectory() as output_dir:
        config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = output_dir
        config_params.EXP_DATE_TIME = "test-date"
        with tempfile.NamedTemporaryFile(suffix=".pkl") as substrate_file:
            save_myelinated_substrate_to_pickle(substrate_file.name, outer, inner, box_length, 0.7, 0.5)
            saved = along_fiber_plot.save_effective_axon_diameter_stats_from_pickle(
                substrate_file.name,
                slice_step_um=0.5,
                grid_resolution_um=0.05,
            )

        assert set(saved.keys()) == {"outer", "inner"}
        for component, path in saved.items():
            with open(path) as f:
                stats = json.load(f)
            assert stats["component_label"] == component
            assert stats["num_axons"] == 2
            assert stats["bundle"]["num_slices"] > 0
            assert stats["substrate"]["stored_segment_counts"][component] == 2
