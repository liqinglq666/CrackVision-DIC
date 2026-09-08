from __future__ import annotations

import h5py
import numpy as np

from src.core.io_bridge import CrackVisionNcorrH5Loader
from src.core.physics import CrackPhysicsEngine


def test_lightweight_bridge_converts_formatted_mm_displacement_to_internal_pixels(tmp_path):
    path = tmp_path / "specimen_CrackVision.h5"
    shape = (2, 7, 9)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "CrackVision-Ncorr"
        f.attrs["format_version"] = 1
        f.attrs["coordinate_system"] = "reference"
        f.attrs["strain_measure"] = "Green-Lagrange"
        f.attrs["numeric_precision"] = "single"
        f.attrs["source_units"] = "mm"
        f.attrs["pixel_size_mm"] = 0.05
        f.attrs["ncorr_spacing_raw"] = 2.0
        f.attrs["dic_step_px"] = 3.0
        f.attrs["dic_point_spacing_mm"] = 0.15

        fields = f.create_group("fields")
        for name in ("u", "v", "exx", "eyy", "exy"):
            arr = np.zeros(shape, dtype=np.float32)
            if name == "u":
                arr[1] = 0.05  # formatted Ncorr displacement in mm = 1 image pixel
            fields.create_dataset(name, data=arr)
        fields.create_dataset("mask", data=np.ones(shape, dtype=np.uint8))
        f.create_dataset("time_s", data=np.array([[0.0], [5.0]]))

    frames = list(CrackVisionNcorrH5Loader.stream_frames(path))
    assert len(frames) == 2
    assert frames[0].u_map.shape == (7, 9)
    assert np.isclose(frames[1].u_map[0, 0], 1.0)
    assert np.isclose(frames[0].pixel_size_mm, 0.05)
    assert np.isclose(frames[0].dic_point_spacing_mm, 0.15)
    assert np.isclose(frames[1].time_s, 5.0)
    assert "formatted_mm_to_pixel" in frames[0].metadata_source


def test_formatted_h5_jump_returns_correct_physical_cod(tmp_path):
    path = tmp_path / "cod_CrackVision.h5"
    h = w = 61
    center = 30
    shape = (1, h, w)

    u = np.zeros(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    v[:, center + 1 :, :] = 0.05  # 0.050 mm true crack opening
    exx = np.zeros(shape, dtype=np.float32)
    eyy = np.zeros(shape, dtype=np.float32)
    exy = np.zeros(shape, dtype=np.float32)
    eyy[:, center - 1 : center + 2, 8:-8] = 0.02

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "CrackVision-Ncorr"
        f.attrs["format_version"] = 1
        f.attrs["coordinate_system"] = "reference"
        f.attrs["strain_measure"] = "Green-Lagrange"
        f.attrs["numeric_precision"] = "single"
        f.attrs["source_units"] = "mm"
        f.attrs["pixel_size_mm"] = 0.05
        f.attrs["ncorr_spacing_raw"] = 1.0
        f.attrs["dic_step_px"] = 2.0
        f.attrs["dic_point_spacing_mm"] = 0.10
        fields = f.create_group("fields")
        for name, arr in (("u", u), ("v", v), ("exx", exx), ("eyy", eyy), ("exy", exy)):
            fields.create_dataset(name, data=arr)
        fields.create_dataset("mask", data=np.ones(shape, dtype=np.uint8))
        f.create_dataset("time_s", data=np.array([0.0]))

    frame = CrackVisionNcorrH5Loader.read_nearest_frame(path, 0.0)
    assert np.isclose(frame.v_map[-1, 0], 1.0)

    engine = CrackPhysicsEngine(
        {
            "physics": {
                "detection": {
                    "threshold_k": 1.0,
                    "min_tensile_strain": 1e-5,
                    "min_crack_area_points": 3,
                    "min_crack_length_mm": 0.5,
                    "normal_radius_points": 4,
                },
                "cod": {
                    "near_mm": 0.2,
                    "far_mm": 0.8,
                    "samples_per_side": 7,
                    "min_valid_per_side": 3,
                    "min_samples_per_crack": 3,
                    "min_width_mm": 0.0,
                    "max_width_mm": 1.0,
                },
            }
        }
    )
    summary, details = engine.analyze_frame(frame)
    assert summary["cod_status"] == "ok"
    assert len(details) == 1
    assert np.isclose(details.loc[0, "W_median_mm"], 0.050, atol=2e-3)


def test_bridge_rejects_generic_h5(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(3))

    try:
        list(CrackVisionNcorrH5Loader.stream_frames(path))
    except ValueError as exc:
        assert "not a CrackVision-Ncorr bridge" in str(exc)
    else:
        raise AssertionError("generic H5 should be rejected")
