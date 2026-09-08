from __future__ import annotations

import h5py
import numpy as np

from src.core.io_bridge import CrackVisionNcorrH5Loader


def test_lightweight_bridge_streams_frame_major_h5(tmp_path):
    path = tmp_path / "specimen_CrackVision.h5"
    shape = (2, 7, 9)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "CrackVision-Ncorr"
        f.attrs["format_version"] = 1
        f.attrs["coordinate_system"] = "reference"
        f.attrs["strain_measure"] = "Green-Lagrange"
        f.attrs["numeric_precision"] = "single"
        f.attrs["pixel_size_mm"] = 0.05
        f.attrs["ncorr_spacing_raw"] = 2.0
        f.attrs["dic_step_px"] = 3.0
        f.attrs["dic_point_spacing_mm"] = 0.15

        fields = f.create_group("fields")
        for name in ("u", "v", "exx", "eyy", "exy"):
            arr = np.zeros(shape, dtype=np.float32)
            if name == "u":
                arr[1] = 1.25
            fields.create_dataset(name, data=arr)
        fields.create_dataset("mask", data=np.ones(shape, dtype=np.uint8))
        f.create_dataset("time_s", data=np.array([[0.0], [5.0]]))

    frames = list(CrackVisionNcorrH5Loader.stream_frames(path))
    assert len(frames) == 2
    assert frames[0].u_map.shape == (7, 9)
    assert np.isclose(frames[1].u_map[0, 0], 1.25)
    assert np.isclose(frames[0].pixel_size_mm, 0.05)
    assert np.isclose(frames[0].dic_point_spacing_mm, 0.15)
    assert np.isclose(frames[1].time_s, 5.0)
    assert frames[0].metadata_source.startswith("crackvision_ncorr_h5")


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
