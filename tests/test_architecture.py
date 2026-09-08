from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.config import with_mm_per_pixel
from src.core.input import input_kind, output_stem
from src.core.models import FrameData
from src.core.pipeline import analyze_file


def test_config_override_does_not_mutate_base_config():
    base = {"experiment": {"mm_per_pixel": 0.045}, "physics": {}}
    updated = with_mm_per_pixel(base, 0.05)
    assert base["experiment"]["mm_per_pixel"] == 0.045
    assert updated["experiment"]["mm_per_pixel"] == 0.05


def test_input_kind_and_output_stem_are_stable():
    assert input_kind(Path("sample.h5")) == "CrackVision-Ncorr H5"
    assert input_kind(Path("sample.mat")) == "original Ncorr MAT"
    assert output_stem(Path("sample_CrackVision.h5")) == "sample"
    assert output_stem(Path("sample.mat")) == "sample"


def test_pipeline_preserves_status_and_applies_time_fallback(monkeypatch):
    shape = (9, 9)
    zeros = np.zeros(shape, dtype=float)
    frame = FrameData(
        frame_id=2,
        u_map=zeros.copy(),
        v_map=zeros.copy(),
        exx_map=zeros.copy(),
        eyy_map=zeros.copy(),
        exy_map=zeros.copy(),
        mask=np.ones(shape, dtype=bool),
        pixel_size_mm=0.05,
        dic_point_spacing_mm=0.15,
        dic_step_px=3.0,
        metadata_source="synthetic",
    )

    monkeypatch.setattr(
        "src.core.pipeline.stream_frames",
        lambda path, config: iter([frame]),
    )

    config = {
        "experiment": {"sampling_interval_s": 5.0},
        "physics": {
            "detection": {
                "threshold_k": 2.0,
                "min_tensile_strain": 0.0002,
                "max_threshold": 0.05,
                "min_crack_area_points": 4,
                "min_crack_length_mm": 0.15,
            },
            "cod": {
                "near_mm": 0.15,
                "far_mm": 0.75,
                "samples_per_side": 7,
                "min_valid_per_side": 3,
                "min_samples_per_crack": 3,
                "min_width_mm": 0.001,
                "max_width_mm": 2.0,
            },
        },
    }

    result = analyze_file(Path("sample.mat"), config)
    assert result is not None
    assert result.frame_count == 1
    assert result.frame_df.loc[0, "cod_status"] == "no_crack_candidate"
    assert result.frame_df.loc[0, "Time_s"] == 10.0
    assert result.frame_df.loc[0, "time_source"] == "frame_index_fallback"
