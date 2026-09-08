from __future__ import annotations

from pathlib import Path

from src.core.config import with_mm_per_pixel
from src.core.input import input_kind, output_stem


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
