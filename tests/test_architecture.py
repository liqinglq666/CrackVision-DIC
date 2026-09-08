from __future__ import annotations

from pathlib import Path

from src.core.config import load_config
from src.core.input import output_stem


def test_default_config_loads_as_mapping():
    config = load_config()
    assert isinstance(config, dict)
    assert "experiment" in config
    assert "physics" in config


def test_output_stem_is_stable():
    assert output_stem(Path("sample_CrackVision.h5")) == "sample"
    assert output_stem(Path("sample.mat")) == "sample"
