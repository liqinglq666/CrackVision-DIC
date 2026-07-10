from __future__ import annotations

import numpy as np

import src.core.physics as physics
from src.core.physics import CrackPhysicsEngine


def test_frame_w95_uses_all_valid_cod_samples(monkeypatch):
    config = {
        "physics": {
            "require_v_map_for_cod": False,
            "cod_min_mm": 0.0,
            "cod_min_mean_mm": 0.0,
            "cod_max_mm": 20.0,
            "min_crack_length_mm": 0.0,
        }
    }
    engine = CrackPhysicsEngine(config)
    skeleton = np.zeros((12, 12), dtype=bool)
    skeleton[2:5, 2] = True
    skeleton[7:10, 8] = True
    widths = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 10.0])

    monkeypatch.setattr(
        physics,
        "_fast_cod_kernel",
        lambda *args, **kwargs: (widths.copy(), np.arange(len(widths))),
    )

    result = engine.compute_cod(
        np.zeros_like(skeleton, dtype=float),
        skeleton,
        displacement_scale_mm=1.0,
        dic_point_spacing_mm=1.0,
    )

    assert result["cod_status"] == "ok"
    assert result["w_95"] == np.percentile(widths, 95)
    assert result["w_99"] == np.percentile(widths, 99)


def test_missing_v_is_not_reported_as_zero_width():
    engine = CrackPhysicsEngine({"physics": {"require_v_map_for_cod": True}})
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True

    result = engine.compute_cod(
        np.zeros((7, 7), dtype=float),
        skeleton,
        displacement_scale_mm=0.05,
        dic_point_spacing_mm=0.05,
    )

    assert result["cod_status"] == "missing_v_map_required"
    assert np.isnan(result["w_avg"])
    assert np.isnan(result["w_95"])
