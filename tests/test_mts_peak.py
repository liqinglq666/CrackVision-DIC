from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.core.mts import read_mts_peak
from src.core.pipeline import analyze_peak_frame, apply_equal_weight_crack_summary


def _write_mts_csv(path: Path, forces: list[float], times: list[float]) -> None:
    lines = [
        '"文件路径: synthetic"',
        '"试验: ECC拉伸"',
        '"试验运行: Test Run 1"',
        '"日期: 2026/1/1 00:00:00"',
        '""',
        '""',
        '"横梁 ","力 ","时间 ","引伸计 "',
        '"mm","N","sec","mm"',
    ]
    for i, (force, time_s) in enumerate(zip(forces, times)):
        lines.append(f'"{i}","{force}","{time_s}","0"')
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _write_bridge(path: Path) -> None:
    shape = (4, 9, 9)
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
            fields.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
        fields.create_dataset("mask", data=np.ones(shape, dtype=np.uint8))
        f.create_dataset("time_s", data=np.array([0.0, 5.0, 10.0, 15.0]))


def test_parser_reads_real_mts_style_preamble_and_peak(tmp_path):
    csv_path = tmp_path / "mts.csv"
    _write_mts_csv(csv_path, [10.0, 20.0, 55.0, 40.0], [0.0, 1.0, 2.0, 3.0])
    peak = read_mts_peak(csv_path)
    assert peak.peak_force_N == 55.0
    assert peak.peak_time_s == 2.0
    assert peak.tension_sign == 1
    assert peak.force_column == "力"
    assert peak.time_column == "时间"


def test_parser_supports_negative_tension_sign(tmp_path):
    csv_path = tmp_path / "mts_negative.csv"
    _write_mts_csv(csv_path, [-2.0, -20.0, -70.0, -40.0], [0.0, 1.0, 2.0, 3.0])
    peak = read_mts_peak(csv_path)
    assert peak.peak_force_N == -70.0
    assert peak.peak_time_s == 2.0
    assert peak.tension_sign == -1


def test_pipeline_analyses_only_nearest_peak_stress_frame(tmp_path):
    h5_path = tmp_path / "sample_CrackVision.h5"
    mts_path = tmp_path / "mts.csv"
    _write_bridge(h5_path)
    _write_mts_csv(mts_path, [1.0, 2.0, 9.0, 3.0], [0.0, 4.0, 12.0, 16.0])

    config = {
        "experiment": {"sampling_interval_s": 5.0, "mm_per_pixel": 0.05},
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

    result = analyze_peak_frame(h5_path, mts_path, config)
    assert result is not None
    assert len(result.frame_df) == 1
    assert int(result.frame_df.loc[0, "Frame"]) == 2
    assert result.selection.selected_dic_time_s == 10.0
    assert result.selection.mts_peak_time_s == 12.0
    assert result.selection.match_error_s == -2.0
    assert result.frame_df.loc[0, "cod_status"] == "no_crack_candidate"


def test_equal_weight_summary_uses_one_representative_width_per_crack():
    summary = {
        "W_avg_mm": 0.080,
        "W_median_mm": 0.070,
        "W_95_mm": 0.150,
        "W_max_mm": 0.200,
    }
    details = pd.DataFrame(
        {
            "Crack_ID": [1, 2, 3],
            # These are the representative median widths of three cracks.
            "W_median_mm": [0.020, 0.040, 0.100],
            # Different local sample counts must not change crack weights.
            "COD_samples": [100, 5, 3],
        }
    )

    out = apply_equal_weight_crack_summary(summary, details)

    assert out["crack_width_basis"] == "equal_weight_per_crack_W_median"
    assert out["crack_representative_count"] == 3
    assert np.isclose(out["Crack_width_mean_mm"], (0.020 + 0.040 + 0.100) / 3)
    assert np.isclose(out["Crack_width_median_mm"], 0.040)
    assert np.isclose(out["Crack_width_max_mm"], 0.100)
    assert np.isclose(out["W_avg_mm"], out["Crack_width_mean_mm"])
    assert np.isclose(out["W_median_mm"], out["Crack_width_median_mm"])
