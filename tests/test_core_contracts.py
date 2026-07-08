import numpy as np
import pandas as pd

from src.core.io_sync import PipelineIO
from src.core.physics import CrackPhysicsEngine
from src.gui.worker import AnalysisPipelineWorker


def test_missing_v_map_returns_explicit_status_when_required():
    config = {
        "physics": {
            "require_v_map_for_cod": True,
            "cod_sampling": {"delta_points": 1, "max_search_points": 3},
        },
        "quality": {},
    }
    engine = CrackPhysicsEngine(config)
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    u = np.zeros((7, 7), dtype=float)

    result = engine.compute_cod(u, skeleton, 0.05, 0.05, v_map=None)

    assert result["crack_count"] == 0
    assert result["cod_status"] == "missing_v_map_required"


def test_frame_count_mismatch_is_not_silently_truncated():
    try:
        PipelineIO._ensure_equal_frame_count([object(), object()], [object()])
    except ValueError as exc:
        assert "Refusing to silently truncate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched DIC frame lists")


def test_frame_time_falls_back_to_interval_when_missing():
    class DummyFrame:
        frame_id = 4
        time_s = np.nan

    time_s, source = AnalysisPipelineWorker._resolve_frame_time(DummyFrame(), 2.5)

    assert time_s == 10.0
    assert source == "frame_index_interval_fallback"


def test_frame_time_prefers_mat_metadata():
    class DummyFrame:
        frame_id = 4
        time_s = 1.25

    time_s, source = AnalysisPipelineWorker._resolve_frame_time(DummyFrame(), 2.5)

    assert time_s == 1.25
    assert source == "mat_metadata_time"


def test_export_frame_table_drops_object_payloads_and_keeps_plot_columns():
    df = pd.DataFrame(
        [
            {
                "Frame": 0,
                "Time_s": 0.0,
                "global_strain": 0.0,
                "w_avg": 0.0,
                "w_99": 0.0,
                "w_max": 0.0,
                "crack_count": 0,
                "raw_widths": np.array([]),
                "per_crack_details": pd.DataFrame(),
                "cod_status": "no_skeleton",
                "sync_status": "no_mts",
            },
            {
                "Frame": 1,
                "Time_s": 5.0,
                "global_strain": 0.02,
                "w_avg": 0.012,
                "w_99": 0.030,
                "w_max": 0.040,
                "crack_count": 3,
                "raw_widths": np.array([0.012]),
                "per_crack_details": pd.DataFrame({"Crack_ID": [1]}),
                "cod_status": "ok",
                "sync_status": "synced",
            },
        ]
    )

    out = AnalysisPipelineWorker._prepare_frame_table("S1", df)

    assert "raw_widths" not in out.columns
    assert "per_crack_details" not in out.columns
    assert out.loc[1, "Strain_pct"] == 2.0
    assert out.loc[1, "W_avg_um"] == 12.0
    assert out.loc[1, "W_99_um"] == 30.0
    assert out.loc[1, "W_max_um"] == 40.0
    assert out.loc[1, "Normalized_Strain"] == 1.0


def test_target_state_table_keeps_not_reached_rows():
    worker = AnalysisPipelineWorker({}, None, {"export": {"target_strains": [1.0, 3.0]}})
    frame_df = pd.DataFrame(
        [
            {
                "Specimen": "S1",
                "Frame": 0,
                "Time_s": 0.0,
                "Strain_pct": 0.0,
                "Normalized_Strain": 0.0,
                "crack_count": 0,
                "crack_spacing_mm": 0.0,
                "W_avg_um": 0.0,
                "W_99_um": 0.0,
                "W_max_um": 0.0,
            },
            {
                "Specimen": "S1",
                "Frame": 1,
                "Time_s": 5.0,
                "Strain_pct": 2.0,
                "Normalized_Strain": 1.0,
                "crack_count": 4,
                "crack_spacing_mm": 20.0,
                "W_avg_um": 15.0,
                "W_99_um": 35.0,
                "W_max_um": 40.0,
            },
        ]
    )

    targets = worker._build_target_state_table("S1", frame_df)

    assert list(targets["Status"]) == ["reached", "not_reached"]
    assert list(targets["Target_Strain_pct"]) == [1.0, 3.0]
    assert targets.loc[1, "Real_Strain_pct"] == 2.0


def test_crack_distribution_is_tidy_long_format():
    crack_tidy = pd.DataFrame(
        [
            {
                "Specimen": "S1",
                "State": "Ultimate",
                "Crack_ID": 1,
                "Frame": 10,
                "Real_Strain_pct": 2.5,
                "W_avg_um": 12.0,
                "W_max_um": 30.0,
            }
        ]
    )

    dist = AnalysisPipelineWorker._build_distribution_table(crack_tidy)

    assert list(dist["Metric"]) == ["W_avg_um", "W_max_um"]
    assert list(dist["Value_um"]) == [12.0, 30.0]
    assert set(["Specimen", "State", "Metric", "Value_um", "Crack_ID"]).issubset(dist.columns)
