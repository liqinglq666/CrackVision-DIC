import numpy as np

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
