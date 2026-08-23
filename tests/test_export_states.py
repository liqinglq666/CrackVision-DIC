from __future__ import annotations

import numpy as np
import pandas as pd

import src.gui.worker_safe as safe_worker


def test_invalid_cod_and_strain_export_as_nan():
    df = pd.DataFrame(
        [
            {
                "Frame": 1,
                "Time_s": 5.0,
                "global_strain": 0.0,
                "w_avg": 0.0,
                "w_median": 0.0,
                "w_95": 0.0,
                "w_99": 0.0,
                "w_max": 0.0,
                "W_global_est_um": 0.0,
                "crack_count": 0,
                "cod_status": "missing_v_map_required",
                "strain_source": "missing_band_displacement",
            }
        ]
    )

    out = safe_worker.AnalysisPipelineWorker._prepare_frame_table("S1", df)

    assert np.isnan(out.loc[0, "W_avg_um"])
    assert np.isnan(out.loc[0, "W_95_um"])
    assert np.isnan(out.loc[0, "global_strain"])
    assert np.isnan(out.loc[0, "Strain_pct"])
    assert np.isnan(out.loc[0, "W_global_est_um"])


def test_task_marks_invalid_virtual_extensometer(monkeypatch):
    monkeypatch.setattr(
        safe_worker,
        "_base_task",
        lambda payload: {"global_strain": 0.0, "strain_source": "no_valid_columns"},
    )

    result = safe_worker.analyze_single_frame_task(object())

    assert np.isnan(result["global_strain"])
