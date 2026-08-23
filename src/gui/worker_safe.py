from __future__ import annotations

import numpy as np
import pandas as pd

from src.gui import worker as _base

_base_task = _base.analyze_single_frame_task

_INVALID_COD = {
    "missing_v_map_required",
    "insufficient_cod_samples",
    "cod_out_of_range",
    "cod_floor_removed_all",
    "quality_rejected",
}
_INVALID_STRAIN = {
    "no_valid_columns",
    "invalid_extensometer",
    "missing_band_displacement",
    "invalid_gauge_length",
}
_WIDTH_COLUMNS = [
    "W_avg_um",
    "W_median_um",
    "W_95_um",
    "W_99_um",
    "W_max_um",
]
_STRAIN_COLUMNS = [
    "global_strain",
    "Strain_pct",
    "Normalized_Strain",
    "W_global_est_um",
]


def analyze_single_frame_task(payload):
    result = _base_task(payload)
    if result and result.get("strain_source") in _INVALID_STRAIN:
        result["global_strain"] = np.nan
    return result


# ProcessPoolExecutor pickles this module-level wrapper, so spawned workers get the same guard.
_base.analyze_single_frame_task = analyze_single_frame_task


class AnalysisPipelineWorker(_base.AnalysisPipelineWorker):
    @staticmethod
    def _prepare_frame_table(specimen: str, df: pd.DataFrame) -> pd.DataFrame:
        out = _base.AnalysisPipelineWorker._prepare_frame_table(specimen, df)

        if "cod_status" in out.columns:
            invalid_cod = out["cod_status"].isin(_INVALID_COD)
            out.loc[invalid_cod, _WIDTH_COLUMNS] = np.nan

        if "strain_source" in out.columns:
            invalid_strain = out["strain_source"].isin(_INVALID_STRAIN)
            out.loc[invalid_strain, _STRAIN_COLUMNS] = np.nan

        return out
