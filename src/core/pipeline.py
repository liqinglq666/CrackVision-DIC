from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .export import export_workbook, prepare_crack_details, prepare_frame_summary
from .input import SelectedFrame, output_stem, select_nearest_frame
from .mts import MtsPeak, read_mts_peak
from .physics import CrackPhysicsEngine


@dataclass(frozen=True, slots=True)
class PeakFrameSelection:
    mts_peak_force_N: float
    mts_peak_time_s: float
    selected_frame_id: int
    selected_dic_time_s: float
    match_error_s: float


@dataclass(slots=True)
class AnalysisResult:
    input_path: Path
    frame_df: pd.DataFrame
    crack_df: pd.DataFrame
    selection: PeakFrameSelection


def apply_equal_weight_crack_summary(
    summary: dict[str, Any],
    details: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize accepted cracks with one equal-weight representative width each."""
    out = dict(summary)
    for field in (
        "Crack_width_mean_mm",
        "Crack_width_median_mm",
        "Crack_width_95_mm",
        "Crack_width_max_mm",
    ):
        out[field] = float("nan")

    if details.empty or "W_median_mm" not in details.columns:
        return out

    widths = pd.to_numeric(details["W_median_mm"], errors="coerce").to_numpy(dtype=float)
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return out

    out.update(
        {
            "Crack_width_mean_mm": float(np.mean(widths)),
            "Crack_width_median_mm": float(np.median(widths)),
            "Crack_width_95_mm": float(np.percentile(widths, 95)),
            "Crack_width_max_mm": float(np.max(widths)),
        }
    )
    return out


def analyze_peak_frame(
    data_path: Path,
    mts_csv_path: Path,
    config: dict[str, Any],
) -> AnalysisResult:
    """Analyse only the DIC frame nearest to the MTS peak tensile-force time."""
    mts_peak: MtsPeak = read_mts_peak(Path(mts_csv_path))
    selected: SelectedFrame = select_nearest_frame(
        Path(data_path), config, float(mts_peak.peak_time_s)
    )

    engine = CrackPhysicsEngine(config)
    summary, details = engine.analyze_frame(selected.frame)
    summary = apply_equal_weight_crack_summary(summary, details)

    match_error_s = float(selected.dic_time_s - mts_peak.peak_time_s)
    selection = PeakFrameSelection(
        mts_peak_force_N=float(mts_peak.peak_force_N),
        mts_peak_time_s=float(mts_peak.peak_time_s),
        selected_frame_id=int(selected.frame.frame_id),
        selected_dic_time_s=float(selected.dic_time_s),
        match_error_s=match_error_s,
    )

    summary.update(
        {
            "Time_s": float(selected.dic_time_s),
            "MTS_peak_force_N": float(mts_peak.peak_force_N),
            "MTS_peak_time_s": float(mts_peak.peak_time_s),
            "DIC_selected_time_s": float(selected.dic_time_s),
            "frame_match_error_s": match_error_s,
        }
    )

    return AnalysisResult(
        input_path=Path(data_path),
        frame_df=prepare_frame_summary(summary),
        crack_df=prepare_crack_details(details),
        selection=selection,
    )


def export_result(result: AnalysisResult, out_dir: Path) -> Path:
    output = Path(out_dir) / f"{output_stem(result.input_path)}_CrackVision.xlsx"
    export_workbook(output, result.frame_df, result.crack_df)
    return output
