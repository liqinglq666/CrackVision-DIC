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
    mts_tension_sign: int
    selected_frame_id: int
    selected_dic_time_s: float
    match_error_s: float


@dataclass(slots=True)
class AnalysisResult:
    input_path: Path
    mts_path: Path
    frame_df: pd.DataFrame
    crack_df: pd.DataFrame
    selection: PeakFrameSelection

    @property
    def cod_status(self) -> str:
        if self.frame_df.empty or "cod_status" not in self.frame_df:
            return "unknown"
        return str(self.frame_df.iloc[0]["cod_status"])


def apply_equal_weight_crack_summary(
    summary: dict[str, Any],
    details: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize accepted cracks with one equal-weight width per crack."""
    out = dict(summary)
    out["crack_width_basis"] = "equal_weight_per_crack_W_median"
    out["crack_representative_count"] = 0

    for field in (
        "Crack_width_mean_mm",
        "Crack_width_median_mm",
        "Crack_width_95_mm",
        "Crack_width_max_mm",
    ):
        out[field] = float("nan")

    if details is None or details.empty or "W_median_mm" not in details.columns:
        return out

    widths = pd.to_numeric(details["W_median_mm"], errors="coerce").to_numpy(dtype=float)
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return out

    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    p95_width = float(np.percentile(widths, 95))
    max_width = float(np.max(widths))

    out.update(
        {
            "crack_representative_count": int(widths.size),
            "Crack_width_mean_mm": mean_width,
            "Crack_width_median_mm": median_width,
            "Crack_width_95_mm": p95_width,
            "Crack_width_max_mm": max_width,
            # Internal compatibility aliases. The workbook exposes only the
            # explicit paper-facing Crack_width_* fields.
            "W_avg_mm": mean_width,
            "W_median_mm": median_width,
            "W_95_mm": p95_width,
            "W_max_mm": max_width,
        }
    )
    return out


def analyze_peak_frame(
    data_path: Path,
    mts_csv_path: Path,
    config: dict[str, Any],
) -> AnalysisResult:
    """Analyse only the DIC frame nearest to the MTS peak tensile-force time.

    The project workflow assumes MTS and image acquisition start together, so
    both time axes share t=0 and no user-entered synchronization offset exists.
    """
    mts_peak: MtsPeak = read_mts_peak(Path(mts_csv_path))
    target_dic_time_s = float(mts_peak.peak_time_s)

    selected: SelectedFrame = select_nearest_frame(
        Path(data_path), config, target_dic_time_s
    )

    engine = CrackPhysicsEngine(config)
    summary, details = engine.analyze_frame(selected.frame)
    summary = apply_equal_weight_crack_summary(summary, details)

    match_error_s = float(selected.dic_time_s - mts_peak.peak_time_s)
    selection = PeakFrameSelection(
        mts_peak_force_N=float(mts_peak.peak_force_N),
        mts_peak_time_s=float(mts_peak.peak_time_s),
        mts_tension_sign=int(mts_peak.tension_sign),
        selected_frame_id=int(selected.frame.frame_id),
        selected_dic_time_s=float(selected.dic_time_s),
        match_error_s=match_error_s,
    )

    summary.update(
        {
            "Time_s": float(selected.dic_time_s),
            "time_source": selected.time_source,
            "selection_mode": "nearest_dic_frame_to_mts_peak_tensile_force",
            "MTS_peak_force_N": float(mts_peak.peak_force_N),
            "MTS_peak_time_s": float(mts_peak.peak_time_s),
            "MTS_tension_sign": int(mts_peak.tension_sign),
            "DIC_selected_time_s": float(selected.dic_time_s),
            "frame_match_error_s": match_error_s,
        }
    )

    return AnalysisResult(
        input_path=Path(data_path),
        mts_path=Path(mts_csv_path),
        frame_df=prepare_frame_summary([summary]),
        crack_df=prepare_crack_details([details] if not details.empty else []),
        selection=selection,
    )


def export_result(result: AnalysisResult, out_dir: Path) -> Path:
    output = Path(out_dir) / f"{output_stem(result.input_path)}_CrackVision.xlsx"
    export_workbook(output, result.frame_df, result.crack_df)
    return output
