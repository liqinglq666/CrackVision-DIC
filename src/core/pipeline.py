from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .export import build_qa, export_workbook, prepare_crack_details, prepare_frame_summary
from .input import SelectedFrame, output_stem, select_nearest_frame
from .mts import MtsPeak, read_mts_peak
from .physics import CrackPhysicsEngine


@dataclass(frozen=True, slots=True)
class PeakFrameSelection:
    mts_peak_force_N: float
    mts_peak_time_s: float
    mts_tension_sign: int
    dic_frame0_mts_time_s: float
    target_dic_time_s: float
    selected_frame_id: int
    selected_dic_time_s: float
    selected_mts_time_s: float
    match_error_s: float


@dataclass(slots=True)
class AnalysisResult:
    input_path: Path
    mts_path: Path
    frame_df: pd.DataFrame
    crack_df: pd.DataFrame
    qa_df: pd.DataFrame
    selection: PeakFrameSelection

    @property
    def cod_status(self) -> str:
        if self.frame_df.empty or "cod_status" not in self.frame_df:
            return "unknown"
        return str(self.frame_df.iloc[0]["cod_status"])


def analyze_peak_frame(
    data_path: Path,
    mts_csv_path: Path,
    config: dict[str, Any],
    *,
    dic_frame0_mts_time_s: float = 0.0,
    should_continue: Callable[[], bool] | None = None,
) -> AnalysisResult | None:
    """Analyse only the DIC frame nearest to MTS peak tensile stress/force."""
    keep_running = should_continue or (lambda: True)
    if not keep_running():
        return None

    mts_peak: MtsPeak = read_mts_peak(Path(mts_csv_path))
    target_dic_time_s = float(mts_peak.peak_time_s - dic_frame0_mts_time_s)

    selected: SelectedFrame = select_nearest_frame(
        Path(data_path), config, target_dic_time_s
    )
    if not keep_running():
        return None

    engine = CrackPhysicsEngine(config)
    summary, details = engine.analyze_frame(selected.frame)

    selected_mts_time_s = float(selected.dic_time_s + dic_frame0_mts_time_s)
    match_error_s = float(selected_mts_time_s - mts_peak.peak_time_s)
    selection = PeakFrameSelection(
        mts_peak_force_N=float(mts_peak.peak_force_N),
        mts_peak_time_s=float(mts_peak.peak_time_s),
        mts_tension_sign=int(mts_peak.tension_sign),
        dic_frame0_mts_time_s=float(dic_frame0_mts_time_s),
        target_dic_time_s=target_dic_time_s,
        selected_frame_id=int(selected.frame.frame_id),
        selected_dic_time_s=float(selected.dic_time_s),
        selected_mts_time_s=selected_mts_time_s,
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
            "DIC_frame0_MTS_time_s": float(dic_frame0_mts_time_s),
            "DIC_selected_time_s": float(selected.dic_time_s),
            "MTS_time_at_selected_DIC_frame_s": selected_mts_time_s,
            "frame_match_error_s": match_error_s,
        }
    )

    frame_df = prepare_frame_summary([summary])
    crack_df = prepare_crack_details([details] if not details.empty else [])
    qa_df = build_qa(frame_df)
    qa_extra = pd.DataFrame(
        [
            {"Metric": "selection_mode", "Value": summary["selection_mode"]},
            {"Metric": "MTS_peak_force_N", "Value": mts_peak.peak_force_N},
            {"Metric": "MTS_peak_time_s", "Value": mts_peak.peak_time_s},
            {"Metric": "MTS_tension_sign", "Value": mts_peak.tension_sign},
            {"Metric": "DIC_frame0_MTS_time_s", "Value": dic_frame0_mts_time_s},
            {"Metric": "selected_DIC_frame", "Value": selected.frame.frame_id},
            {"Metric": "selected_DIC_time_s", "Value": selected.dic_time_s},
            {"Metric": "frame_match_error_s", "Value": match_error_s},
        ]
    )
    qa_df = pd.concat([qa_df, qa_extra], ignore_index=True)

    return AnalysisResult(
        input_path=Path(data_path),
        mts_path=Path(mts_csv_path),
        frame_df=frame_df,
        crack_df=crack_df,
        qa_df=qa_df,
        selection=selection,
    )


def export_result(result: AnalysisResult, out_dir: Path) -> Path:
    output = Path(out_dir) / f"{output_stem(result.input_path)}_CrackVision.xlsx"
    export_workbook(output, result.frame_df, result.crack_df, result.qa_df)
    return output
