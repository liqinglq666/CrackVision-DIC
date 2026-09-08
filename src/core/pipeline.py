from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .export import build_qa, export_workbook, prepare_crack_details, prepare_frame_summary
from .input import output_stem, stream_frames
from .models import FrameData
from .physics import CrackPhysicsEngine


@dataclass(slots=True)
class AnalysisResult:
    input_path: Path
    frame_df: pd.DataFrame
    crack_df: pd.DataFrame
    qa_df: pd.DataFrame

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_df))

    @property
    def cod_ok_count(self) -> int:
        if self.frame_df.empty or "cod_status" not in self.frame_df:
            return 0
        return int((self.frame_df["cod_status"] == "ok").sum())

    @property
    def cod_failed_count(self) -> int:
        return self.frame_count - self.cod_ok_count


def analyze_file(
    data_path: Path,
    config: dict[str, Any],
    *,
    should_continue: Callable[[], bool] | None = None,
    on_first_frame: Callable[[FrameData], None] | None = None,
) -> AnalysisResult | None:
    """Run the complete scientific pipeline for one input file.

    Returns ``None`` when cancellation is requested before all frames are analysed.
    """
    path = Path(data_path)
    keep_running = should_continue or (lambda: True)
    fallback_dt = float(config.get("experiment", {}).get("sampling_interval_s", 5.0))
    if fallback_dt <= 0:
        raise ValueError("experiment.sampling_interval_s must be > 0")

    engine = CrackPhysicsEngine(config)
    summaries: list[dict[str, Any]] = []
    detail_tables: list[pd.DataFrame] = []
    first = True

    for frame in stream_frames(path, config):
        if not keep_running():
            return None
        if first:
            if on_first_frame is not None:
                on_first_frame(frame)
            first = False

        summary, details = engine.analyze_frame(frame)
        if not np.isfinite(summary.get("Time_s", np.nan)):
            summary["Time_s"] = frame.frame_id * fallback_dt
            summary["time_source"] = "frame_index_fallback"
        else:
            summary["time_source"] = "input_metadata"

        summaries.append(summary)
        if not details.empty:
            detail_tables.append(details)

    if not summaries:
        raise ValueError("No DIC frames were found in the input file")

    frame_df = prepare_frame_summary(summaries)
    crack_df = prepare_crack_details(detail_tables)
    qa_df = build_qa(frame_df)
    return AnalysisResult(path, frame_df, crack_df, qa_df)


def export_result(result: AnalysisResult, out_dir: Path) -> Path:
    output = Path(out_dir) / f"{output_stem(result.input_path)}_CrackVision.xlsx"
    export_workbook(output, result.frame_df, result.crack_df, result.qa_df)
    return output
