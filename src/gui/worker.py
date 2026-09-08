from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.pipeline import analyze_peak_frame, export_result

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """One-shot worker for one Ncorr + one MTS specimen pair."""

    completed = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        data_path: Path,
        mts_path: Path,
        out_dir: Path,
        config: dict,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.mts_path = Path(mts_path)
        self.out_dir = Path(out_dir)
        self.config = config

    @staticmethod
    def _number(row, key: str) -> float | None:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            result = analyze_peak_frame(self.data_path, self.mts_path, self.config)
            output = export_result(result, self.out_dir)

            row = result.frame_df.iloc[0]
            self.completed.emit(
                {
                    "output_path": str(output),
                    "peak_force_N": result.selection.mts_peak_force_N,
                    "peak_time_s": result.selection.mts_peak_time_s,
                    "selected_frame": result.selection.selected_frame_id,
                    "match_error_s": result.selection.match_error_s,
                    "crack_count": int(row.get("crack_count", 0) or 0),
                    "mean_width_um": self._number(row, "Crack_width_mean_um"),
                    "p95_width_um": self._number(row, "Crack_width_95_um"),
                    "max_width_um": self._number(row, "Crack_width_max_um"),
                    "cod_status": str(row.get("cod_status", "unknown")),
                }
            )
        except Exception as exc:
            logger.exception("Peak-frame analysis failed")
            self.failed.emit(str(exc))
