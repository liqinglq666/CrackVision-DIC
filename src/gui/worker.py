from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.core.input import input_kind
from src.core.pipeline import analyze_peak_frame, export_result

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Qt thread adapter for one Ncorr + one MTS specimen pair."""

    progress = Signal(int, int)
    log = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        data_path: Path,
        mts_path: Path,
        out_dir: Path,
        config: dict,
        dic_frame0_mts_time_s: float,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.mts_path = Path(mts_path)
        self.out_dir = Path(out_dir)
        self.config = config
        self.dic_frame0_mts_time_s = float(dic_frame0_mts_time_s)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"▶ {self.data_path.name} [{input_kind(self.data_path)}]")
            self.log.emit(f"MTS: {self.mts_path.name}")

            result = analyze_peak_frame(
                self.data_path,
                self.mts_path,
                self.config,
                dic_frame0_mts_time_s=self.dic_frame0_mts_time_s,
                should_continue=lambda: self._running,
            )
            if result is None:
                self.log.emit("Analysis cancelled.")
                return

            selection = result.selection
            sign_text = "+" if selection.mts_tension_sign > 0 else "-"
            self.log.emit(
                "MTS peak | "
                f"force={selection.mts_peak_force_N:.6g} N; "
                f"t={selection.mts_peak_time_s:.3f} s; "
                f"tension sign={sign_text}"
            )
            self.log.emit(
                "Selected DIC | "
                f"Frame={selection.selected_frame_id}; "
                f"DIC t={selection.selected_dic_time_s:.3f} s; "
                f"MTS-equivalent t={selection.selected_mts_time_s:.3f} s; "
                f"Δt={selection.match_error_s:+.3f} s"
            )

            row = result.frame_df.iloc[0]
            step = row.get("dic_step_px")
            step_text = "n/a" if step is None else f"{float(step):.6g}"
            self.log.emit(
                "Scale | "
                f"{float(row['pixel_size_mm']):.6g} mm/px; "
                f"DIC step={step_text} px; "
                f"grid={float(row['dic_point_spacing_mm']):.6g} mm/point"
            )

            output = export_result(result, self.out_dir)
            self.log.emit(f"✓ peak-stress frame COD status={result.cod_status}")
            self.log.emit(f"Saved: {output.name}")
            self.progress.emit(1, 1)
        except Exception as exc:
            logger.exception("Peak-frame analysis failed")
            self.failed.emit(str(exc))
