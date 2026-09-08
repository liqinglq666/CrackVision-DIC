from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from src.core.export import build_qa, export_workbook, prepare_crack_details, prepare_frame_summary
from src.core.io_ncorr import NcorrLoader
from src.core.physics import CrackPhysicsEngine

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    specimen_finished = Signal(str)
    failed = Signal(str)

    def __init__(self, mat_files: list[Path], out_dir: Path, config: dict) -> None:
        super().__init__()
        self.mat_files = [Path(p) for p in mat_files]
        self.out_dir = Path(out_dir)
        self.config = config
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.mat_files)
            for index, mat_path in enumerate(self.mat_files, start=1):
                if not self._running:
                    self.log.emit("Analysis cancelled.")
                    return
                try:
                    self._process_one(mat_path)
                except Exception as exc:
                    logger.exception("Failed to analyse %s", mat_path)
                    self.log.emit(f"❌ {mat_path.name}: {exc}")
                self.progress.emit(index, total)
        except Exception as exc:
            logger.exception("Worker failed")
            self.failed.emit(str(exc))

    def _process_one(self, mat_path: Path) -> None:
        exp = self.config.get("experiment", {})
        fallback_ratio = float(exp.get("mm_per_pixel", 0.045))
        fallback_dt = float(exp.get("sampling_interval_s", 1.0))
        if fallback_ratio <= 0 or fallback_dt <= 0:
            raise ValueError("mm_per_pixel and sampling_interval_s must be > 0")

        engine = CrackPhysicsEngine(self.config)
        summaries: list[dict] = []
        detail_tables: list[pd.DataFrame] = []
        first_metadata_logged = False

        self.log.emit(f"▶ {mat_path.name}")
        for frame in NcorrLoader.stream_frames(mat_path, fallback_ratio, self.config):
            if not self._running:
                return
            if not first_metadata_logged:
                self.log.emit(
                    "Scale | "
                    f"{frame.pixel_size_mm:.6g} mm/px; "
                    f"DIC step={frame.dic_step_px:.6g} px; "
                    f"grid={frame.dic_point_spacing_mm:.6g} mm/point; "
                    f"source={frame.metadata_source}"
                )
                first_metadata_logged = True

            summary, details = engine.analyze_frame(frame)
            if not np.isfinite(summary.get("Time_s", np.nan)):
                summary["Time_s"] = frame.frame_id * fallback_dt
                summary["time_source"] = "frame_index_fallback"
            else:
                summary["time_source"] = "mat_metadata"
            summaries.append(summary)
            if not details.empty:
                detail_tables.append(details)

        if not summaries:
            raise ValueError("No DIC frames were found in the MAT file")

        frame_df = prepare_frame_summary(summaries)
        crack_df = prepare_crack_details(detail_tables)
        qa_df = build_qa(frame_df)
        output = self.out_dir / f"{mat_path.stem}_CrackVision.xlsx"
        export_workbook(output, frame_df, crack_df, qa_df)

        ok = int((frame_df["cod_status"] == "ok").sum())
        failed = int(len(frame_df) - ok)
        self.log.emit(f"✓ {mat_path.name}: {len(frame_df)} frames, COD ok={ok}, not measurable={failed}")
        self.log.emit(f"Saved: {output.name}")
        self.specimen_finished.emit(str(output))
