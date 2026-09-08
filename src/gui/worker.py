from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.core.input import input_kind
from src.core.models import FrameData
from src.core.pipeline import analyze_file, export_result

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Qt thread adapter around the pure core analysis pipeline."""

    progress = Signal(int, int)
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, data_files: list[Path], out_dir: Path, config: dict) -> None:
        super().__init__()
        self.data_files = [Path(path) for path in data_files]
        self.out_dir = Path(out_dir)
        self.config = config
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.data_files)
            for index, data_path in enumerate(self.data_files, start=1):
                if not self._running:
                    self.log.emit("Analysis cancelled.")
                    return
                try:
                    self._process_one(data_path)
                except Exception as exc:
                    logger.exception("Failed to analyse %s", data_path)
                    self.log.emit(f"❌ {data_path.name}: {exc}")
                self.progress.emit(index, total)
        except Exception as exc:
            logger.exception("Worker failed")
            self.failed.emit(str(exc))

    def _process_one(self, data_path: Path) -> None:
        self.log.emit(f"▶ {data_path.name} [{input_kind(data_path)}]")
        result = analyze_file(
            data_path,
            self.config,
            should_continue=lambda: self._running,
            on_first_frame=self._log_metadata,
        )
        if result is None:
            return

        output = export_result(result, self.out_dir)
        self.log.emit(
            f"✓ {data_path.name}: {result.frame_count} frames, "
            f"COD ok={result.cod_ok_count}, not measurable={result.cod_failed_count}"
        )
        self.log.emit(f"Saved: {output.name}")

    def _log_metadata(self, frame: FrameData) -> None:
        step_text = "n/a" if frame.dic_step_px is None else f"{frame.dic_step_px:.6g}"
        self.log.emit(
            "Scale | "
            f"{frame.pixel_size_mm:.6g} mm/px; "
            f"DIC step={step_text} px; "
            f"grid={frame.dic_point_spacing_mm:.6g} mm/point; "
            f"source={frame.metadata_source}"
        )
