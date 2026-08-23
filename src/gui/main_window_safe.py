from __future__ import annotations

from src.gui import main_window as _base
from src.gui.worker_safe import AnalysisPipelineWorker

_base.AnalysisPipelineWorker = AnalysisPipelineWorker
MainWindow = _base.MainWindow
