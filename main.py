from __future__ import annotations

import logging
import multiprocessing
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Type

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.gui.main_window_safe import MainWindow


def setup_global_logging() -> None:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_dir / "app_runtime.log", encoding="utf-8", mode="a")
    ]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
        force=True,
        handlers=handlers,
    )
    logging.info("CrackVision-DIC started")


def global_exception_handler(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType,
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical("Unhandled exception:\n%s", error_msg)


def main() -> None:
    setup_global_logging()
    sys.excepthook = global_exception_handler

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("CrackVision-DIC")
    app.setApplicationVersion("2.0.0-Core")
    app.setOrganizationName("ScientificMechanics")
    app.setStyle("Fusion")

    try:
        window = MainWindow()
        window.show()
        exit_code = app.exec()
        logging.info("Event loop ended with code %d", exit_code)
        sys.exit(exit_code)
    except Exception as exc:
        logging.critical("Main window crashed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Windows/PyInstaller 不加这个会递归拉起子进程。
    multiprocessing.freeze_support()
    main()
