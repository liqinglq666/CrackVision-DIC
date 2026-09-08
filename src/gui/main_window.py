from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.config import load_config, with_mm_per_pixel
from src.core.input import FILE_DIALOG_FILTER
from src.gui.worker import AnalysisWorker


class MainWindow(QMainWindow):
    """Small Ncorr-only UI. Compact bridge HDF5 is the preferred input."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC · Ncorr COD")
        self.resize(900, 650)
        self.data_files: list[Path] = []
        self.out_dir: Path | None = None
        self.worker: AnalysisWorker | None = None
        self.config = load_config()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        title = QLabel("CrackVision-DIC")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        subtitle = QLabel(
            "Ncorr → CrackVision H5 → 最大主拉应变 → 裂缝骨架 → 双侧多点拟合 COD"
        )
        subtitle.setStyleSheet("color:#5f6368;font-size:13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setVerticalSpacing(14)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        choose_files = QPushButton("选择 H5 / MAT")
        choose_files.clicked.connect(self._choose_files)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(choose_files)
        form.addRow("Ncorr 数据", file_row)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        choose_out = QPushButton("选择目录")
        choose_out.clicked.connect(self._choose_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(choose_out)
        form.addRow("输出目录", out_row)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(6)
        self.scale_spin.setRange(0.000001, 1000.0)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setValue(
            float(self.config.get("experiment", {}).get("mm_per_pixel", 0.045))
        )
        self.scale_spin.setSuffix(" mm/px")
        self.scale_spin.setToolTip(
            "仅供旧 Ncorr MAT 缺少 pixtounits 时兜底；CrackVision H5 会读取自身标定。"
        )
        form.addRow("尺度兜底", self.scale_spin)
        layout.addLayout(form)

        note = QLabel(
            "推荐输入：matlab/export_ncorr_to_crackvision.m 生成的 CrackVision-Ncorr H5。"
            "它只保留全过程 U/V/Exx/Eyy/Exy、mask、时间和尺度信息；"
            "巨大原始 Ncorr MAT 仅作为旧数据兼容入口。"
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#f5f7fa;border-radius:8px;padding:12px;color:#374151;"
        )
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.start_button = QPushButton("开始分析")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("运行状态会显示在这里。")
        layout.addWidget(self.log_box, 1)

        self.setStyleSheet(
            "QMainWindow{background:#ffffff;}"
            "QPushButton{padding:8px 14px;}"
            "QLineEdit,QDoubleSpinBox,QTextEdit{"
            "border:1px solid #d7dce2;border-radius:6px;padding:7px;}"
            "QProgressBar{height:12px;border:1px solid #d7dce2;"
            "border-radius:6px;text-align:center;}"
        )

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 CrackVision-Ncorr H5 或原始 Ncorr MAT",
            "",
            FILE_DIALOG_FILTER,
        )
        if not files:
            return

        self.data_files = [Path(path) for path in files]
        self.file_edit.setText(
            files[0] if len(files) == 1 else f"已选择 {len(files)} 个数据文件"
        )
        if self.out_dir is None:
            self.out_dir = self.data_files[0].parent / "CrackVision_Output"
            self.out_edit.setText(str(self.out_dir))

    def _choose_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.out_dir = Path(folder)
            self.out_edit.setText(folder)

    def _start(self) -> None:
        if not self.data_files:
            QMessageBox.warning(
                self,
                "缺少输入",
                "请先选择 CrackVision-Ncorr .h5 或原始 Ncorr .mat 文件。",
            )
            return
        if self.out_dir is None:
            QMessageBox.warning(self, "缺少输出目录", "请选择输出目录。")
            return

        run_config = with_mm_per_pixel(self.config, float(self.scale_spin.value()))
        self.worker = AnalysisWorker(self.data_files, self.out_dir, run_config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.progress.setRange(0, len(self.data_files))
        self.progress.setValue(0)
        self.log_box.clear()
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self._append_log("Cancelling after the current frame...")

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)
        bar = self.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "分析失败", message)

    def _on_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()
