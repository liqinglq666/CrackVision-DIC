import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

# 🌟 确保所有 UI 组件全部被正确导入
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QMessageBox, QLabel, QDoubleSpinBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QStackedWidget, QTextBrowser
)
from PySide6.QtGui import QDesktopServices, QAction
from PySide6.QtCore import Qt, QUrl

from src.gui.worker import AnalysisPipelineWorker

# NOTE: 模块级日志配置
logger = logging.getLogger(__name__)


class UserManualDialog(QDialog):
    """内置的科研用户说明书与排错指南"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CrackVision-DIC 用户说明书与物理参数指南")
        self.resize(750, 650)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # 使用 HTML 混排撰写工业级说明书
        browser.setHtml("""
        <h2 style='color: #2F3640;'>CrackVision-DIC 物理分析引擎说明书</h2>

        <h3 style='color: #e1b12c;'>一、 核心物理参数释义</h3>
        <ul>
            <li><b>宏观拉伸标距 (Gauge Length)</b>: 极其重要！试验机测量全局位移的物理跨度。若试件标距为 80mm，必须填入 80。填错会导致应变数据（Strain_pct）出现如 17%、20% 的严重虚高。</li>
            <li><b>缺省兜底比例尺 (Scale)</b>: 当引擎无法从 Ncorr 的 .mat 文件底层读取到标定信息时，将使用此数值进行 mm/px 的换算。</li>
            <li><b>底噪拦截防线 (COD Min)</b>: 用于抹杀 DIC 的亚像素计算噪声。建议设为 <b>0.005 mm (5μm)</b>。低于此值的裂缝将被视为“数学幽灵”并在统计时被强行抹除，防止拉低裂缝的真实平均宽度。</li>
            <li><b>MAD 稳健阈值 (k)</b>: 应变场提取骨架的敏感度。默认 <b>2.0</b>。若发现提取的微裂缝过多、过杂，可上调至 2.5 或 3.0；若主裂缝边缘断裂提取不全，可下调至 1.5。</li>
        </ul>

        <h3 style='color: #e84118;'>二、 经典数据异常与排错指南 (Troubleshooting)</h3>
        <ul>
            <li><b>异常 1：导出的 Excel 里裂缝宽度出现 0 或 0.3μm 的极小值？</b><br>
                <i>诊断：</i> 物理底噪防线被击穿或设置过低。请确保 [底噪拦截防线] 至少为 0.005。<br>
                <i>处理：</i> 引擎已内置 2μm 平均宽度和 5μm 最大宽度的绝对底线，此类数据将被自动洗净。
            </li>
            <li><b>异常 2：主裂缝在破坏阶段 (Ultimate) 突然消失，裂缝数量暴跌？</b><br>
                <i>诊断：</i> DIC 失相关黑洞 (Decorrelation)。裂缝张开过大导致散斑剥落，Ncorr 计算出 NaN。<br>
                <i>处理：</i> 本引擎已搭载“雷达跨越探测”与“形态学缝合”技术，自动向外跨越 15 像素黑洞。若依然断裂，说明该试件散斑破坏过于严重。
            </li>
            <li><b>异常 3：导出的 Excel 没有 Fig3 或 Fig4，或者提示 [No Cracks Detected]？</b><br>
                <i>诊断：</i> 试件在该应变节点完全没有开裂（例如在线弹性阶段早期），或者裂缝实在太细被防线全部拦截。<br>
                <i>处理：</i> 属于正常的物理现象，Excel 表格的行列结构会被强制保留并打上空白标签，绝不影响后续使用 MATLAB/Python 脚本自动无脑拼接数据。
            </li>
            <li><b>异常 4：提示“时域同步失败，退回纯 DIC”？</b><br>
                <i>诊断：</i> 挂载的 MTS 曲线时间轴与 DIC 图像时间轴完全没有重合交集，或者 CSV 文件格式被损坏。<br>
                <i>处理：</i> 请检查 MTS 和 DIC 的起始触发时间是否一致。退回纯 DIC 模式后，应力（Stress_MPa）将被视为 0。
            </li>
        </ul>
        <hr>
        <p style='color: #718093; font-size: 12px;'>Engineered for Concrete Mechanics | Built with robust defensive programming.</p>
        """)
        layout.addWidget(browser)

        btn_close = QPushButton("我已了解")
        btn_close.setMinimumHeight(40)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class DataPairingDialog(QDialog):
    def __init__(self, dic_dir: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("批处理: 智能数据对齐控制台")
        self.resize(800, 500)
        self.dic_dir = Path(dic_dir)
        self.paired_data: Dict[str, str] = {}
        self.mts_count = 0
        self._init_ui()
        self._load_dic_files()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 🌟 MTS 一键匹配区
        grp_auto = QGroupBox("智能配对区 (可选)")
        layout_auto = QHBoxLayout(grp_auto)

        self.edit_mts_dir = QLineEdit()
        self.edit_mts_dir.setPlaceholderText("选择包含 MTS .csv 文件的目录以启动一键配对...")
        btn_browse_mts = QPushButton("选择 MTS 目录")
        btn_browse_mts.clicked.connect(self._select_mts_dir)

        btn_auto_match = QPushButton("🤖 一键智能配对")
        btn_auto_match.setStyleSheet("background-color: #2F3640; color: #FFFFFF; font-weight: bold;")
        btn_auto_match.clicked.connect(self._auto_match)

        layout_auto.addWidget(self.edit_mts_dir)
        layout_auto.addWidget(btn_browse_mts)
        layout_auto.addWidget(btn_auto_match)
        layout.addWidget(grp_auto)

        # 🌟 批量勾选操作区
        layout_tools = QHBoxLayout()
        layout_tools.addWidget(QLabel("勾选需要进入计算队列的试件：", styleSheet="font-weight: bold;"))
        layout_tools.addStretch()
        btn_select_all = QPushButton("全部勾选")
        btn_select_all.clicked.connect(lambda: self._toggle_all(Qt.CheckState.Checked))
        btn_deselect_all = QPushButton("全部取消")
        btn_deselect_all.clicked.connect(lambda: self._toggle_all(Qt.CheckState.Unchecked))
        layout_tools.addWidget(btn_select_all)
        layout_tools.addWidget(btn_deselect_all)
        layout.addLayout(layout_tools)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["DIC 数据矩阵 (勾选以计算)", "MTS 时域曲线 (为空即退回纯DIC)", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        btn_confirm = QPushButton("确认勾选并锁定队列")
        btn_confirm.setMinimumHeight(40)
        btn_confirm.clicked.connect(self._on_confirm)
        layout.addWidget(btn_confirm)

    def _toggle_all(self, state: Qt.CheckState) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item: item.setCheckState(state)

    def _select_mts_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择 MTS 力学数据根目录")
        if dir_path:
            self.edit_mts_dir.setText(dir_path)
            self._auto_match()

    def _auto_match(self) -> None:
        mts_dir = self.edit_mts_dir.text().strip()
        if not mts_dir or not Path(mts_dir).exists():
            QMessageBox.information(self, "提示", "请先选择有效的 MTS 目录。")
            return

        csv_files = list(Path(mts_dir).rglob("*.csv")) + list(Path(mts_dir).rglob("*.CSV"))
        csv_dict = {f.stem.lower(): str(f) for f in csv_files}
        match_count = 0

        for row in range(self.table.rowCount()):
            mat_stem = Path(self.table.item(row, 0).text()).stem.lower()
            matched_csv = ""
            for csv_stem, csv_path in csv_dict.items():
                if mat_stem in csv_stem or csv_stem in mat_stem:
                    matched_csv = csv_path
                    match_count += 1
                    break
            self.table.item(row, 1).setText(matched_csv)

        QMessageBox.information(self, "配对完成",
                                f"智能扫描结束。\n尝试匹配: {self.table.rowCount()} 组\n成功匹配: {match_count} 组")

    def _load_dic_files(self) -> None:
        if not self.dic_dir.exists(): return
        mat_files = list(self.dic_dir.rglob("*.mat")) + list(self.dic_dir.rglob("*.MAT"))
        self.table.setRowCount(len(mat_files))

        for row, mat_file in enumerate(mat_files):
            item_mat = QTableWidgetItem(mat_file.name)
            item_mat.setData(Qt.ItemDataRole.UserRole, str(mat_file))
            item_mat.setFlags(item_mat.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_mat.setCheckState(Qt.CheckState.Checked)

            self.table.setItem(row, 0, item_mat)
            self.table.setItem(row, 1, QTableWidgetItem(""))

            btn_browse = QPushButton("手动指定...")
            btn_browse.clicked.connect(lambda checked=False, r=row: self._browse_csv(r))
            self.table.setCellWidget(row, 2, btn_browse)

    def _browse_csv(self, row: int) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择力学文件", "", "CSV Files (*.csv *.CSV)")
        if file_path: self.table.item(row, 1).setText(file_path)

    def _on_confirm(self) -> None:
        self.paired_data.clear()
        self.mts_count = 0
        for row in range(self.table.rowCount()):
            item_mat = self.table.item(row, 0)
            if item_mat.checkState() == Qt.CheckState.Checked:
                mat_path = item_mat.data(Qt.ItemDataRole.UserRole)
                csv_path = self.table.item(row, 1).text().strip()
                self.paired_data[mat_path] = csv_path
                if csv_path: self.mts_count += 1
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CrackVision-DIC Core Engine")
        self.resize(650, 850)

        self.worker: Optional[AnalysisPipelineWorker] = None
        self.paired_dict: Dict[str, str] = {}
        self.config_path = Path("config/default.yaml")
        self.config: dict = {}

        self._load_config()
        self._apply_minimalist_style()
        self._init_ui()
        self._init_menu_bar()

    def _apply_minimalist_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background-color: #F5F6FA; }
            QGroupBox { font-weight: bold; border: 1px solid #DCDDE1; border-radius: 4px; margin-top: 2ex; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #2F3640; }
            QPushButton { background-color: #ECDFE6; border: 1px solid #DCDDE1; padding: 6px; border-radius: 3px; color: #2F3640; }
            QPushButton:hover { background-color: #DCDDE1; }
            QPushButton#BtnStart { background-color: #2F3640; color: #F5F6FA; font-weight: bold; font-size: 11pt; border: none; }
            QPushButton#BtnStart:hover { background-color: #353B48; }
            QPushButton#BtnStart:disabled { background-color: #718093; }
            QLineEdit, QDoubleSpinBox { border: 1px solid #DCDDE1; padding: 5px; border-radius: 3px; background-color: #FFFFFF; }
            QTextEdit { border: 1px solid #DCDDE1; background-color: #2F3640; color: #F5F6FA; font-family: Consolas, monospace; font-size: 9pt; }
            QProgressBar { border: 1px solid #DCDDE1; border-radius: 3px; text-align: center; background-color: #FFFFFF; }
            QProgressBar::chunk { background-color: #44BD32; }
        """)

    def _load_config(self) -> None:
        default_config = {
            'experiment': {'mm_per_pixel': 0.02753, 'sampling_interval_s': 5.0, 'gauge_length_mm': 80.0},
            'physics': {'strain_threshold_k': 2.0, 'cod_min_mm': 0.005}
        }
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = default_config
        except Exception:
            self.config = default_config

    def _init_menu_bar(self) -> None:
        """构建顶层系统菜单，唤起说明书"""
        menubar = self.menuBar()
        help_menu = menubar.addMenu("系统帮助 (&Help)")

        action_manual = QAction("打开用户说明书与排错指南", self)
        action_manual.setShortcut("F1")
        action_manual.triggered.connect(self._show_manual)

        help_menu.addAction(action_manual)

    def _show_manual(self) -> None:
        dialog = UserManualDialog(self)
        dialog.exec()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        self._init_data_group(main_layout)
        self._init_param_group(main_layout)

        self.btn_start = QPushButton("启动物理分析引擎")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self._start_pipeline)
        main_layout.addWidget(self.btn_start)

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(15)
        main_layout.addWidget(self.progress)

        main_layout.addWidget(QLabel("引擎运行日志 (Engine Logs):", styleSheet="font-weight: bold; color: #2F3640;"))
        self.logger_console = QTextEdit()
        self.logger_console.setReadOnly(True)
        main_layout.addWidget(self.logger_console)

    def _init_data_group(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("I/O 挂载配置")
        main_vbox = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.radio_single = QRadioButton("单点计算 (Single)")
        self.radio_batch = QRadioButton("批处理队列 (Batch)")
        self.radio_single.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_batch)
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_batch)
        main_vbox.addLayout(mode_layout)

        self.stack_data = QStackedWidget()

        page_single = QWidget()
        form_single = QFormLayout(page_single)
        form_single.setContentsMargins(0, 5, 0, 0)
        self.edit_s_mat = QLineEdit()
        btn_s_mat = QPushButton("浏览")
        btn_s_mat.clicked.connect(lambda: self._select_file(self.edit_s_mat, "MAT Files (*.mat *.MAT)"))
        h_s_mat = QHBoxLayout()
        h_s_mat.addWidget(self.edit_s_mat)
        h_s_mat.addWidget(btn_s_mat)
        form_single.addRow("DIC 矩阵文件:", h_s_mat)

        self.edit_s_mts = QLineEdit()
        self.edit_s_mts.setPlaceholderText("(无力学数据可留空)")
        btn_s_mts = QPushButton("浏览")
        btn_s_mts.clicked.connect(lambda: self._select_file(self.edit_s_mts, "CSV Files (*.csv *.CSV)"))
        h_s_mts = QHBoxLayout()
        h_s_mts.addWidget(self.edit_s_mts)
        h_s_mts.addWidget(btn_s_mts)
        form_single.addRow("MTS 时域文件:", h_s_mts)
        self.stack_data.addWidget(page_single)

        page_batch = QWidget()
        form_batch = QFormLayout(page_batch)
        form_batch.setContentsMargins(0, 5, 0, 0)
        self.edit_dic_dir = QLineEdit()
        btn_dic_dir = QPushButton("浏览")
        btn_dic_dir.clicked.connect(lambda: self._select_dir(self.edit_dic_dir))
        h_dic_dir = QHBoxLayout()
        h_dic_dir.addWidget(self.edit_dic_dir)
        h_dic_dir.addWidget(btn_dic_dir)
        form_batch.addRow("DIC 工作目录:", h_dic_dir)

        self.btn_pair = QPushButton("打开智能挂载台")
        self.btn_pair.clicked.connect(self._open_pairing_dialog)
        self.lbl_pair_status = QLabel("状态: 未挂载")
        h_pair = QHBoxLayout()
        h_pair.addWidget(self.btn_pair)
        h_pair.addWidget(self.lbl_pair_status)
        form_batch.addRow("批处理队列:", h_pair)
        self.stack_data.addWidget(page_batch)

        main_vbox.addWidget(self.stack_data)
        self.radio_single.toggled.connect(lambda: self.stack_data.setCurrentIndex(0))
        self.radio_batch.toggled.connect(lambda: self.stack_data.setCurrentIndex(1))

        form_out = QFormLayout()
        form_out.setContentsMargins(0, 10, 0, 0)
        self.edit_out = QLineEdit()
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(lambda: self._select_dir(self.edit_out))
        h_out = QHBoxLayout()
        h_out.addWidget(self.edit_out)
        h_out.addWidget(btn_out)
        form_out.addRow("输出落地目录:", h_out)

        main_vbox.addLayout(form_out)
        grp.setLayout(main_vbox)
        layout.addWidget(grp)

    def _init_param_group(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("物理算子与防线参数")
        form = QFormLayout()

        self.spin_gauge_len = QDoubleSpinBox()
        self.spin_gauge_len.setRange(1.0, 1000.0)
        self.spin_gauge_len.setDecimals(1)
        self.spin_gauge_len.setSingleStep(5.0)
        self.spin_gauge_len.setValue(float(self.config.get('experiment', {}).get('gauge_length_mm', 80.0)))
        form.addRow("宏观拉伸标距 (mm):", self.spin_gauge_len)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setDecimals(5)
        self.spin_scale.setSingleStep(0.0001)
        self.spin_scale.setValue(float(self.config.get('experiment', {}).get('mm_per_pixel', 0.02753)))
        form.addRow("缺省兜底比例尺 (mm/px):", self.spin_scale)

        self.spin_cod_min = QDoubleSpinBox()
        self.spin_cod_min.setDecimals(4)
        self.spin_cod_min.setSingleStep(0.001)
        self.spin_cod_min.setValue(float(self.config.get('physics', {}).get('cod_min_mm', 0.005)))
        form.addRow("底噪拦截防线 (mm):", self.spin_cod_min)

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setSingleStep(0.1)
        self.spin_k.setValue(float(self.config.get('physics', {}).get('strain_threshold_k', 2.0)))
        form.addRow("MAD 稳健阈值 (k):", self.spin_k)

        self.edit_target_strains = QLineEdit("0.2, 2.0, 4.0, 6.0")
        form.addRow("多梯度切片目标 (%):", self.edit_target_strains)

        grp.setLayout(form)
        layout.addWidget(grp)

    def _select_dir(self, line_edit: QLineEdit) -> None:
        dialog = QFileDialog(self, "选择目录")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            paths = dialog.selectedFiles()
            if paths: line_edit.setText(str(Path(paths[0])))

    def _select_file(self, line_edit: QLineEdit, file_filter: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path: line_edit.setText(file_path)

    def _open_pairing_dialog(self) -> None:
        dic_dir = self.edit_dic_dir.text()
        if not dic_dir or not Path(dic_dir).exists():
            QMessageBox.warning(self, "中断", "请先挂载有效的 DIC 工作目录。")
            return

        dialog = DataPairingDialog(dic_dir, self)
        if dialog.exec():
            self.paired_dict = dialog.paired_data
            self.lbl_pair_status.setText(f"已勾选: {len(self.paired_dict)} 组 (其中含 MTS: {dialog.mts_count} 组)")

    def _update_progress(self, current: int, total: int) -> None:
        if total > 0: self.progress.setValue(int(current / total * 100))

    def _start_pipeline(self) -> None:
        out_dir_str = self.edit_out.text().strip()
        if not out_dir_str:
            QMessageBox.warning(self, "中断", "输出目录不能为空。")
            return

        # 🌟 发射前体检：拦截致命物理参数
        gauge_len = self.spin_gauge_len.value()
        scale_val = self.spin_scale.value()
        cod_min = self.spin_cod_min.value()

        if gauge_len <= 0.1:
            QMessageBox.critical(self, "物理参数违规",
                                 "宏观标距 (Gauge Length) 必须大于 0.1 mm！\n否则会导致应变计算出现除零错误或无限大。")
            return
        if scale_val <= 0.00001:
            QMessageBox.critical(self, "物理参数违规", "比例尺 (Scale) 设置过低，请检查小数点精度！")
            return
        if cod_min == 0.0:
            reply = QMessageBox.question(
                self, "高危操作确认",
                "您将底噪拦截防线设为了 0 mm。\n这将导致大量 DIC 数学插值伪影（如 0.01μm 的噪点）混入统计，严重污染论文的数据图表。\n\n是否坚持使用 0 mm？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.spin_cod_min.setValue(0.005)
                return

        try:
            target_strains = [float(s.strip()) for s in self.edit_target_strains.text().split(',') if s.strip()]
            self.config.setdefault('export', {})['target_strains'] = target_strains
            self.config.setdefault('experiment', {})['gauge_length_mm'] = gauge_len
            self.config['experiment']['mm_per_pixel'] = scale_val
            self.config.setdefault('physics', {})['strain_threshold_k'] = self.spin_k.value()
            self.config['physics']['cod_min_mm'] = cod_min
        except ValueError:
            QMessageBox.warning(self, "格式错误", "参数解析失败，请确保梯度输入如 '0.2, 2.0, 4.0' 的格式。")
            return

        process_dict = {}
        if self.radio_single.isChecked():
            mat_f = self.edit_s_mat.text().strip()
            if not mat_f:
                QMessageBox.warning(self, "中断", "单次模式下必须指定 MAT 文件。")
                return
            process_dict[mat_f] = self.edit_s_mts.text().strip()
        else:
            if not self.paired_dict:
                QMessageBox.warning(self, "中断", "批处理队列为空。请先进入【智能挂载台】勾选试件。")
                return
            process_dict = self.paired_dict

        Path(out_dir_str).mkdir(parents=True, exist_ok=True)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("引擎全速运转中...")
        self.progress.setValue(0)
        self.logger_console.clear()

        self.logger_console.append(f"🟢 [Pre-flight Check] 体检通过。待计算队列总数: {len(process_dict)} 组。")
        self.logger_console.append(f"🔧 宏观标距锁定: {gauge_len:.1f} mm | 底噪防线: {cod_min:.3f} mm")
        self.logger_console.append("-" * 50)

        self.worker = AnalysisPipelineWorker(process_dict, Path(out_dir_str), self.config)
        self.worker.error_occurred.connect(lambda err: self.logger_console.append(f"\n🔴 [FATAL] {err}"))
        self.worker.log_emitted.connect(self.logger_console.append)
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.specimen_processed.connect(
            lambda p1, p2: self.logger_console.append(f"🟢 [SUCCESS] 数据已安全落盘: {Path(p1).stem}")
        )
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.start()

    def _on_pipeline_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_start.setText("启动物理分析引擎")
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.edit_out.text().strip()))