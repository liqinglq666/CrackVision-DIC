import sys
import logging
import traceback
import multiprocessing
from pathlib import Path
from typing import Type
from types import TracebackType

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from src.gui.main_window import MainWindow


def setup_global_logging() -> None:
    """
    初始化全局日志系统。
    使用绝对路径与强覆盖策略，确保打包部署后日志必定精准落盘。
    """
    # 🌟 修复 1：锚定绝对路径。防止跨目录终端调用或快捷方式启动导致日志乱飞
    root_dir = Path(__file__).resolve().parent
    log_dir = root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app_runtime.log"

    # 🌟 修复 2：追加 force=True (Python 3.8+)。
    # 强制覆盖任何第三方依赖在 import 阶段可能抢占生成的劣质 Root Logger。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode="a"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("=" * 60)
    logging.info("🚀 CrackVision-DIC 极简算力引擎启动")
    logging.info("=" * 60)


def global_exception_handler(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType
) -> None:
    """
    全局未捕获异常拦截器。
    接管系统异常，防止 GUI 闪退，确保留下完整的崩溃堆栈以供复盘。
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical(f"🔥 发生未捕获的全局致命异常:\n{error_msg}")


def main() -> None:
    setup_global_logging()
    sys.excepthook = global_exception_handler

    # Qt 高分屏自适应渲染策略 (防止 4K 屏幕下界面模糊)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    # 🌟 修复 3：补全系统级应用元数据。让操作系统的任务管理器能正确识别进程名称。
    app.setApplicationName("CrackVision-DIC")
    app.setApplicationVersion("2.0.0-Core")
    app.setOrganizationName("ScientificMechanics")

    app.setStyle("Fusion")

    try:
        window = MainWindow()
        window.show()

        exit_code = app.exec()
        logging.info(f"🛑 引擎主事件循环安全结束，退出码: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logging.critical(f"主窗口初始化或事件循环运行期间发生灾难性崩溃: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 防御性编程：在 Windows 环境下使用 ProcessPoolExecutor 必须挂载此方法。
    # 否则 PyInstaller 打包后，子进程会误以为自己是主进程，引发无限弹出新窗口的 OOM 死机。
    multiprocessing.freeze_support()
    main()