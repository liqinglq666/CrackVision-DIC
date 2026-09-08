from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def build_demo_pair(root: Path) -> tuple[Path, Path]:
    demo_dir = root / "demo_input"
    demo_dir.mkdir(parents=True, exist_ok=True)
    h5_path = demo_dir / "ECC_DIC_Demo.h5"
    mts_path = demo_dir / "ECC_MTS_Demo.csv"

    n_frames, h, w = 5, 101, 151
    u = np.zeros((n_frames, h, w), dtype=np.float64)
    v = np.zeros_like(u)
    exx = np.full_like(u, 1.0e-5)
    eyy = np.zeros_like(u)
    exy = np.zeros_like(u)
    mask = np.ones((n_frames, h, w), dtype=np.uint8)

    crack_x = (35, 75, 115)
    jumps_mm = (0.040, 0.060, 0.080)
    peak_frame = 2

    # Create three vertical fine cracks. Principal strain locates the crack
    # geometry; cumulative horizontal displacement jumps provide the COD.
    for x in crack_x:
        exx[peak_frame, 14:87, x] = 0.020

    cumulative = np.zeros((h, w), dtype=np.float64)
    running = 0.0
    last = 0
    for x, jump in zip(crack_x, jumps_mm):
        cumulative[:, last:x] = running
        running += jump
        last = x
    cumulative[:, last:] = running
    u[peak_frame] = cumulative

    # Non-peak frames carry smaller deformations to keep the demo physically
    # plausible while preserving frame-2 as the MTS-selected analysis target.
    u[1] = cumulative * 0.25
    u[3] = cumulative * 1.10
    u[4] = cumulative * 1.20

    with h5py.File(h5_path, "w") as f:
        f.attrs["format"] = "CrackVision-Ncorr"
        f.attrs["format_version"] = 1
        f.attrs["pixel_size_mm"] = 0.030
        f.attrs["dic_step_px"] = 3.0
        f.attrs["ncorr_spacing_raw"] = 2.0
        f.attrs["dic_point_spacing_mm"] = 0.090
        f.attrs["sampling_interval_s"] = 5.0
        f.attrs["start_time_s"] = 0.0
        f.attrs["displacement_units"] = "mm"
        f.attrs["coordinate_system"] = "reference"
        f.attrs["strain_measure"] = "Green-Lagrange"
        f.attrs["numeric_precision"] = "float64"
        fields = f.create_group("fields")
        fields.create_dataset("u", data=u)
        fields.create_dataset("v", data=v)
        fields.create_dataset("exx", data=exx)
        fields.create_dataset("eyy", data=eyy)
        fields.create_dataset("exy", data=exy)
        fields.create_dataset("mask", data=mask)
        f.create_dataset("time_s", data=np.arange(n_frames, dtype=float) * 5.0)

    times = np.arange(0.0, 20.01, 0.5)
    forces = np.where(
        times <= 10.5,
        100.0 + 400.0 * times,
        4300.0 - 210.0 * (times - 10.5),
    )
    with mts_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Time,Force\n")
        f.write("s,N\n")
        for t, force in zip(times, forces):
            f.write(f"{t:.2f},{force:.3f}\n")

    return h5_path, mts_path


def grab(window: MainWindow, path: Path, app: QApplication) -> None:
    window.show()
    app.processEvents()
    time.sleep(0.25)
    app.processEvents()
    pixmap = window.grab()
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/softreg"))
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path, mts_path = build_demo_pair(out_dir)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    grab(window, out_dir / "01_软件启动界面.png", app)

    window.data_file = data_path
    window.mts_file = mts_path
    window.data_edit.setText(data_path.name)
    window.mts_edit.setText(mts_path.name)
    window._refresh_ready_state()
    grab(window, out_dir / "02_数据载入就绪.png", app)

    # Exercise the real GUI workflow: MainWindow -> AnalysisWorker -> pipeline ->
    # CrackPhysicsEngine -> Excel exporter -> GUI completed signal.
    window._start()
    deadline = time.time() + 60.0
    while window.worker is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    if window.worker is not None:
        raise TimeoutError("CrackVision-DIC analysis did not finish within 60 s")

    grab(window, out_dir / "03_峰值帧裂缝分析完成.png", app)

    output_path = window.output_path
    if output_path is None or not output_path.exists():
        raise RuntimeError("GUI run finished without an exported workbook")

    # Store machine-readable evidence next to the screenshots.
    evidence = {
        "software": "CrackVision-DIC",
        "run_mode": "GitHub Actions real GUI + core analysis run",
        "input_h5": str(data_path.name),
        "input_mts": str(mts_path.name),
        "output_workbook": str(output_path.name),
        "status_text": window.status_label.text(),
        "result_text": window.result_label.text(),
    }
    (out_dir / "run_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy the generated workbook into the artifact root for registration QA.
    workbook_copy = out_dir / output_path.name
    if output_path.resolve() != workbook_copy.resolve():
        workbook_copy.write_bytes(output_path.read_bytes())

    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
