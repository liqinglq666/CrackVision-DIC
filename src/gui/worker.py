import tempfile
import shutil
import multiprocessing
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Any

from src.core.io_sync import PipelineIO
from src.core.physics import CrackPhysicsEngine
from src.core.evolution_analyzer import EvolutionAnalyzer

import logging

logger = logging.getLogger(__name__)

_worker_engine = None


@dataclass(frozen=True)
class FrameTaskPayload:
    config: dict
    u_path: str
    exx_path: str
    ratio: float
    frame_id: int
    time_s: float


def analyze_single_frame_task(payload: FrameTaskPayload) -> Optional[Dict[str, Any]]:
    global _worker_engine
    try:
        if _worker_engine is None:
            _worker_engine = CrackPhysicsEngine(payload.config)

        u = np.load(payload.u_path)
        exx = np.load(payload.exx_path)
        mask = ~np.isnan(exx)

        skeleton = _worker_engine.extract_skeleton(exx, mask)
        res = _worker_engine.compute_cod(u, skeleton, payload.ratio)

        L0 = float(payload.config.get('experiment', {}).get('gauge_length_mm', 80.0))

        # 🌟 OOM 终极修复：基于一维列投影的边界探测，杜绝千万级矩阵扩张
        valid_u_mask = ~np.isnan(u) & mask
        valid_cols_bool = np.any(valid_u_mask, axis=0)
        active_cols = np.where(valid_cols_bool)[0]

        strain = 0.0
        if active_cols.size > 50:
            # 仅截取最左和最右各 10 列进行切片，内存开销近乎为 0
            left_cols = active_cols[:10]
            right_cols = active_cols[-10:]

            with np.errstate(all='ignore'):
                left_u = np.nanmedian(u[:, left_cols])
                right_u = np.nanmedian(u[:, right_cols])

            if not np.isnan(left_u) and not np.isnan(right_u):
                du = abs(right_u - left_u) * payload.ratio
                strain = du / L0

        res.update({
            'Frame': int(payload.frame_id),
            'Time_s': float(payload.time_s),
            'global_strain': max(0.0, float(strain))
        })

        crack_count = res.get('crack_count', 0)
        res['crack_spacing_mm'] = float(L0 / crack_count) if crack_count > 0 else 0.0

        return res
    except Exception as e:
        logger.error(f"子任务崩溃 (Frame {payload.frame_id}): {e}", exc_info=True)
        return None


class AnalysisPipelineWorker(QThread):
    progress_updated = Signal(int, int)
    log_emitted = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()
    specimen_processed = Signal(str, str)

    def __init__(self, paired_data: dict, out_dir: Path, config: dict) -> None:
        super().__init__()
        self.paired_data = paired_data
        self.out_dir = out_dir
        self.config = config
        self._is_running = True

    def run(self) -> None:
        try:
            fallback_ratio = float(self.config['experiment']['mm_per_pixel'])
            interval = float(self.config['experiment']['sampling_interval_s'])
            total = len(self.paired_data)

            for i, (mat_f, mts_f) in enumerate(self.paired_data.items()):
                if not self._is_running: break
                try:
                    self._process_specimen(Path(mat_f), Path(mts_f) if mts_f else None, fallback_ratio, interval)
                except Exception as e:
                    self.log_emitted.emit(f"❌ 试件 {Path(mat_f).name} 发生致命异常已跳过: {e}")
                    logger.error(f"Specimen process failed: {mat_f}", exc_info=True)
                self.progress_updated.emit(i + 1, total)

        except Exception as e:
            self.error_occurred.emit(f"多进程调度崩溃: {e}")
        finally:
            self.finished.emit()

    def _process_specimen(self, mat_path: Path, mts_path: Optional[Path], ratio: float, interval: float) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="cv_"))

        try:
            with h5py.File(mat_path, 'r') as f:
                ratio = float(f['data_dic_save']['dispinfo']['pixtounits'][0, 0])
        except Exception:
            pass

        self.log_emitted.emit(f"🚀 解析试件: {mat_path.name} | Scale: {ratio:.5f} mm/px")

        tasks = []
        for frame in PipelineIO.stream_dic_frames(mat_path, ratio):
            u_p, exx_p = temp_dir / f"u_{frame.frame_id}.npy", temp_dir / f"exx_{frame.frame_id}.npy"
            np.save(u_p, frame.u_map)
            np.save(exx_p, frame.exx_map)
            tasks.append(
                FrameTaskPayload(self.config, str(u_p), str(exx_p), ratio, frame.frame_id, frame.frame_id * interval))

        results = []
        cur_max_strain = 0.0
        max_workers = min(10, max(1, multiprocessing.cpu_count() - 2))

        with ProcessPoolExecutor(max_workers=max_workers) as exec:
            for res in exec.map(analyze_single_frame_task, tasks):
                if res:
                    # 强制物理单调递增
                    cur_max_strain = max(cur_max_strain, res.get('global_strain', 0.0))
                    res['global_strain'] = cur_max_strain
                    results.append(res)

        shutil.rmtree(temp_dir, ignore_errors=True)

        if not results:
            self.log_emitted.emit(f"⚠️ {mat_path.name} 提取失败：所有帧被物理底噪拦截或无有效数据。")
            return

        results.sort(key=lambda x: x['Frame'])
        df = pd.DataFrame(results)

        if mts_path and mts_path.exists():
            try:
                df = EvolutionAnalyzer(self.config, mts_path).synchronize(df)
                self.log_emitted.emit(f"✅ {mat_path.name} 时域同步成功")
            except Exception as e:
                self.log_emitted.emit(f"⚠️ 同步失败(退回纯DIC): {e}")

        # ---------------- 数据清洗与硬编码格式防断裂 ----------------
        df['Strain_pct'] = df.get('global_strain', 0.0) * 100.0
        df['W_avg_um'] = df.get('w_avg', 0.0) * 1000.0
        df['W_max_um'] = df.get('w_max', 0.0) * 1000.0
        df['W_99_um'] = df.get('w_99', 0.0) * 1000.0
        df['crack_spacing_mm'] = df.get('crack_spacing_mm', 0.0)

        for col in ['Strain_pct', 'crack_count', 'crack_spacing_mm', 'W_avg_um', 'W_99_um', 'W_max_um']:
            if col not in df.columns: df[col] = 0.0

        sat_idx = df['crack_count'].idxmax()
        ult_idx = df['Stress_MPa'].idxmax() if 'Stress_MPa' in df.columns and not df['Stress_MPa'].isna().all() else df[
            'Strain_pct'].idxmax()

        ult_s_raw = df.loc[ult_idx, 'global_strain']
        df['Normalized_Strain'] = df['global_strain'] / ult_s_raw if ult_s_raw > 0 else 0.0

        sat_row = df.loc[sat_idx]
        ult_row = df.loc[ult_idx]

        # ---------------- 导出 1：Origin 画图专供 ----------------
        out_f = self.out_dir / f"{mat_path.stem}_Origin_Plot_Data.xlsx"
        try:
            with pd.ExcelWriter(out_f, engine='openpyxl') as writer:
                df[['Strain_pct', 'crack_count', 'crack_spacing_mm', 'W_avg_um', 'W_99_um', 'W_max_um']].to_excel(
                    writer, sheet_name='Fig1_Dynamics', index=False)
                df[['Normalized_Strain', 'crack_count', 'W_avg_um', 'W_max_um']].to_excel(writer,
                                                                                          sheet_name='Fig2_Normalized',
                                                                                          index=False)

                p_sat = next(
                    (r.get('per_crack_details', pd.DataFrame()) for r in results if r['Frame'] == sat_row['Frame']),
                    pd.DataFrame())
                p_ult = next(
                    (r.get('per_crack_details', pd.DataFrame()) for r in results if r['Frame'] == ult_row['Frame']),
                    pd.DataFrame())

                dist_dict = {}
                dist_dict['Saturated_um'] = pd.Series(
                    p_sat['W_avg_mm'].values * 1000.0) if not p_sat.empty else pd.Series([], dtype=float)
                dist_dict['Ultimate_um'] = pd.Series(
                    p_ult['W_avg_mm'].values * 1000.0) if not p_ult.empty else pd.Series([], dtype=float)
                pd.DataFrame(dist_dict).to_excel(writer, sheet_name='Fig3_Distribution', index=False)

                target_strains = self.config.get('export', {}).get('target_strains', [0.2, 2.0, 4.0, 6.0])
                grad_data = {}
                for ts in target_strains:
                    if ts <= df['Strain_pct'].max():
                        idx = (df['Strain_pct'] - ts).abs().idxmin()
                        p_d = next((r.get('per_crack_details', pd.DataFrame()) for r in results if
                                    r['Frame'] == df.loc[idx, 'Frame']), pd.DataFrame())
                        if not p_d.empty: grad_data[f'Strain_{df.loc[idx, "Strain_pct"]:.2f}%_um'] = p_d[
                                                                                                         'W_avg_mm'] * 1000.0

                if not grad_data: grad_data = {'Notice': ['No cracks reached target strains']}
                pd.DataFrame({k: pd.Series(v) for k, v in grad_data.items()}).to_excel(writer,
                                                                                       sheet_name='Fig4_Gradient',
                                                                                       index=False)
        except Exception as e:
            self.log_emitted.emit(f"⚠️ Origin 数据导出失败: {e}")

        # ---------------- 导出 2：SCI 全量统计报告 ----------------
        stat_f = self.out_dir / f"{mat_path.stem}_Statistics_Report.xlsx"
        try:
            with pd.ExcelWriter(stat_f, engine='openpyxl') as writer:
                has_stress = 'Stress_MPa' in df.columns and not df['Stress_MPa'].isna().all()
                summary_dict = {
                    'Specimen': [mat_path.stem],
                    'UTS_Stress_MPa': [float(ult_row.get('Stress_MPa', np.nan)) if has_stress else np.nan],
                    'Ultimate_Strain_pct': [float(ult_row['Strain_pct'])],
                    'Saturated_Crack_Count': [int(sat_row['crack_count'])],
                    'Saturated_Spacing_mm': [float(sat_row['crack_spacing_mm'])],
                    'Saturated_W_avg_um': [float(sat_row['W_avg_um'])],
                    'Ultimate_W_99_um': [float(ult_row['W_99_um'])],
                    'Ultimate_W_max_um': [float(ult_row['W_max_um'])]
                }
                pd.DataFrame(summary_dict).to_excel(writer, sheet_name='01_Macro_Summary', index=False)

                grad_rows = []
                check_strains = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                for ts in check_strains:
                    if ts <= df['Strain_pct'].max():
                        idx = (df['Strain_pct'] - ts).abs().idxmin()
                        grad_rows.append({
                            'Target_Strain_pct': ts,
                            'Real_Strain_pct': float(df.loc[idx, 'Strain_pct']),
                            'Crack_Count': int(df.loc[idx, 'crack_count']),
                            'Spacing_mm': float(df.loc[idx, 'crack_spacing_mm']),
                            'W_avg_um': float(df.loc[idx, 'W_avg_um']),
                            'W_99_um': float(df.loc[idx, 'W_99_um'])
                        })

                if not grad_rows: grad_rows = [{'Notice': 'No data reached target strains'}]
                pd.DataFrame(grad_rows).to_excel(writer, sheet_name='02_Gradient_States', index=False)

                def _export_crack_details(p_df: pd.DataFrame, sheet_name: str) -> None:
                    if not p_df.empty:
                        out_df = p_df.copy()
                        out_df['W_avg_um'] = out_df['W_avg_mm'] * 1000.0
                        out_df['W_max_um'] = out_df['W_max_mm'] * 1000.0
                        out_df = out_df[['Crack_ID', 'Length_mm', 'W_avg_um', 'W_max_um']].sort_values('W_avg_um',
                                                                                                       ascending=False)
                        out_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        pd.DataFrame({'Notice': ['No Cracks Detected']}).to_excel(writer, sheet_name=sheet_name,
                                                                                  index=False)

                _export_crack_details(p_sat, '03_Saturated_Cracks')
                _export_crack_details(p_ult, '04_Ultimate_Cracks')

            self.log_emitted.emit(f"📊 统计报告已生成: {stat_f.name}")
        except Exception as e:
            self.log_emitted.emit(f"⚠️ 统计报告导出失败: {e}")

        for r in results:
            r.pop('per_crack_details', None)
            r.pop('raw_widths', None)

        self.specimen_processed.emit(str(out_f), str(stat_f))

    def stop(self) -> None:
        self._is_running = False