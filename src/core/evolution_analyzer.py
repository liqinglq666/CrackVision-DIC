import logging
import io
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Optional, List

logger = logging.getLogger(__name__)


class EvolutionAnalyzer:
    """
    宏微观时域同步引擎。
    核心职责：确保应变计算严格基于 80mm 全局物理标距，而非局部 ROI 跨度。
    """

    def __init__(self, config: dict, mts_path: Path) -> None:
        self.config = config
        self.mts_path = Path(mts_path)
        # 🌟 强制从配置中提取物理参数
        self.area_mm2 = float(self.config.get('experiment', {}).get('cross_section_area_mm2', 100.0))
        self.gauge_length_mm = float(self.config.get('experiment', {}).get('gauge_length_mm', 80.0))

    def _decode_file(self) -> str:
        """多编码自动识别读取 MTS 原始数据。"""
        try:
            raw_bytes = self.mts_path.read_bytes()
        except OSError as e:
            logger.error(f"无法读取 MTS 数据文件: {e}")
            raise

        for enc in ['utf-8-sig', 'gbk', 'utf-8', 'gb18030', 'ansi']:
            try:
                decoded_text = raw_bytes.decode(enc)
                if any(k in decoded_text for k in ['时间', 'Time', 'sec', '横梁']):
                    return decoded_text.replace('"', '').replace("'", "")
            except UnicodeDecodeError:
                continue
        raise ValueError(f"MTS 编码解析失败，请检查文件编码格式。")

    def _locate_header(self, lines: List[str]) -> int:
        """启发式定位 CSV 表头索引。"""
        for i, line in enumerate(lines):
            line_l = line.lower()
            if any(k in line_l for k in ['时间', 'time']) and any(k in line_l for k in ['力', 'load', '横梁', 'disp']):
                return i
        raise ValueError("MTS 文件中未检测到有效力学表头字段。")

    def _smart_read_mts(self) -> pd.DataFrame:
        """清洗并标准化力学数据流。"""
        content = self._decode_file()  # ✅ 已修复 self 调用
        lines = content.splitlines()
        header_idx = self._locate_header(lines)

        data_str = "\n".join(lines[header_idx:])
        sep = '\t' if '\t' in lines[header_idx] else ','

        try:
            df = pd.read_csv(io.StringIO(data_str), sep=sep, on_bad_lines='skip')
            # 过滤单位行
            if df.iloc[0].astype(str).str.contains(r'mm|n|sec|kn', case=False).any():
                df = df.drop(0).reset_index(drop=True)

            df.columns = [str(c).strip() for c in df.columns]

            # 模糊匹配核心物理列
            time_col = next((c for c in df.columns if any(k in c.lower() for k in ['时间', 'time', 'sec'])), None)
            force_col = next((c for c in df.columns if any(k in c.lower() for k in ['力', 'load', 'force'])), None)
            disp_col = next((c for c in df.columns if any(k in c.lower() for k in ['横梁', 'disp', '引伸计'])), None)

            clean_df = pd.DataFrame()
            clean_df['Time_s'] = pd.to_numeric(df[time_col], errors='coerce')
            clean_df['Force_N'] = pd.to_numeric(df[force_col], errors='coerce')
            clean_df['Disp_mm'] = pd.to_numeric(df[disp_col], errors='coerce')

            clean_df = clean_df.dropna(subset=['Time_s', 'Force_N']).replace([np.inf, -np.inf], np.nan)

            # N -> MPa 换算
            f_vals = np.abs(clean_df['Force_N'].values)
            if np.nanmax(f_vals) < 100: f_vals *= 1000.0  # 自动纠正 kN 单位
            clean_df['Stress_MPa'] = f_vals / self.area_mm2

            return clean_df
        except Exception as e:
            raise ValueError(f"MTS 数据解析崩溃: {e}")

    def synchronize(self, df_dic: pd.DataFrame) -> pd.DataFrame:
        """执行 1D 线性插值时间轴对齐。"""
        df_mts = self._smart_read_mts()
        dic_times = df_dic['Time_s'].values
        mts_times = df_mts['Time_s'].values

        _, unique_idx = np.unique(mts_times, return_index=True)
        mts_t_u, df_mts_u = mts_times[unique_idx], df_mts.iloc[unique_idx]

        df_sync = df_dic.copy()
        for col in ['Stress_MPa', 'Disp_mm', 'Force_N']:
            if col in df_mts_u.columns:
                interp = interp1d(mts_t_u, df_mts_u[col].values, bounds_error=False, fill_value="extrapolate")
                df_sync[col] = interp(dic_times)

        if 'Disp_mm' in df_sync.columns:
            # 🌟 物理核心：强制基于全局标距换算应变
            df_sync['global_strain'] = np.abs(df_sync['Disp_mm']) / self.gauge_length_mm

        return df_sync