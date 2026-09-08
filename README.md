# CrackVision-DIC 3.1 · Ncorr 专用裂缝宽度后处理

CrackVision-DIC 现在只服务一个核心任务：

> **把 Ncorr 的二维 DIC 位移/应变结果转换成可审计的 ECC / SHCC 裂缝宽度（COD）结果。**

不再把完整 Ncorr 工程、相机裂缝分割、MTS 同步和多套统计报表塞进同一条链路。

## 推荐工作流

### 新试件：不要先保存巨大 Ncorr MAT

以后建议用明确的句柄启动 Ncorr：

```matlab
handles_ncorr = ncorr;
```

按正常流程完成：

```text
Load Reference Image
Load Current Image(s)
Set ROI
Set DIC Parameters
Perform DIC Analysis
Format Displacements
Get Unit Conversion（使用 mm 标尺）
Calculate Strains
```

然后在 MATLAB 命令行：

```matlab
addpath('CrackVision-DIC/matlab')

export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5);
```

最后的 `5` 表示每张图相隔 5 s。

如果 Ncorr 是直接以 `ans` 打开的，也可以临时：

```matlab
export_ncorr_to_crackvision(ans, 'Specimen01_CrackVision.h5', 5);
```

但更推荐 `handles_ncorr = ncorr;`，避免 `ans` 被后续 MATLAB 命令覆盖。

### 已经有巨大 MAT 的旧试件

只需要转换一次：

```matlab
addpath('CrackVision-DIC/matlab')

export_ncorr_to_crackvision( ...
    'Specimen01_Ncorr.mat', ...
    'Specimen01_CrackVision.h5', ...
    5);
```

MATLAB 会读取 `data_dic_save`，只导出 CrackVision 真正需要的数据。

## CrackVision-Ncorr H5 保留什么

```text
/fields/u
/fields/v
/fields/exx
/fields/eyy
/fields/exy
/fields/mask
/time_s

metadata:
pixel_size_mm
ncorr_spacing_raw
dic_step_px
dic_point_spacing_mm
coordinate_system = reference
strain_measure = Green-Lagrange
numeric_precision
```

桥接脚本只导出 Ncorr 的 reference-formatted 数值场：

```text
plot_u_ref_formatted
plot_v_ref_formatted

plot_exx_ref_formatted
plot_eyy_ref_formatted
plot_exy_ref_formatted
```

这样 U/V、裂缝位置和应变张量处于同一个参考坐标系里。Ncorr 的 current-formatted Eulerian 场仍可用于可视化，但 CrackVision 主 COD 链路不把 reference 和 current 坐标系混用。

## 为什么这个 H5 比完整 Ncorr MAT 小

完整 Ncorr 保存还可能包含：

- reference/current image 信息
- ROI 对象
- correlation coefficient
- 多种 formatted plots
- GUI/分析状态
- 其他中间结果

CrackVision-Ncorr H5 只保留 5 个核心数值场、mask、时间和尺度。

默认使用：

```text
single precision (float32)
+ HDF5 chunking
+ gzip / Deflate compression
```

如果必须保留 double：

```matlab
export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5, ...
    'double');
```

轻量 H5 仍可能较大，因为全过程二维场本身就有大量数据；目标是**删掉与裂缝宽度无关的 Ncorr 数据**，不是把几百帧 DIC 场压成几 MB。

## CrackVision 计算链

```text
Ncorr
  ↓
U / V
Exx / Eyy / Exy
  ↓
maximum principal tensile strain
  ↓
crack candidate
  ↓
skeleton
  ↓
local tangent / normal
  ↓
sample U/V on both crack faces
  ↓
robust line fit on each side
  ↓
extrapolate to crack plane
  ↓
COD = normal displacement discontinuity
  ↓
Excel + QA
```

最大主拉应变：

```text
ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
```

Ncorr reference-formatted `Exy` 按 Green-Lagrange 张量剪切分量处理。

## 裂缝宽度不是应变带宽度

CrackVision 不把彩色 Exx/主应变带的“粗细”当裂缝宽度。

主结果来自裂缝两侧位移跳量：

```text
COD = |Δu · n|
COD = |ΔU * nx + ΔV * ny|
```

并通过裂缝两侧多点回归后外推到裂缝面，尽量去掉连续弹性位移梯度。

## Ncorr spacing

Ncorr 原生 `spacing` 按 skipped-pixel count 处理：

```text
DIC step px = spacing + 1
DIC grid spacing mm = (spacing + 1) × pixtounits
```

CrackVision 将两种尺度分开：

```text
U/V displacement px × pixel_size_mm
    → displacement mm

DIC grid index × dic_point_spacing_mm
    → crack geometry / physical sampling distance
```

## 输入优先级

GUI 支持：

```text
推荐：
*.h5 / *.hdf5
CrackVision-Ncorr bridge

兼容：
*.mat
original Ncorr data_dic_save
```

日常分析优先 H5。原始 MAT 只保留作旧数据兼容和追溯。

## 输出

每个试件只生成一个：

```text
<Specimen>_CrackVision.xlsx
```

包括：

```text
00_READ_ME
01_Frame_Summary
02_Crack_Details
03_QA
```

推荐论文指标：

```text
W_median_um
W_avg_um
W_95_um
W_max_um
crack_count
COD_samples
Fit_R2_median
cod_status
```

计算失败保持 `NaN`，不会再强制写成 `0 μm`。

## 运行

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

测试：

```bash
python -m pytest -q
```

## 不要把实验数据提交到 GitHub

仓库已经忽略：

```text
*.mat
*.h5
*.hdf5
```

GitHub 只保存代码。原始图片、完整 Ncorr MAT 和 CrackVision bridge H5 都应保存在实验数据目录或实验室存储中。
