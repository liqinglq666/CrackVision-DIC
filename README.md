# CrackVision-DIC

<p align="center">
  <strong>面向 ECC / SHCC 单轴拉伸试验的峰值拉应力状态裂缝宽度分析工具</strong><br/>
  <sub>Peak-stress-state Crack Opening Displacement Analysis based on Ncorr DIC + MTS</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white" alt="PySide6" />
  <img src="https://img.shields.io/badge/Data-HDF5-005B96" alt="HDF5" />
  <img src="https://img.shields.io/badge/DIC-Ncorr-6B7280" alt="Ncorr" />
</p>

---

## 项目简介 · Overview

**CrackVision-DIC** 是一个专门用于 **ECC / SHCC 单轴拉伸 Digital Image Correlation (DIC)** 后处理的科研工具。

项目只解决一个明确问题：

> **从 MTS 原始拉伸数据中自动定位 Peak Tensile Stress 对应时刻，匹配最近的 Ncorr DIC frame，并基于该帧位移场与应变场提取全部有效裂缝的 Crack Opening Displacement (COD)。**

最终输出：

- 峰值拉应力对应的 DIC frame；
- 峰值状态下全部有效裂缝的代表宽度；
- specimen-level 平均 / 中位 / P95 / 最大裂缝宽度；
- 时间匹配、尺度、阈值、有效区等 QA 信息；
- 可直接用于论文整理的 Excel workbook。

当前实验条件假定：

```text
MTS t = 0 s  ⇔  DIC frame 0 = 0 s
```

因此 GUI 中不再提供人为 synchronization offset。

---

## 整体流程 · Scientific Workflow

```mermaid
flowchart TD
    A[MTS 原始 CSV] --> B[解析 Force 与 Time]
    B --> C[定位 Peak Tensile Force Time]

    D[Ncorr H5 / MAT] --> E[DIC 时间轴]
    C --> F[匹配最近 DIC Frame]
    E --> F

    F --> G[峰值拉应力对应 Frame]
    G --> H[读取 U / V / Exx / Eyy / Exy]

    H --> I[Maximum Principal Tensile Strain]
    I --> J[Adaptive Crack Candidate]
    J --> K[Skeletonization + Crack Labeling]
    K --> L[Local Crack Normal]

    H --> M[裂缝两侧位移采样]
    L --> M
    M --> N[Robust Bilateral Regression]
    N --> O[Extrapolate to Crack Plane]
    O --> P[Local COD]

    P --> Q[每条裂缝取 Median COD]
    Q --> R[Equal-weight Statistics]
    R --> S[Paper-ready Excel]
```

核心计算链：

```text
MTS Peak State
    ↓
DIC Peak Frame
    ↓
Maximum Principal Tensile Strain
    ↓
Crack Geometry
    ↓
Displacement Jump along Local Normal
    ↓
COD
    ↓
Crack-wise Representative Width
    ↓
Equal-weight Specimen Statistics
```

---

# 1. 峰值拉应力帧选择 · Peak-stress Frame Selection

## 1.1 为什么只需要峰值拉力时刻

对名义截面积 `A` 固定的拉伸试件：

```text
σ(t) = F(t) / A
```

因为 `A` 为常数，所以：

```text
arg max_t σ(t) = arg max_t F(t)
```

也就是说，**Peak Tensile Stress 与 Peak Tensile Force 出现在同一时刻**。

因此 CrackVision-DIC 在选择目标 DIC frame 时，只需要 MTS 中的：

```text
Force
Time
```

不需要额外输入试件截面积。

对于不同 MTS 导出格式，拉伸方向可能记为正值或负值。程序先判断 tension sign：

```text
s ∈ {-1, +1}
```

再确定峰值时刻：

```text
t_peak = arg max_t [s · F(t)]
```

MTS 在本项目中只负责**确定 Peak Tensile Stress 对应时刻**，不参与 COD 计算。

## 1.2 峰值时刻与 DIC frame 匹配

设 DIC 时间轴为：

```text
t_DIC(0), t_DIC(1), ..., t_DIC(N-1)
```

程序选择距离 MTS 峰值时刻最近的一帧：

```text
k* = arg min_k |t_DIC(k) - t_peak|
```

时间匹配误差：

```text
Δt = t_DIC(k*) - t_peak
```

对于当前 **5 s / image** 的采集方式，只要 MTS 与相机同步起始，nearest-frame matching 的绝对误差通常不超过约半个采样间隔。

```mermaid
sequenceDiagram
    participant MTS as MTS CSV
    participant CV as CrackVision
    participant DIC as Ncorr Data

    MTS->>CV: Force(t), Time
    CV->>CV: Find t_peak
    DIC->>CV: DIC frame times
    CV->>CV: Find nearest frame k*
    CV->>DIC: Read selected frame
    DIC-->>CV: U, V, Exx, Eyy, Exy, mask
```

---

# 2. Ncorr 数据接口 · Data Contract

对于新试验，推荐使用 CrackVision 专用 **compact HDF5 bridge**，避免 Python 直接解析体积很大的原始 Ncorr `.mat`。

## 2.1 推荐导出方式

完成 Ncorr：

```text
DIC Analysis
→ Format Displacements
→ Get Unit Conversion (mm)
→ Calculate Strains
```

然后在 MATLAB 中运行：

```matlab
handles_ncorr = ncorr;
% ... 完成正常 Ncorr 计算 ...

addpath('path/to/CrackVision-DIC/matlab')
export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5);   % 5 s / image
```

已有 Ncorr MAT 也可以转换：

```matlab
export_ncorr_to_crackvision( ...
    'Specimen01.mat', ...
    'Specimen01_CrackVision.h5', ...
    5);
```

## 2.2 HDF5 保存内容

```text
/
├─ fields/
│  ├─ u
│  ├─ v
│  ├─ exx
│  ├─ eyy
│  ├─ exy
│  └─ mask
├─ time_s
└─ attributes
   ├─ coordinate_system = reference
   ├─ strain_measure = Green-Lagrange
   ├─ pixel_size_mm
   ├─ ncorr_spacing_raw
   ├─ dic_step_px
   ├─ dic_point_spacing_mm
   ├─ sampling_interval_s
   └─ numeric_precision
```

位移与应变统一采用 **Reference Configuration**：

```text
plot_u_ref_formatted
plot_v_ref_formatted
plot_exx_ref_formatted
plot_eyy_ref_formatted
plot_exy_ref_formatted
```

这样可以避免不同 configuration 的 displacement / strain field 混用。

## 2.3 Ncorr spacing 与物理网格间距

Ncorr 原生 `spacing` 按 skipped-pixel count 处理，因此：

```text
DIC centre step (px) = Ncorr spacing + 1
```

记作：

```text
Δp_DIC = spacing_Ncorr + 1
```

若图像尺度为：

```text
p = pixel_size_mm   [mm/pixel]
```

则 DIC 网格物理间距：

```text
Δx_DIC = p · Δp_DIC   [mm/point]
```

这里必须区分：

- `pixel_size_mm`：displacement pixel → mm；
- `dic_point_spacing_mm`：DIC grid index → physical distance。

两者相关，但不是同一个量。

---

# 3. 裂缝定位 · Maximum Principal Tensile Strain

裂缝定位不直接依赖单一 `Exx` 或 `Eyy`，而使用 **Maximum Principal Tensile Strain**。

二维 Green-Lagrange strain tensor 对应的最大主拉应变为：

```text
ε1 = (Exx + Eyy) / 2
     + sqrt( ((Exx - Eyy) / 2)^2 + Exy^2 )
```

当前 Ncorr bridge 中，`Exy` 按 **tensor Green-Lagrange shear component** 处理，而不是 engineering shear strain。

采用 `ε1` 的原因是：ECC / SHCC 裂缝可能水平、竖直或倾斜；Maximum Principal Tensile Strain 对全局坐标方向更不敏感，更适合 orientation-independent crack localization。

---

# 4. 自适应阈值 · Robust Adaptive Threshold

CrackVision-DIC 不使用单一固定 strain threshold 直接判裂缝，而是根据峰值帧背景应变自动计算 threshold。

首先计算背景中位数：

```text
m = median(ε1)
```

Median Absolute Deviation：

```text
MAD = median( |ε1 - m| )
```

robust scale：

```text
σ_robust = 1.4826 · MAD
```

原始阈值：

```text
T_raw = m + k · σ_robust
```

当前默认：

```text
k = 2.0
```

最终 threshold：

```text
T = clip(T_raw, 2e-4, 5e-2)
```

候选 crack region 满足：

```text
ε1 ≥ T  and  ε1 > 0
```

随后执行：

```text
Small-object Removal
→ Skeletonization
→ Connected-component Labeling
→ Minimum Length Filter
```

当前默认：

```text
min_crack_area_points = 4
min_crack_length_mm   = 0.15 mm
```

```mermaid
flowchart TD
    A[Exx / Eyy / Exy] --> B[Maximum Principal Strain ε1]
    B --> C[Median + MAD]
    C --> D[Adaptive Threshold T]
    D --> E[Candidate Region]
    E --> F[Remove Small Objects]
    F --> G[Skeletonization]
    G --> H[Crack Labels]
    H --> I[Length Filter]
```

---

# 5. 局部裂缝坐标系 · Local Crack Coordinate System

对每个 crack skeleton point，程序根据邻域 skeleton coordinates 的 covariance / eigenvector 估计局部 tangent。

设单位切向量：

```text
t = (tx, ty)
```

则单位法向量：

```text
n = (-ty, tx)
```

因此裂缝宽度始终沿 **local crack normal** 测量，而不是机械地沿全局 `x` 或 `y` 方向测量。

这对 ECC / SHCC 中大量倾斜细裂缝尤其重要。

---

# 6. COD 计算 · Bilateral Displacement Regression

> **应变场负责“找裂缝”；裂缝宽度本身来自 displacement discontinuity，而不是高应变带的视觉厚度。**

设 DIC displacement vector：

```text
u_vec = (u, v)
```

沿 local normal 的位移分量：

```text
u_n = u · nx + v · ny
```

沿 local tangent 的位移分量：

```text
u_t = -u · ny + v · nx
```

其中：

- `u_n` → Crack Opening Displacement；
- `u_t` → tangential slip diagnostic。

## 6.1 裂缝两侧采样

每个 skeleton point 沿 crack normal 两侧采样。

默认物理采样范围：

```text
0.15 mm ≤ |s| ≤ 0.75 mm
```

当前默认：

```text
samples_per_side   = 7
min_valid_per_side = 3
```

## 6.2 双侧鲁棒拟合

正、负两侧分别拟合 normal displacement：

```text
u_n+(s) = a+ · s + b+
u_n-(s) = a- · s + b-
```

拟合过程中使用 residual MAD 进行 robust filtering。

默认残差保留规则：

```text
|r_i - median(r)| ≤ 3.5 · σ_robust
```

这样可以降低少量 DIC interpolation / correlation 异常点对最终 COD 的影响。

## 6.3 外推到 Crack Plane

在 crack plane：

```text
s = 0
```

两侧法向位移分别为：

```text
u_n+(0) = b+
u_n-(0) = b-
```

因此 local COD：

```text
COD_px = |b+ - b-|
```

位移场在 `FrameData` 中以 image-pixel displacement 保存，因此物理宽度为：

```text
COD_mm = COD_px · pixel_size_mm
COD_μm = COD_mm · 1000
```

```mermaid
flowchart TD
    A[Crack Skeleton Point] --> B[Estimate Local Normal]
    B --> C[Sample Positive Side]
    B --> D[Sample Negative Side]
    C --> E[Project U,V onto Normal]
    D --> F[Project U,V onto Normal]
    E --> G[Robust Line Fit +]
    F --> H[Robust Line Fit -]
    G --> I[Extrapolate to s = 0]
    H --> I
    I --> J[COD = abs b+ - b-]
```

---

# 7. 每条裂缝的代表宽度 · Crack-level Width

一条实际裂缝通常包含多个有效 local COD：

```text
Crack j:
w_j,1 , w_j,2 , ... , w_j,m
```

当前至少需要：

```text
m ≥ 3
```

该裂缝才进入最终统计。

单条裂缝的代表宽度定义为：

```text
W_crack(j) = median( local COD values of crack j )
```

选择 median 而不是单点宽度的原因：

- ECC 裂缝沿长度方向并非完全等宽；
- 单个 DIC 点可能受到局部散斑质量影响；
- median 对少量异常 COD 更稳健；
- 每条裂缝最终只得到一个明确的 representative width。

当前默认物理接受范围：

```text
1 μm ≤ W_crack ≤ 2 mm
```

Excel `02_裂缝明细` 中的 **代表宽度 (μm)** 就是该值。

---

# 8. 试件级统计 · Equal-weight Specimen Statistics

假设峰值拉应力帧最终识别出 `Nc` 条有效裂缝：

```text
W1, W2, W3, ..., WNc
```

每条裂缝只贡献一个代表宽度，因此所有裂缝具有**相同统计权重**。

平均裂缝宽度：

```text
W_mean = (W1 + W2 + ... + WNc) / Nc
```

中位裂缝宽度：

```text
W_median = median(W1, W2, ..., WNc)
```

P95 裂缝宽度：

```text
W_P95 = percentile(W1, W2, ..., WNc, 95)
```

最大裂缝宽度：

```text
W_max = max(W1, W2, ..., WNc)
```

这样可以避免长裂缝因为 COD sample 数更多而在 specimen-level statistics 中获得更高权重。

---

# 9. 科学参数 · Default Scientific Parameters

| 模块 | Parameter | Default | 含义 |
|---|---|---:|---|
| Detection | `threshold_k` | 2.0 | MAD 自适应阈值系数 |
| Detection | `min_tensile_strain` | 0.0002 | 最低主拉应变阈值 |
| Detection | `max_threshold` | 0.05 | 最高阈值限制 |
| Detection | `min_crack_area_points` | 4 | 最小候选区域点数 |
| Detection | `min_crack_length_mm` | 0.15 mm | 最短有效裂缝 |
| Detection | `normal_radius_points` | 4 | local normal 邻域 |
| COD | `near_mm` | 0.15 mm | 离裂缝最近采样距离 |
| COD | `far_mm` | 0.75 mm | 离裂缝最远采样距离 |
| COD | `samples_per_side` | 7 | 每侧请求采样点数 |
| COD | `min_valid_per_side` | 3 | 每侧最低有效点数 |
| COD | `min_samples_per_crack` | 3 | 单裂缝最低 local COD 数 |
| COD | `robust_sigma` | 3.5 | residual MAD filtering |
| COD | `min_width_mm` | 0.001 mm | 最小代表宽度 |
| COD | `max_width_mm` | 2.0 mm | 最大允许 COD |

参数定义位于：

```text
config/default.yaml
```

---

# 10. Excel 输出 · Paper-ready Workbook

每个试件自动生成：

```text
<Ncorr folder>/CrackVision_Output/<specimen>_CrackVision.xlsx
```

工作簿仅保留 3 张表。

## `01_结果汇总`

用于直接查看论文主结果：

```text
Peak Tensile Force
Peak Time
Selected DIC Frame
Frame Match Error
Accepted Crack Count
Mean Crack Width
Median Crack Width
P95 Crack Width
Maximum Crack Width
COD Status
```

同时包含峰值拉应力状态下各裂缝 representative width 图。

## `02_裂缝明细`

每行对应一条有效裂缝：

| 裂缝编号 | 裂缝长度 (mm) | 代表宽度 (μm) | COD 有效点数 | 拟合 R² 中位数 |
|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

论文绘制 crack-width distribution 时，直接使用 **代表宽度 (μm)**。

## `03_质量检查`

保留必要 QA：

```text
COD Status
MTS Peak Force / Time
Selected DIC Frame / Time
Frame Match Error
Pixel Scale
DIC Grid Spacing
Valid Fraction
Principal-strain Threshold
Candidate Points
Skeleton Points
Accepted Crack Count
Metadata Source
```

失败测量保持 `NaN`，不会静默改写为 `0 μm`。

---

# 11. GUI / UX

主界面只保留实际使用所需操作：

```mermaid
flowchart TD
    A[选择 Ncorr H5 / MAT] --> C[分析峰值拉应力帧]
    B[选择 MTS CSV] --> C
    C --> D[自动寻找 Peak Time]
    D --> E[匹配最近 DIC Frame]
    E --> F[计算全部有效裂缝 COD]
    F --> G[显示核心结果]
    G --> H[自动保存 Excel]
```

GUI 不再提供：

- output-directory picker；
- synchronization offset；
- manual scale input；
- progress bar；
- debug log panel；
- 多模式分析入口。

输出目录自动创建：

```text
CrackVision_Output/
```

---

# 12. 软件架构 · Architecture

```mermaid
flowchart TD
    GUI[PySide6 GUI] --> Worker[Worker Thread]
    Worker --> Pipeline[Peak-frame Pipeline]

    Pipeline --> MTS[MTS CSV Parser]
    Pipeline --> Input[Input Selection]

    Input --> H5[CrackVision HDF5 Reader]
    Input --> MAT[Legacy Ncorr MAT Reader]

    H5 --> Frame[FrameData]
    MAT --> Frame

    Frame --> Physics[CrackPhysicsEngine]
    Physics --> Summary[Equal-weight Crack Summary]
    Summary --> Export[Excel Export]
```

模块职责：

```text
GUI       → 输入与结果展示
Worker    → 后台线程生命周期
Pipeline  → Peak Time → DIC Frame → COD → Export
MTS       → 解析原始 MTS CSV
Input     → 选择目标 DIC frame
Reader    → 读取 Ncorr 数据
Physics   → 裂缝识别 + COD
Export    → Paper-ready Excel
```

设计原则：

```text
GUI 不知道 Ncorr 内部结构
Worker 不实现科研计算
Reader 不计算 COD
Physics 不依赖 Qt / Excel
Export 不改变科研结果
```

---

# 13. 项目结构 · Project Structure

```text
CrackVision-DIC/
├─ main.py
├─ config/
│  └─ default.yaml
├─ matlab/
│  └─ export_ncorr_to_crackvision.m
├─ src/
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ mts.py
│  │  ├─ input.py
│  │  ├─ io_bridge.py
│  │  ├─ io_ncorr.py
│  │  ├─ models.py
│  │  ├─ physics.py
│  │  ├─ pipeline.py
│  │  └─ export.py
│  └─ gui/
│     ├─ main_window.py
│     └─ worker.py
└─ tests/
```

---

# 14. 安装与运行 · Installation

推荐 Python 3.10+。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

安装依赖并启动：

```bash
pip install -r requirements.txt
python main.py
```

---

# 15. Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

当前测试覆盖：

- principal strain calculation；
- horizontal / vertical crack COD；
- continuous background deformation regression；
- Ncorr spacing semantics；
- classic MAT / v7.3 MAT loading；
- compact HDF5 bridge；
- MTS peak-force parsing；
- positive / negative tension sign；
- peak-frame selection；
- equal-weight crack statistics；
- Excel workbook layout。

---

# 16. 方法学总结 · Method Summary

CrackVision-DIC 当前方法可以压缩为以下 9 步：

```text
1. t_peak = arg max_t [s · F(t)]
2. k* = arg min_k |t_DIC(k) - t_peak|
3. ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
4. T = clip(median(ε1) + 2.0 × 1.4826 × MAD, 2e-4, 5e-2)
5. Candidate → Skeleton → Crack Labels → Local Normal
6. u_n = u · nx + v · ny
7. COD = |b+ - b-| × pixel_size_mm
8. W_crack = median(local COD values)
9. W_mean = mean(W_crack,1 ... W_crack,Nc)
```

其中：

> **Strain tells CrackVision where the crack is; displacement discontinuity tells CrackVision how wide it is.**

这也是本项目与 image-based apparent crack-width measurement 最核心的区别。
