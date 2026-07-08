# CrackVision-DIC

Ncorr/DIC 导出的 `.mat` 很难直接写论文；CrackVision-DIC 把 DIC 位移/应变场和 MTS 曲线接起来，批量算裂缝数量、裂缝间距、COD、宽度分位数和 QA 表。

它干一件事：把 DIC 后处理从“手搓 Excel 地狱”拉回工程管线。

## 能处理什么

输入：

```text
DIC .mat
├─ u displacement map
├─ v displacement map        # 算斜裂缝 COD 强烈建议有
├─ exx strain map
├─ valid mask / quality map   # 有就用，没有就 finite-only
└─ scale / spacing / time metadata

MTS .csv                     # 可选
├─ time
├─ force / load
└─ displacement / extension
```

输出：

```text
[Specimen]_Origin_Plot_Data.xlsx
[Specimen]_Statistics_Report.xlsx
_Batch_Summary.xlsx
```

算这些：

```text
crack_count
crack_spacing_mm
W_avg_um
W_99_um
W_max_um
global_strain
Stress_MPa
QA_Metadata
```

不算这些：

```text
原始散斑图像相关匹配
DIC solver
自动修复烂掉的实验数据
```

原始 DIC 仍然交给 Ncorr、DICe、VIC-2D。这个仓库只管后处理。分工清楚，少点玄学。

## 🚀 快速开始

要求：Python >= 3.10。

```bash
git clone https://github.com/liqinglq666/CrackVision-DIC.git
cd CrackVision-DIC

conda create -n crackvision_env python=3.10 -y
conda activate crackvision_env

pip install -r requirements.txt
python main.py
```

跑测试：

```bash
pytest
```

没 GUI？先看 PySide6 装没装：

```bash
python -c "import PySide6; print('PySide6 OK')"
```

## 怎么用

### 单个试件

1. 打开软件。
2. 选择 `Single`。
3. 选 DIC `.mat`。
4. 有 MTS 曲线就选 `.csv`，没有就留空。
5. 选输出目录。
6. 点运行。

### 批处理

1. 选择 `Batch`。
2. 选 DIC 工作目录。
3. 打开挂载台。
4. 选 MTS 目录，自动匹配。
5. 检查匹配结果。
6. 跑。

自动匹配现在是严格 token 匹配。`E1` 不会乱配 `E10`。匹配不到就手动指定，别和文件名硬刚。

## 配置

主配置：

```text
config/default.yaml
```

常用参数：

```yaml
experiment:
  # raw image scale，位移 u/v 从 px 转 mm 用它
  mm_per_pixel: 0.045

  # DIC 网格点间距，单位是 raw pixel / DIC point
  dic_subset_spacing_px: 1.0

  # .mat 没有真实时间轴时，用 Frame_ID * 这个值
  sampling_interval_s: 5.0

  # MTS 应力换算：Stress_MPa = Force_N / area_mm2
  cross_section_area_mm2: 100.0

  # MTS 位移应变换算用
  gauge_length_mm: 80.0

sync:
  # DIC 和 MTS 时间轴至少重叠多少才同步
  min_overlap_fraction: 0.60

  # 插值后最多允许多少 NaN
  max_missing_fraction: 0.05

  # 触发不同步就调这里。正数 = 时间轴往后挪
  dic_time_offset_s: 0.0
  mts_time_offset_s: 0.0

physics:
  # exx 裂缝候选区阈值：median + k * 1.4826 * MAD
  strain_threshold_k: 1.5

  # 斜裂缝 COD 需要 v map。没有 v 就别假装很准
  require_v_map_for_cod: true

  # COD 底噪过滤，单位 mm
  cod_min_mm: 0.002
  cod_min_mean_mm: 0.002
  cod_max_mm: 5.0

  # 单轴拉伸建议开；循环加载/卸载关掉
  enforce_monotonic_strain: true
```

改完配置，重启 GUI。别赌热加载。代码不欠你这个魔法。

## 计算口径

### 尺度

别把 DIC grid point 当 raw pixel。那是灾难的开始。

```text
pixel_size_mm        = raw image pixel scale, mm / px
subset_spacing_px    = DIC subset center spacing, px / point
dic_point_spacing_mm = pixel_size_mm * subset_spacing_px
```

用法：

```text
u/v 位移换算       -> pixel_size_mm
裂缝长度/搜索半径  -> dic_point_spacing_mm
virtual gauge 距离 -> dic_point_spacing_mm
```

### 时间轴

```text
优先级：
1. .mat 里的 time / time_s / frame_time / timestamp
2. Frame_ID * sampling_interval_s
3. 同步失败就保留纯 DIC，写 sync_status
```

看输出里的：

```text
dic_time_source
sync_status
```

这两个字段比感觉靠谱。

### COD

默认算法：法向位移跳量。

```text
w_mm = abs((u_plus - u_minus) * nx + (v_plus - v_minus) * ny) * pixel_size_mm
```

如果 `require_v_map_for_cod: true` 但 `.mat` 没有 v 位移场，结果会写：

```text
missing_v_map_required
```

这不是 bug。是软件拒绝陪你一起骗自己。

### 裂缝骨架

`exx` 场先过稳健阈值：

```text
threshold = median(exx) + k * 1.4826 * MAD
```

然后：

```text
remove small objects
skeletonize
sample COD along normal direction
filter by length / COD floor / COD ceiling
```

## 输出文件

导出逻辑已经改成更适合画图和批量汇总的排布。核心原则：

```text
逐帧数据 -> 一行一帧
裂缝数据 -> 一行一条裂缝
分布数据 -> long format，不再横着乱铺
目标应变点 -> 达不到也保留 not_reached 行
批处理汇总 -> 单独写 _Batch_Summary.xlsx
```

### `[Specimen]_Origin_Plot_Data.xlsx`

给 Origin / Python / Excel 画图用。表尽量少，字段直接。

```text
00_READ_ME              # 每个 sheet 怎么用
01_Frame_Curves         # 一行一帧：strain/stress/crack count/spacing/COD
02_Target_States        # 目标应变点摘要；没达到也写 not_reached
03_Distribution_Tidy    # 一行一个分布值：State + Metric + Value_um
04_Crack_Tidy           # 一行一条裂缝：State/Frame/Crack_ID/Length/Width
```

最常用：

```text
画时序曲线 -> 01_Frame_Curves
画箱线图/小提琴图 -> 03_Distribution_Tidy
做裂缝级透视表 -> 04_Crack_Tidy
```

### `[Specimen]_Statistics_Report.xlsx`

给论文表格、复核、查错用。别拿这个直接画图，除非你喜欢折磨自己。

```text
00_Specimen_Summary     # 单试件总摘要：UTS、极限应变、饱和裂缝、尺度、同步状态
01_Frame_All            # 完整逐帧表，保留 QA/同步/COD 状态
02_Target_Summary       # 目标应变点摘要
03_Key_Crack_Details    # First_Crack / Saturated / Ultimate / Max_Width 裂缝明细
04_Distribution_Tidy    # 分布长表备份
05_QA_Frame_Status      # 一行一帧的 QA 状态
06_QA_Metadata          # 元数据来源、尺度、时间轴、状态计数
07_Validation           # 人工标注对比，没标注就提示
```

### `_Batch_Summary.xlsx`

批处理时自动更新。不是每个试件一个，而是整个输出目录一个。

```text
Specimen_Summary        # 每个试件一行
Target_Summary          # 每个试件 × 每个目标应变点一行
```

拿它做组间对比。别再把十几个单试件 Excel 手动复制进一个总表。那种操作属于考古，不属于科研。

## QA 必看

打开 `05_QA_Frame_Status` 和 `06_QA_Metadata`。别跳过。

重点看：

```text
pixel_size_mm
subset_spacing_px
dic_point_spacing_mm
metadata_source
dic_time_source
v_map_present
quality_valid_fraction
cod_status
sync_status
strain_source
```

判断规则很粗暴：

```text
metadata_source 里全是 config_fallback  -> 去确认比例尺和 subset spacing
dic_time_source 是 frame_index_interval_fallback -> 去确认 sampling_interval_s
v_map_present 是 False 且 require_v_map_for_cod 是 true -> COD 不会算
quality_valid_fraction 很低 -> 这帧别拿去吹论文
sync_status 不是 synced -> MTS 没同步上，别画 stress-strain 联动结论
```

## 人工标注对照

在 `.mat` 同目录放一个文件：

```text
[Specimen]_annotations.csv
```

最小格式：

```csv
Frame,crack_count,W_avg_um,W_max_um
0,0,0,0
10,3,18.2,41.5
20,7,31.0,86.4
```

软件会写入：

```text
07_Validation
```

没有人工标注也能跑。只是少一层保险。

## 常见翻车点

### MTS 同步失败

先看：

```yaml
experiment:
  sampling_interval_s: 5.0

sync:
  dic_time_offset_s: 0.0
  mts_time_offset_s: 0.0
```

再看 CSV 表头有没有这些词：

```text
time / sec / second / 时间 / 秒
load / force / kn / 载荷 / 力
disp / displacement / extension / 位移 / 伸长
```

### COD 全是 0

检查：

```yaml
physics:
  require_v_map_for_cod: true
```

再看 QA：

```text
v_map_present
cod_status
```

如果是：

```text
missing_v_map_required
```

那就补 v map，或者把 `require_v_map_for_cod` 改成 `false`。后者能跑，但斜裂缝会变得不那么物理。便利和准确，自己选。

### 裂缝太多

调高：

```yaml
physics:
  strain_threshold_k: 2.0
  min_crack_area_points: 20
  cod_min_mm: 0.005
```

### 主裂缝断裂

调低一点：

```yaml
physics:
  strain_threshold_k: 1.2
```

或者检查 DIC 失相关。散斑烂了，算法不是巫术。

## 本地开发

```bash
pip install -r requirements.txt
pytest
python main.py
```

只想测核心契约：

```bash
pytest tests/test_core_contracts.py
```

当前测试盯着几件要命的事：

```text
缺 v map时明确返回 missing_v_map_required
DIC strains/displacements 帧数不一致时直接报错
.mat 有时间轴就用 metadata time
.mat 没时间轴就用 sampling_interval_s
Excel 导出必须去掉 object payload
目标应变没达到也必须保留 not_reached 行
裂缝分布必须是 tidy long format
```

## 目录

```text
CrackVision-DIC
├─ main.py
├─ config/
│  └─ default.yaml
├─ src/
│  ├─ core/
│  │  ├─ io_sync.py              # MAT 读取、metadata、时间轴
│  │  ├─ physics.py              # skeleton、COD、quality mask
│  │  ├─ evolution_analyzer.py   # MTS CSV 同步
│  │  └─ models.py
│  └─ gui/
│     ├─ main_window.py
│     └─ worker.py               # Excel 导出也在这里
├─ tests/
│  └─ test_core_contracts.py
└─ requirements.txt
```

## 一句话结论

先看 `_Batch_Summary.xlsx`，再用 `01_Frame_Curves` 画时序，用 `03_Distribution_Tidy` 画分布。别再手动复制粘贴。那不是科研，是体力劳动 cosplay。
