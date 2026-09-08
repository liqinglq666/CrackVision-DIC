# CrackVision-DIC

CrackVision-DIC is a focused post-processing tool for **ECC / SHCC tensile-test DIC data calculated by Ncorr**.

The project intentionally does one thing:

```text
Ncorr H5 / MAT
      +
MTS original CSV
      ↓
peak tensile-force time
      ↓
nearest DIC frame
      ↓
U / V / Exx / Eyy / Exy
      ↓
maximum principal tensile strain
      ↓
crack skeleton
      ↓
local crack normal
      ↓
multiple U/V samples on both crack faces
      ↓
regression + crack-plane extrapolation
      ↓
COD crack width
      ↓
one representative width per crack
      ↓
equal-weight specimen statistics
      ↓
paper-ready Excel
```

For a constant specimen cross-section, peak tensile force and peak tensile stress occur at the same time. Therefore specimen area is not needed to select the target DIC frame.

## Assumed test synchronization

This project workflow assumes **MTS recording and image acquisition start together**:

```text
MTS t = 0 s
DIC frame 0 = 0 s
```

There is no synchronization-offset control in the GUI.

## MTS CSV

The parser supports the MTS / DAQ CSV layout used in this project, including metadata rows before the data table. It automatically locates force/load and time columns such as:

```text
横梁 | 力 | 时间 | 引伸计 | ...
mm   | N  | sec  | mm     | ...
```

Positive- and negative-tension sign conventions are both supported.

## Recommended Ncorr input

For new tests, use the compact CrackVision HDF5 bridge instead of making Python parse a very large Ncorr MAT.

After Ncorr has completed **Format Displacements**, physical-unit conversion in mm and **Calculate Strains**:

```matlab
handles_ncorr = ncorr;
% ... normal Ncorr workflow ...

addpath('path/to/CrackVision-DIC/matlab')
export_ncorr_to_crackvision( ...
    handles_ncorr, ...
    'Specimen01_CrackVision.h5', ...
    5);   % 5 s/image
```

The bridge stores only the data required for crack-width analysis:

```text
U, V
Exx, Eyy, Exy
finite mask
time
pixel_size_mm
Ncorr spacing
DIC step
DIC point spacing in mm
```

Existing Ncorr `.mat` files remain supported. H5 is preferred because CrackVision can random-read only the selected peak-stress frame.

## Crack-width definition

Crack position is detected from maximum principal tensile strain:

```text
ε1 = (Exx + Eyy)/2 + sqrt(((Exx - Eyy)/2)^2 + Exy^2)
```

For each accepted crack, CrackVision samples the Ncorr displacement field on both crack faces, fits the local displacement fields and extrapolates them to the crack plane. The normal displacement discontinuity is the local COD.

A crack usually contains many valid local COD measurements. Its representative width is:

```text
W_crack = median(local COD values along that crack)
```

At specimen level, every accepted crack contributes exactly one representative width with equal weight:

```text
w1, w2, ..., wn

mean crack width = mean(w1, w2, ..., wn)
```

The paper-facing specimen metrics are:

```text
Crack_width_mean_um
Crack_width_median_um
Crack_width_95_um
Crack_width_max_um
```

Failed measurements remain `NaN`; they are never silently changed to `0 μm`.

## Minimal GUI

The main window contains only:

```text
1. Select Ncorr H5 / MAT
2. Select matching MTS CSV
3. Analyse peak tensile-stress frame
```

There is no output-directory picker, synchronization input, scale input, progress bar or debug log panel.

Results are saved automatically to:

```text
<Ncorr folder>/CrackVision_Output/<specimen>_CrackVision.xlsx
```

After analysis the UI shows only the useful result summary:

```text
peak force + peak time
selected DIC frame + time mismatch
accepted crack count
mean / P95 / maximum crack width
output path
```

## Excel output

Each specimen creates one compact workbook with three sheets:

```text
01_结果汇总
02_裂缝明细
03_质量检查
```

### 01_结果汇总

Paper-facing dashboard containing:

```text
peak tensile force
peak time
selected DIC frame
frame-match error
accepted crack count
mean crack width
median crack width
P95 crack width
maximum crack width
COD status
```

It also contains one chart showing the representative width of each accepted crack.

### 02_裂缝明细

One row per accepted crack, with only the fields needed for reporting and audit:

```text
裂缝编号
裂缝长度 (mm)
代表宽度 (μm)
COD 有效点数
拟合 R² 中位数
```

`代表宽度 (μm)` is the crack's median local COD and is the value to use when plotting the crack-width distribution.

### 03_质量检查

Contains only the parameters needed to verify that the exported result is trustworthy:

```text
COD status
MTS peak time / force
selected DIC frame / time
frame-match error
pixel scale
DIC grid spacing
valid fraction
principal-strain threshold
candidate / skeleton point counts
metadata source
```

## Run

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```text
CrackVision-DIC/
├─ main.py
├─ config/default.yaml
├─ matlab/export_ncorr_to_crackvision.m
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

The project intentionally does not include camera crack segmentation, image-based width estimates, MTS curve dashboards, batch report systems, multiprocessing/temp-file plumbing or multiple competing analysis modes.
