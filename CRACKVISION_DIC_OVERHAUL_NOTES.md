# CrackVision-DIC Core Overhaul Notes

This package contains the core changes for a more reliable DIC crack-analysis pipeline.

## What Was Fixed

1. Scale and DIC step spacing are now separated.
   - `pixel_size_mm` converts Ncorr displacement values from pixels to mm.
   - `dic_point_spacing_mm = pixel_size_mm * subset_spacing_px` converts DIC grid steps to physical length.
   - Crack length, search distance, and virtual extensometer length now use the DIC grid spacing instead of raw camera pixel scale.

2. COD calculation now uses vector-normal displacement jump.
   - Default mode requires both `u` and `v` maps.
   - Width is calculated as `abs(du * nx + dv * ny) * pixel_size_mm`.
   - You can disable this strict requirement with `physics.require_v_map_for_cod: false`, but that is less reliable for angled cracks.

3. Global strain now uses a virtual extensometer.
   - The software samples two vertical bands inside the DIC valid region.
   - Default bands are at 10% and 90% of the valid DIC width.
   - If `experiment.virtual_extensometer.gauge_length_mm` is empty, the gauge length is computed from DIC point spacing.
   - MTS displacement can be exported as `MTS_Strain` but does not overwrite DIC strain unless configured.

4. MTS synchronization is now strict.
   - It rejects non-overlapping time axes.
   - It rejects very small overlap.
   - It uses no extrapolation by default.
   - It keeps DIC virtual strain if MTS sync fails.

5. DIC quality filtering is supported.
   - If a quality/correlation map exists in the MAT file, finite-value filtering is applied by default.
   - Optional threshold filtering can be enabled in `config/default.yaml`.
   - Each output frame records quality coverage and COD status.

6. Validation output is added.
   - `Statistics_Report.xlsx` now contains `05_QA_Metadata` and `06_Validation`.
   - If no manual annotation file exists, the report states that explicitly.

## Manual Annotation Format

Place one of these files next to the `.mat` file:

- `<specimen>_annotations.csv`
- `<specimen>_annotation.csv`
- `<specimen>_annotations.xlsx`

Minimum required column:

```csv
Frame
```

Useful optional columns:

```csv
Frame,crack_count,W_avg_um,W_max_um
0,0,0,0
10,5,28.4,65.2
20,8,42.1,101.7
```

The validation sheet compares calculated results with manual values and reports MAE, bias, and max absolute error.

## Important Parameters

```yaml
experiment:
  mm_per_pixel: 0.045
  dic_subset_spacing_px: 1.0
  virtual_extensometer:
    left_fraction: 0.10
    right_fraction: 0.90
    band_width_points: 3
    gauge_length_mm:

sync:
  min_overlap_fraction: 0.60
  max_missing_fraction: 0.05
  override_dic_strain_with_mts: false

quality:
  enabled: true
  metric_mode: finite_only
  threshold:
  min_valid_fraction: 0.20

physics:
  require_v_map_for_cod: true
  max_cracking_strain_threshold: 0.03
```

## Verification Status

The replacement files pass Python syntax compilation in the Codex workspace. Full runtime verification still requires the project dependencies (`numba`, `PySide6`, `scikit-image`, etc.) and a real Ncorr `.mat` file.
