# Day 6 — RGB-D Localization Error Diagnosis and Improvement

## Objective

Investigate systematic localization errors observed in a simulated RGB-D perception pipeline and evaluate independent improvements to object surface selection, pixel localization, and vertical pixel alignment.

## Experimental Setup

- Simulator: PyBullet TinyRenderer
- Camera: Fixed overhead RGB-D camera
- Resolution: 320 × 240
- Target: Red cube, with the top-face center used as ground truth
- Evaluation: Nine predetermined object positions
- Baseline: RGB thresholding, integer-pixel selection, and depth-based 3D back-projection

## Error Diagnosis

**Vertical alignment:** Ground-truth reprojection revealed a nearly constant one-pixel vertical discrepancy between projected object centers and rendered mask centroids. The discrepancy persisted across three image resolutions (320 × 240, 640 × 480, and 321 × 241), suggesting a renderer-specific pixel alignment or rasterization convention.

**Horizontal localization:** RGB and segmentation mask comparisons showed that visible cube side faces influenced the detected object centroid. Depth analysis confirmed that additional side-face pixels were approximately 24.29 mm behind the nearest top surface in affected trials. Integer-pixel selection introduced further horizontal localization error in some positions.

## Ablation Study

Eight variants evaluated three factors: integer versus subpixel selection, RGB-mask versus depth-filtered top-face selection, and uncorrected versus experimentally adjusted vertical pixel alignment.

| Metric | Baseline A | Improved G |
|---|---:|---:|
| Mean 3D error | 4.903 mm | 0.463 mm |
| P90 error | 6.114 mm | 0.626 mm |
| Maximum error | 6.284 mm | 0.631 mm |

The improved configuration combined depth-based top-face filtering with a one-pixel vertical alignment adjustment. Mean 3D localization error decreased by **90.55%** across the nine evaluated scenes.

Subpixel centroid selection improved some configurations but provided no additional improvement over the top-face-based method in this experiment.

## Validation

- 72 localization estimates across eight configurations
- Baseline consistency check passed
- All 27 automated tests passed
- All nine evaluated positions showed lower 3D error with configuration G than with the baseline

## Limitations and Next Steps

The evaluation used the same nine simulated positions employed during error diagnosis. The one-pixel vertical adjustment is empirical and should not be assumed valid for other renderers or camera configurations.

The top-face filtering method relies on a depth threshold and may be sensitive to noise, surface orientation, and object geometry.

Further experiments will evaluate noisy and missing depth measurements, unseen object positions, and alternative camera configurations before making claims about robustness or generalization.