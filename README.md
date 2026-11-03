# RGB-D Perception for Robotic Reaching

**Simulation-based RGB-D localization, robustness analysis, and perception-driven robotic reaching**

This project develops and evaluates an RGB-D perception pipeline that estimates the **3D World-coordinate position of a target object's top-face center** from rendered color and depth images. The work covers camera geometry, systematic error diagnosis, robust depth estimation, and generalization studies under unseen positions, altered viewpoints, and corrupted depth measurements.

The resulting perception system is integrated with a simulated **Franka Panda** robot using position-only inverse kinematics and joint-space position control. A single clean RGB-D scene demonstrates end-to-end perception-driven reaching. **Localization generalization and depth robustness are evaluated separately; robotic reaching across multiple scenes or corruption conditions has not yet been evaluated.**

## Highlights

- Built an RGB-D pipeline from **camera intrinsics and extrinsics** to a 3D target in World coordinates.
- Reduced mean 3D localization error from **4.903 mm (baseline A)** to **0.463 mm (method G)** on nine original clean overhead scenes—a **90.55% reduction**, measured on development scenes.
- Developed a robust estimator (**method R**) using depth median/MAD filtering, a continuous surface centroid, and explicit validity checks.
- Evaluated **33 positions × 4 camera viewpoints = 132 clean scenes**. Method R returned a position within the experimental 5 mm tolerance in all 132 cases, including 96 scenes involving previously unseen positions.
- Stress-tested the frozen methods on **4,128 clean/corrupted input scenarios**, producing **12,384 paired localization predictions** on 96 held-out position/viewpoint combinations.
- Identified **failure mechanisms**, not just average error: nearest-depth selection biases under oblique views, inaccurate returns under severe depth dropout, and limitations of bounding-span confidence checks.
- Integrated frozen **method R → Panda IK → simulated joint control**, achieving **2.182 mm final reaching error** in one clean scene; **119 automated tests passed** across the repository.

> **Scope of the evidence:** Results come from controlled **PyBullet simulations** with one 5 cm red cube. The 5 mm localization criterion is an experimental metric, not a robot safety threshold. **The 132 clean scenes and 12,384 predictions evaluate localization, whereas reaching has been demonstrated in one clean scene only.**

## Research questions

1. How can RGB pixels and metric depth be transformed reliably into 3D World coordinates?

2. Why do object masks, surface selection, and camera conventions introduce systematic localization error?

3. How do Gaussian depth noise and missing-depth patterns affect accuracy and the ability to reject unreliable estimates?

4. Does a robust localization method retain accuracy at **unseen object positions** and **oblique camera viewpoints**?

5. Under what conditions does a method **return an inaccurate coordinate** instead of explicitly **rejecting** an estimate?

6. Can the frozen RGB-D estimator drive a simulated robot toward an estimated 3D approach point, and how do perception, IK, and control errors relate to the final reaching error?

## Perception-to-reaching pipeline

```text
PyBullet scene: red cube + calibrated RGB-D camera
        │
        ├── RGB ──► red-object mask ───┐
        └── Depth ──► metric depth ────┤
                                      ▼
                    Frozen method R: robust surface selection
                                      │
                    Pixel + depth + camera intrinsics/view
                                      │
                                      ▼
                    3D top-face estimate (World frame)
                                      │
                       + 0.10 m World-Z offset
                                      │
                                      ▼
                         Approach target (World)
                                      │
                                      ▼
                        Position-only Panda IK
                                      │
                            Validated joint angles
                                      │
                                      ▼
                       Joint-space position control
                                      │
                                      ▼
                    Measured end-effector (World)
                                      │
                                      ▼
            Final reaching error (evaluation with simulator GT)
Simulator GT ──────────────────► Evaluation and diagnostics ONLY
                               (never a perception/IK/control input)
```

The frozen estimator uses **RGB, metric depth, camera intrinsics, and the camera view matrix**. Its optical-camera-to-World transformation is unchanged from the localization benchmarks. Simulator cube poses and segmentation IDs are **not** used as inputs to RGB detection or robot target commands. The reaching demonstration commands a fixed approach point once per scene; it is **not** closed-loop visual servoing.

### Camera and geometry

- **Simulator / renderer:** PyBullet, CPU TinyRenderer (`DIRECT` mode).
- **Target:** Red cube, edge length **0.05 m**, resting on a horizontal plane.
- **Image:** **320 × 240** RGB-D pixels, vertical field of view **60°**.
- **Depth:** PyBullet depth buffer converted to **metric optical-axis depth**; clipping planes **0.05 m** and **2.0 m**.
- **Coordinates:** Optical frame (X right, Y down, Z forward) → OpenGL camera frame → World frame using the inverse of the PyBullet view matrix.
- **Ground truth:** Cube **top-face center** in World coordinates, computed from the simulated object pose and orientation—not the cube's geometric center.

The clean overhead reference camera is located at `(0, 0, 1.0)` m and points toward the World origin. Generalization experiments also use oblique X, oblique Y, and diagonal views, each directed toward the origin at a fixed **1 m eye-to-origin distance**.

## Localization methods

| Method | Surface and pixel selection | Depth estimation | Validity checks |
|---|---|---|---|
| **A — Baseline** | RGB mask; integer pixel near mask centroid; no vertical correction | Median of valid RGB-mask depths | Fails if no usable pixels |
| **G — Minimum-depth selection** | RGB pixels within **1 mm of the minimum valid depth**; integer selected pixel; empirical `+1` vertical pixel offset | Median of selected depths | Fails if selected surface is empty |
| **R — Robust surface estimator** | Depth median/MAD inliers; **continuous** inlier centroid; same `+1` vertical pixel offset | Median of inlier depths | Minimum **10** valid/inlier pixels and **50%** horizontal/vertical support spans |

The `+1` vertical-pixel correction was found empirically when diagnosing the TinyRenderer projection/rasterization alignment; **it is not a general camera calibration rule**. A, G, and R are whole-pipeline configurations, so their comparisons do **not** isolate the effect of each individual component. Their parameters were frozen before the held-out evaluation.

## Implementation milestones

| Component | Implementation / investigation | Outcome |
|---|---|---|
| Simulation foundation | PyBullet scene, virtual camera, RGB/depth/segmentation rendering, metric depth, GT export | Reproducible synthetic RGB-D inputs |
| Camera geometry | Intrinsics, pixel back-projection, optical/OpenGL-camera/World transforms | Numerical and round-trip geometry validation |
| Object detection | RGB red-object masking and mask-based pixel selection | Object localization region detected without GT segmentation |
| Initial 3D localization | Single-pixel and neighborhood-median methods; nine positions | First quantitative localization benchmark |
| Error diagnosis | GT reprojection, side-face/segmentation alignment, eight-variant ablation | A: 4.903 mm → G: 0.463 mm mean on development scenes |
| Robust estimation | Gaussian noise, random/patch dropout, method R, outcome classification | Depth-robustness benchmark and failure analysis |
| Generalization | 33 positions, four viewpoints, corruption-transfer tests, coverage and pixel-grid diagnostics | Localization performance and limitations characterized |
| Robot integration | Panda URDF, robot frames, position-only FK/IK, joint control | Executable perception-driven reaching pipeline |
| Single-scene reaching audit | EE trajectory, error decomposition, error-identity checks, failure gates | 2.182 mm final reaching error; reproducible report |

## Experimental results

### 1. Clean localization: previously used vs. unseen positions

The nine development positions were used while designing the methods. The **16 interpolation** positions are new locations inside the previous XY bounds; the **eight extrapolation** positions lie modestly outside those bounds.

**Overhead camera — mean 3D error (mm):**

| Position group | Scenes | A | G | R | R within 5 mm |
|---|---:|---:|---:|---:|---:|
| Development reference | 9 | 4.903 | 0.463 | 0.463 | 9/9 |
| Held-out interpolation | 16 | 5.034 | 2.342 | **1.949** | 16/16 |
| Held-out extrapolation | 8 | 6.513 | 1.954 | **1.954** | 8/8 |

Method R satisfied the 5 mm criterion at all 24 new overhead positions. However, its mean error increased from **0.463 mm** on development positions to approximately **1.95 mm** at new positions. This is evidence of useful accuracy within the evaluated workspace—not unchanged submillimeter precision or unrestricted spatial generalization.

### 2. Clean localization: viewpoint generalization

The methods were evaluated at the **same 24 held-out positions** from four different camera views.

**Mean 3D error (mm), 24 positions per view:**

| Camera viewpoint | A | G | R | R within 5 mm |
|---|---:|---:|---:|---:|
| Overhead | 5.527 | 2.213 | **1.951** | 24/24 |
| Oblique X | 5.058 | 23.065 | **1.776** | 24/24 |
| Oblique Y | 6.243 | 22.118 | **1.447** | 24/24 |
| Oblique diagonal | 9.324 | 32.027 | **2.241** | 24/24 |

Across all **132 clean scenes** (including 36 development-position scenes), **method R returned a coordinate within 5 mm in every case**; its largest observed error was **3.994 mm**. In contrast, method G returned coordinates exceeding 5 mm in all **99 oblique-view scenes**.

![Mean clean localization error across camera views](outputs/generalization_clean_viewpoint_errors.png)

**Why G degraded:** Under oblique viewpoints, its minimum-depth criterion concentrated selection on a small, camera-near portion of the observed surface. For the oblique-diagonal view on held-out scenes, G selected **3.0 pixels on average**, versus **124.0** for R; G's mean image-center residual was **6.815 pixels**, with **−7.884 mm** signed optical-depth bias. These diagnostics support a near-edge selection-bias explanation rather than a coordinate-frame failure.

[Inspect the selected-surface visualization](outputs/generalization_surface_selection.png)

### 3. Depth robustness on the original development scenes

Before evaluating unseen scenes, the methods were compared on the original **nine overhead positions** under controlled synthetic depth corruption. The initial paired robustness benchmark covered **1,314 input scenarios** and **3,942 A/G/R evaluations**; these are **development-set** results, not held-out generalization evidence.

**Mean 3D error (mm) under Gaussian depth noise:**

| Noise standard deviation | A | G | R |
|---|---:|---:|---:|
| 0 mm (clean) | 4.903 | 0.463 | 0.463 |
| 1 mm | 4.903 | 11.932 | **0.490** |
| 3 mm | 4.913 | 19.598 | **0.577** |
| 5 mm | 4.940 | 22.596 | **0.686** |

For **80% random depth dropout**, R's mean error over returned estimates was **3.408 mm**, but **18.9%** of inputs yielded an inaccurate return (>5 mm), not a rejection. Under an **11×11 centered missing-depth patch**, R rejected all **nine** development scenes instead of returning a coordinate; a subsequent held-out test revealed more complex failures under changed viewpoints.

[Gaussian noise comparison](outputs/robust_gaussian_error.png) · [Depth-robustness outcome classification](outputs/robust_localization_outcomes.png)

### 4. Depth corruption transferred to unseen positions/viewpoints

A returned estimate is classified as **accurate** when its 3D error is **≤ 5 mm**, **inaccurate** when it is **> 5 mm**, or **rejected** when no valid coordinate is produced. All cases—including rejections—are in the denominators of these rates. Mean / P90 / maximum errors are computed **over returned estimates only**.

The held-out transfer benchmark used **96 held-out position/viewpoint scenes**, seven corruption conditions, **4,128 input scenarios**, and **12,384 A/G/R evaluations**. Each corrupted RGB-D input was shared across the three methods; random corruptions used reproducible seeds.

**Method R, across all four camera views:**

| Input condition | Inputs | Accurate return | Inaccurate return | Rejected |
|---|---:|---:|---:|---:|
| Clean | 96 | 96 (100%) | 0 | 0 |
| Gaussian noise, σ = 3 mm | 960 | 960 (100%) | 0 | 0 |
| Gaussian noise, σ = 5 mm | 960 | 960 (100%) | 0 | 0 |
| Random depth dropout, 40% | 960 | 950 (99.0%) | 10 (1.0%) | 0 |
| Random depth dropout, 80% | 960 | 703 (73.2%) | **257 (26.8%)** | 0 |
| Centered 5×5 missing-depth patch | 96 | 96 (100%) | 0 | 0 |
| Centered 11×11 missing-depth patch | 96 | 0 | **13 (13.5%)** | **83 (86.5%)** |

![Method R outcomes by viewpoint and corruption](outputs/generalization_robustness_outcomes.png)

**Important failure modes:**

- **Severe random dropout:** At 80% dropout, R returned an incorrect coordinate in **257/960 cases** without rejecting any estimate. Its minimum-count and span checks were not sufficient to detect all inaccurate returns.
- **Spatially incomplete depth:** Under an 11×11 patch, some retained inliers spanned most of the object's horizontal and vertical extent while filling little of the enclosed region. In **12 inaccurate oblique-diagonal cases**, mean normalized spans were **0.979 / 0.962**, but mean bounding-box fill was only **0.117**. This illustrates why separate width/height coverage tests can accept unrepresentative surface support.
- **Clean-image sampling bias:** On overhead scenes, R's residual error matched a centroid-only reconstruction while depth-only error was approximately zero. New-location image-centroid offsets averaged approximately **0.426–0.428 pixels**, consistent with pixel-grid/rasterization effects, but **not proof** that integer rounding alone caused the errors.

[Surface-coverage failure comparison](outputs/generalization_patch_coverage_comparison.png) · [Overhead centroid/depth diagnostic](outputs/generalization_overhead_error_components.png) · [Pixel-phase analysis](outputs/generalization_overhead_pixel_phase.png)

## Perception-driven robotic reaching

The frozen method R estimates the cube's top-face center in World coordinates from the simulator's RGB-D camera. A **0.10 m World-Z approach offset** is applied before **position-only IK** and **joint-space position control** on a simulated Franka Panda robot. The end-effector reference is the Panda `panda_grasptarget` link frame.

The robot receives **only perception-derived targets**. Simulator GT is read for the subsequent evaluation and to record cube displacement; it does not determine the commanded pose. Perception rejection or invalid IK prevents the controller from being called. Those rejection paths were verified in tests; their frequency under real corrupted reaching scenarios has not been measured.

### Single-scene demonstration

| Metric | Result |
|---|---:|
| Perception error | 0.620 mm |
| IK-FK position error | 1.890 mm |
| Control error (EE vs. commanded approach point) | 2.516 mm |
| Final reaching error (EE vs. GT approach point) | **2.182 mm** |
| Cube displacement | 0.016 mm |
| Controller status | Converged |
| Simulated time | 1.325 s (318 steps) |
| Automated tests (full repository) | 119 passed |

The three 3D error **vectors** satisfy the identity `e_reaching = e_perception + e_control`, while their Euclidean magnitudes do not generally add. The numerical results demonstrate **one successful clean-scene, open-loop perception-driven reach**, not a measured reaching success rate across different positions, cameras, or depth corruptions.

### Reaching artifacts

- [Integration report](reports/perception_driven_reaching_integration.md)
- [Measured end-effector trajectory](outputs/robot_reaching/single_reach_trajectory_3d.png)
- [Control error over simulated time](outputs/robot_reaching/single_reach_control_error.png)

## Repository structure

```text
rgbd-perception-reaching/
├── src/
│   ├── generate_rgbd_data.py                # Initial RGB-D renderer
│   ├── camera_geometry.py                   # Camera/World geometry
│   ├── object_perception.py                 # RGB object detection
│   ├── robust_localization.py               # Frozen method R
│   ├── depth_corruption.py                  # Synthetic noise/dropout
│   ├── generate_generalization_scenes.py    # Position/viewpoint data
│   ├── evaluate_generalization_robustness.py
│   ├── robot_scene.py                       # Panda and RGB-D scene
│   ├── robot_frames.py                      # Base/World target transforms
│   ├── robot_kinematics.py                  # FK and IK validation
│   ├── reaching_controller.py               # Joint-space position control
│   ├── run_perception_reach.py              # Perception → IK → motors
│   └── run_single_reach.py                  # Audit, plots, report
├── tests/                                    # Localization & reaching tests
├── reports/
│   ├── rgbd_localization_error_analysis.md
│   ├── depth_robustness_evaluation.md
│   ├── localization_generalization_evaluation.md
│   └── perception_driven_reaching_integration.md
├── outputs/
│   └── robot_reaching/                      # Single-scene robot artifacts
├── pyproject.toml
└── README.md
```

This is a **selected map**, not an exhaustive list of evaluation scripts. Original development scene arrays are generated in `outputs/trials/`; the 132-scene benchmark manifest is generated under `outputs/generalization/position_viewpoint_benchmark/`.

## Reproduction

### Prerequisites

- Python **3.11** (as required by `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) for dependency/environment management
- Python dependencies declared in `pyproject.toml` (including PyBullet, NumPy, Matplotlib, and pytest)

```bash
git clone https://github.com/SangzunPark/rgbd-perception-reaching.git
cd rgbd-perception-reaching
uv sync
```

Run commands from the **repository root**, in the order shown below. Rendering is CPU-based and does not require a graphical PyBullet window.

### A. Initial RGB-D pipeline and development scenes

```bash
# Initial single scene (optional).
uv run python -m src.generate_rgbd_data
# Nine development scenes and their localization metrics.
uv run python -m src.generate_trials
uv run python -m src.evaluate_trials
```

### B. Error diagnosis and ablation

```bash
uv run python -m src.diagnose_projection
uv run python -m src.run_ablation
uv run python -m src.analyze_ablation
```

For the full range of camera-alignment and surface-geometry checks, see the diagnostic scripts and [error-analysis report](reports/rgbd_localization_error_analysis.md).

### C. Robustness benchmark

```bash
uv run python -m src.evaluate_gaussian_robustness
uv run python -m src.evaluate_missing_depth
uv run python -m src.evaluate_robust_methods
uv run python -m src.visualize_robustness
```

### D. Held-out generalization and diagnostics

**Important:** The original nine development scenes must exist before this stage; their overhead versions are **reused, not regenerated**.

```bash
# Add 123 new scenes; assemble a manifest referencing 132 total scenes.
uv run python -m src.generate_generalization_scenes
# Clean position and camera-viewpoint evaluation.
uv run python -m src.evaluate_position_generalization
uv run python -m src.evaluate_viewpoint_generalization
# Depth corruption transfer across held-out position/view combinations.
uv run python -m src.evaluate_generalization_robustness
# Generate failure-analysis plots and diagnostics.
uv run python -m src.analyze_generalization
uv run python -m src.diagnose_surface_coverage
uv run python -m src.diagnose_pixel_grid_bias
```

### E. Panda simulation and reaching demonstration

```bash
uv run python -m src.robot_scene
uv run python -m src.robot_frames
uv run python -m src.robot_kinematics
uv run python -m src.reaching_controller
uv run python -m src.run_perception_reach
uv run python -m src.run_single_reach
```

The last two commands each run a single perception-driven reaching simulation; `src.run_single_reach` additionally saves plots and a Markdown report. No GPU or graphical PyBullet window is required.

### F. Automated tests

```bash
uv run python -m pytest -q
```

**119 tests passed** in the completed local robot-integration run. Tests cover camera geometry, missing-depth handling, paired comparisons, localization outcomes, robot frames, FK/IK, joint control, rejection gates, and single-scene audit invariants.

### Generated artifacts

- **Scene arrays:** RGB images, metric `depth_m.npy`, view matrices, GT top-face positions, and diagnostic segmentation images.
- **Development benchmarks:** `outputs/localization_ablation.csv`, `outputs/robust_localization_benchmark.csv`, and associated summary/plot files.
- **Generalization benchmarks:** `outputs/position_generalization_results.csv`, `outputs/viewpoint_generalization_results.csv`, `outputs/viewpoint_paired_deltas.csv`, and `outputs/generalization_robustness_results.csv`.
- **Diagnostic outputs:** geometry, surface coverage, and pixel-grid CSVs, alongside PNG figures.
- **Robot-reaching artifacts:** `outputs/robot_reaching/perception_reach_summary.json`, controller trajectory CSV, scene images, and 3D/error plots.

Raw generated scenes under `outputs/trials/` and `outputs/generalization/`, as well as robot-reaching sensor `.npy` arrays, may be excluded from Git because the scripts regenerate them. The **source code, tests, reports, and selected result files** make the results auditable without checking in every rendered depth array.

## Reports

- [RGB-D Localization Error Analysis](reports/rgbd_localization_error_analysis.md) — systematic offsets, visible-side-face bias, and initial ablation study.
- [Depth Robustness Evaluation](reports/depth_robustness_evaluation.md) — Gaussian noise, random/patch dropout, and method R development.
- [Generalization and Failure Analysis](reports/localization_generalization_evaluation.md) — held-out positions, viewpoint changes, corruption transfer, coverage failures, and centroid error.
- [Perception-Driven Reaching Integration](reports/perception_driven_reaching_integration.md) — Panda IK/control, one scene's reaching errors, trajectory, and limitations.

## Limitations and next steps

This is a **simulation-only** study using one red cube, controlled lighting, fixed image resolution, and nearby camera viewpoints. Localization was evaluated on 132 clean scenes and repeated corrupted inputs, but **robot reaching was demonstrated on only one clean overhead scene**. The observed 2.182 mm reaching error therefore cannot be interpreted as a general reaching accuracy or success rate.

The `+1` pixel correction and confidence thresholds are renderer-specific heuristics. Under severe dropout, method R sometimes returns inaccurate positions rather than rejecting them. The integrated rejection tests verify software gating, not safety or confidence calibration under physical sensing conditions. Joint-space interpolation does not guarantee a collision-free Cartesian path, and this work does not establish grasping ability, visual-servoing performance, real-sensor transfer, or safety on a physical robot.

A possible extension is **paired multi-scene reaching evaluation** using the same A/G/R perception inputs, Panda IK, and controller, including synthetic depth corruption and explicit failure classifications. Other future work includes geometry-aware surface support, calibrated uncertainty, different objects, and real-sensor validation. These extensions are **not required to reproduce the completed single-scene proof of concept**.

---

*Research-oriented simulation implementation; no claim of novel state-of-the-art performance or physical-robot validation.*

