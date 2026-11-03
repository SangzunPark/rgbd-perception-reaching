# Perception-Driven Reaching: Single-Scene Integration

## Research scope

One simulated 5 cm red cube, one fixed overhead RGB-D camera,
and one Panda robot in PyBullet.

The frozen Method R estimates the cube's top-face center.
Its World-frame output plus a 0.100 m
vertical approach offset is passed to position-only IK and
joint-space position control.

Simulator GT is used only for evaluation, not for target commands.

## Recorded result

| Quantity | Value |
|---|---:|
| Pipeline status | `control_converged` |
| Motor command issued | `True` |
| Perception error | 0.620 mm |
| IK-FK position error | 1.890 mm |
| Control error | 2.516 mm |
| Final reaching error | 2.182 mm |
| Cube displacement | 0.016 mm |
| Simulated steps | 318 |
| Simulated time | 1.325 s |
| Vector error identity residual | 0.000000000 mm |

## Geometry and interpretation

- The commanded approach position is derived from Method R, not GT.
- Perception error compares the estimated top-face center with
  the pre-motion simulator GT top-face center.
- Control error measures the final EE relative to the commanded
  approach. It includes IK approximation and motor tracking error.
- Final reaching error compares EE with the GT approach,
  using the same approach-height offset.
- Vector errors obey `e_reaching = e_control + e_perception`.
  Their Euclidean lengths need not add.
- The EE trajectory is produced through joint-space position control
  and is not guaranteed to be a Cartesian straight line.

## Outputs

- `outputs/robot_reaching/perception_reach_before.png`
- `outputs/robot_reaching/perception_reach_after.png`
- `outputs/robot_reaching/perception_reach_summary.json`
- `outputs/robot_reaching/perception_reach_trajectory.csv`
- `outputs/robot_reaching/single_reach_trajectory_3d.png`
- `outputs/robot_reaching/single_reach_control_error.png`

## Reproducibility

From the repository root:

```bash
uv run python -m src.run_single_reach
uv run python -m pytest -q
```

## Limitations

This is a single-scene, simulation-only, open-loop perception
demonstration.

It does not establish performance across unseen scenes, safe motion,
collision-free trajectories, real robot transfer, grasping, or
closed-loop visual servoing.

The 10 mm controller threshold is a provisional implementation
criterion, not a robotic safety standard or final task success
threshold.

Multi-scene A/G/R comparisons and robustness transfer remain
future evaluation work.
