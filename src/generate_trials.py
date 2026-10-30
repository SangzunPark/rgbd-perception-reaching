
"""Generate RGB-D datasets for multiple cube positions."""

from pathlib import Path
import subprocess
import sys


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

SCENE_SCRIPT = ROOT_DIR / "src" / "generate_rgbd_data.py"

OUTPUT_ROOT = ROOT_DIR / "outputs" / "trials"


# 2. Experimental positions (World X, World Y)
POSITIONS = [
    (-0.10,  0.05),  # Trial 01
    ( 0.00,  0.05),  # Trial 02
    ( 0.10,  0.05),  # Trial 03

    (-0.10,  0.00),  # Trial 04
    ( 0.00,  0.00),  # Trial 05
    ( 0.10,  0.00),  # Trial 06

    (-0.10, -0.05),  # Trial 07
    ( 0.00, -0.05),  # Trial 08
    ( 0.10, -0.05),  # Trial 09
]


def main():
    """Generate one RGB-D scene for each position."""

    for index, (x, y) in enumerate(POSITIONS, start=1):

        # 3. Create a unique output directory
        trial_name = f"trial_{index:02d}"
        trial_dir = OUTPUT_ROOT / trial_name

        # 4. Build command
        command = [
            sys.executable,
            str(SCENE_SCRIPT),
            "--x", str(x),
            "--y", str(y),
            "--output-dir", str(trial_dir),
        ]

        # 5. Execute scene generator
        print(
            f"\nGenerating {trial_name}: "
            f"x={x:.2f}, y={y:.2f}",
            flush=True,
        )

        subprocess.run(
            command,
            check=True,
            cwd=ROOT_DIR,
        )

    print(
        f"\nCompleted: {len(POSITIONS)} trials generated."
    )


if __name__ == "__main__":
    main()
