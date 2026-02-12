import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train.py"

MAX_DEPTHS = 10
N_ESTIMATORS = [2, 5, 10, 20, 50, 70, 100, 200]


def main():
    for i, n_estimators in enumerate(N_ESTIMATORS, 1):
        print(f"\n[{i}/{len(N_ESTIMATORS)}] Running XGBoost max_depth={MAX_DEPTHS} n_estimators={N_ESTIMATORS}")
        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--model", "xgboost",
                "--max-depth", str(MAX_DEPTHS),
                "--n-estimators", str(n_estimators),
            ],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"Failed: n_estimators={n_estimators}")
            sys.exit(result.returncode)

    print(f"\nDone. Ran {len(N_ESTIMATORS)} models.")


if __name__ == "__main__":
    main()
