import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_DEPTH = 10
N_ESTIMATORS = [2, 5, 10, 20, 50, 70, 100, 200]


def main():
    print("1. Підготовка даних...")
    result = subprocess.run(
        [sys.executable, "-m", "src.prepare"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("Помилка prepare.")
        sys.exit(result.returncode)

    print("\n2. Навчання моделей...")
    for i, n_estimators in enumerate(N_ESTIMATORS, 1):
        print(f"\n[{i}/{len(N_ESTIMATORS)}] XGBoost n_estimators={n_estimators} max_depth={MAX_DEPTH}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.train",
                "--model", "xgboost",
                "--max-depth", str(MAX_DEPTH),
                "--n-estimators", str(n_estimators),
            ],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"Помилка: n_estimators={n_estimators}")
            sys.exit(result.returncode)

    print(f"\nГотово. Запущено {len(N_ESTIMATORS)} моделей.")


if __name__ == "__main__":
    main()
