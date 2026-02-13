from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"


def _time_to_hour(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X)
    return np.array((arr.ravel() // 3600) % 24, dtype=np.int64)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = df.dropna()
    return df


def prepare_and_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Engineering: Time → hour_of_day
    if "Time" in feature_cols:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["hour_of_day"] = _time_to_hour(X_train["Time"].values)
        X_test["hour_of_day"] = _time_to_hour(X_test["Time"].values)
        X_train = X_train.drop(columns=["Time"])
        X_test = X_test.drop(columns=["Time"])

    # Amount → RobustScaler (fit на train)
    amount_col = "Amount"
    if amount_col in X_train.columns:
        scaler = RobustScaler()
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["Amount_scaled"] = scaler.fit_transform(X_train[[amount_col]])
        X_test["Amount_scaled"] = scaler.transform(X_test[[amount_col]])
        X_train = X_train.drop(columns=[amount_col])
        X_test = X_test.drop(columns=[amount_col])

    train_df = X_train.assign(Class=y_train.values)
    test_df = X_test.assign(Class=y_test.values)
    return train_df, test_df


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Шлях до data/raw/creditcard.csv",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Директорія для data/prepared/",
)
@click.option("--test-size", default=0.2, help="Частка test (0.2 = 20%%)")
@click.option("--random-state", default=42, help="Random state")
def main(
    input_path: Path | None,
    output_dir: Path | None,
    test_size: float,
    random_state: int,
):
    input_path = input_path or DEFAULT_RAW
    output_dir = output_dir or DEFAULT_PREPARED_DIR

    if not input_path.exists():
        raise FileNotFoundError(
            f"Файл {input_path} не знайдено. Запустіть 'dvc pull' для завантаження даних."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = clean_data(df)
    train_df, test_df = prepare_and_split(
        df, test_size=test_size, random_state=random_state
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Підготовлено: {len(train_df)} train, {len(test_df)} test")
    print(f"Збережено: {train_path}, {test_path}")


if __name__ == "__main__":
    main()
