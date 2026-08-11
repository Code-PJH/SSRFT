"""Reproduce the JailBench splits used in the paper.

The original JailBench CSV files use an unnamed ID column and the columns
``query``/``seed`` and ``二级领域``.  The evaluation code expects the schema
``q_id, question, 一级领域, content_policy_name``.  This script normalizes that
schema before applying the deterministic splits used in the experiments.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = ["q_id", "question", "一级领域", "content_policy_name"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "test_datasets" / "JailBench"


def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    """Load an original JailBench CSV and normalize its column names."""
    dataframe = pd.read_csv(path)

    rename_map = {
        "Unnamed: 0": "q_id",
        "query": "question",
        "seed": "question",
        "二级领域": "content_policy_name",
    }
    dataframe = dataframe.rename(columns=rename_map)

    if list(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {path}: {list(dataframe.columns)}. "
            f"Expected {EXPECTED_COLUMNS} after normalization."
        )

    required_columns = ["q_id", "question", "content_policy_name"]
    if dataframe[required_columns].isnull().any().any():
        raise ValueError(f"Missing required values in {path}.")
    if dataframe["q_id"].duplicated().any():
        raise ValueError(f"Duplicate q_id values in {path}.")

    return dataframe


def split_seed_dataset(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Create the train, ID-test, OOD-test, and combined test splits."""
    policy_names = dataframe["content_policy_name"].drop_duplicates()
    if len(policy_names) < 18:
        raise ValueError("JailBench-seed must contain at least 18 policy categories.")

    ood_policy_names = policy_names.sample(n=18, random_state=42)
    ood_test = dataframe[dataframe["content_policy_name"].isin(ood_policy_names)]
    remaining = dataframe[~dataframe["content_policy_name"].isin(ood_policy_names)]

    train, id_test = train_test_split(
        remaining,
        test_size=0.5,
        stratify=remaining["content_policy_name"],
        random_state=42,
    )
    test = pd.concat([id_test, ood_test])

    train.to_csv(output_dir / "train.csv", index=False)
    id_test.to_csv(output_dir / "id_test.csv", index=False)
    ood_test.to_csv(output_dir / "ood_test.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    print(f"Train shape: {train.shape}")
    print(f"ID test shape: {id_test.shape}")
    print(f"OOD test shape: {ood_test.shape}")
    print(f"Combined test shape: {test.shape}")


def sample_full_dataset(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Create the stratified 1,080-example JailBench evaluation sample."""
    sample_size = 1080
    if len(dataframe) < sample_size:
        raise ValueError(
            f"JailBench.csv contains {len(dataframe)} rows; "
            f"at least {sample_size} are required."
        )

    sample, _ = train_test_split(
        dataframe,
        train_size=sample_size,
        stratify=dataframe["content_policy_name"],
        random_state=42,
    )
    output_path = output_dir / "JailBench_sampled_1080.csv"
    sample.to_csv(output_path, index=False)
    print(f"JailBench sample shape: {sample.shape}")
    print(f"Saved sampled JailBench data to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the original JailBench CSVs and reproduce the paper splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing JailBench-seed.csv and JailBench.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --data-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dataframe = load_and_normalize_csv(data_dir / "JailBench-seed.csv")
    full_dataframe = load_and_normalize_csv(data_dir / "JailBench.csv")

    # The evaluation runners consume JailBench-seed.csv directly, so persist
    # the normalized schema as well as the derived split files. Both inputs are
    # loaded before either file is overwritten when data_dir == output_dir.
    seed_dataframe.to_csv(output_dir / "JailBench-seed.csv", index=False)
    full_dataframe.to_csv(output_dir / "JailBench.csv", index=False)

    split_seed_dataset(seed_dataframe, output_dir)
    sample_full_dataset(full_dataframe, output_dir)


if __name__ == "__main__":
    main()
