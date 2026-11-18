from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "COLDataset"
    / "COLDataset"
)

for split in ["train", "dev", "test"]:
    file_path = DATA_DIR / f"{split}.csv"

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    print("=" * 50)
    print(f"Dataset: {split}")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nLabel percentage:")
    print(df["label"].value_counts(normalize=True))

    print("\nMissing values:")
    print(df[["TEXT", "label"]].isnull().sum())

    print("\nExamples:")
    print(df[["TEXT", "label"]].head(3).to_string(index=False), '\n\n')