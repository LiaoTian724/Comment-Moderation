
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "COLDataset" / "COLDataset"
SPLITS = ["train", "dev", "test"]


def load_split(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv", encoding="utf-8-sig")
    df["TEXT"] = df["TEXT"].fillna("").astype(str).str.strip()
    return df


datasets = {s: load_split(s) for s in SPLITS}

for split, df in datasets.items():
    print("\n" + "=" * 60)
    print(f"Dataset: {split}")
    print("\nTopic distribution:")
    print(pd.crosstab(df["topic"], df["label"], margins=True))
    print("\nTopic percentage:")
    print(pd.crosstab(df["topic"], df["label"], normalize="index").round(3))



all_data = pd.concat(
    (df[["TEXT", "label"]].assign(source_split=split) for split, df in datasets.items()),
    ignore_index=True,
)
dup_counts = all_data.groupby("TEXT")["source_split"].nunique()
cross_split_duplicates = dup_counts[dup_counts > 1]

print("\n" + "=" * 60)
print("Cross-split duplicate analysis")
print("重复文本数量:", len(cross_split_duplicates))


if len(cross_split_duplicates) > 0:
    print("\n重复样本: ")
    examples = all_data[all_data["TEXT"].isin(cross_split_duplicates.index)]
    print(examples.head(10).to_string(index=False))


# 3. 抽样查看训练集真实文本
train_df = datasets["train"]
print("\n" + "=" * 60)
print("Training examples")

for label in [0, 1]:
    subset = train_df[train_df["label"] == label]
    print(f"\nLabel: {label}")
    samples = subset.sample(n=min(10, len(subset)), random_state=42)
    for text in samples["TEXT"]:
        print("-", text)



test_df = datasets["test"]
if "fine-grained-label" in test_df.columns:
    print("\nFine-grained label distribution:")
    print(test_df["fine-grained-label"].value_counts().sort_index())
