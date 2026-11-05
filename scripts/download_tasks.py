"""Download Impossible-LiveCodeBench into data/ (task files are not redistributed here)."""
from datasets import load_dataset

ds = load_dataset("fjzzq2002/impossible_livecodebench")
for split in ["original", "oneoff", "conflicting"]:
    ds[split].to_parquet(f"data/impossible_lcb_{split}.parquet")
print(ds)
