import pandas as pd
import numpy as np

def chronological_split(feature_df, train_size=0.70, validation_size=0.15, test_size=0.15):
    """
    Split model-ready rows chronologically. No random shuffle.
    The split percentages must sum to 1.0.
    """
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0.")
    if not (0 < train_size < 1 and 0 < validation_size < 1 and 0 < test_size < 1):
        raise ValueError("Each split proportion must be between 0 and 1.")
    if len(feature_df) < 20:
        raise ValueError("At least 20 model-ready rows are recommended for a chronological split.")
    data = feature_df.copy()
    if "timestamp" not in data:
        raise ValueError("feature_df must contain a timestamp column.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + validation_size))
    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One of the partitions is empty; provide more data or change split proportions.")
    summary = pd.DataFrame([
        {"partition": "train", "rows": len(train), "start": train.timestamp.min(), "end": train.timestamp.max()},
        {"partition": "validation", "rows": len(validation), "start": validation.timestamp.min(), "end": validation.timestamp.max()},
        {"partition": "test", "rows": len(test), "start": test.timestamp.min(), "end": test.timestamp.max()},
    ])
    return train, validation, test, summary

def rolling_origin_folds(feature_df, n_splits=5, min_train_fraction=0.5):
    """
    Expanding-window folds for time-series model comparison.
    Each validation block occurs strictly after its corresponding training block.
    Returns index pairs relative to the sorted feature dataframe.
    """
    data = feature_df.sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    initial = int(n * min_train_fraction)
    remaining = n - initial
    if n_splits < 1 or initial < 2 or remaining < n_splits:
        raise ValueError("Not enough rows for the requested rolling-origin folds.")
    block = remaining // n_splits
    folds = []
    for i in range(n_splits):
        train_end = initial + i * block
        val_end = initial + (i + 1) * block if i < n_splits - 1 else n
        if val_end <= train_end:
            continue
        folds.append({
            "fold": i + 1,
            "train_indices": list(range(0, train_end)),
            "validation_indices": list(range(train_end, val_end)),
            "train_start": data.timestamp.iloc[0],
            "train_end": data.timestamp.iloc[train_end - 1],
            "validation_start": data.timestamp.iloc[train_end],
            "validation_end": data.timestamp.iloc[val_end - 1],
        })
    return folds

def save_split_outputs(train, validation, test, summary, folds, output_dir="reports/tables"):
    from pathlib import Path
    import json
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out/"train.csv", index=False)
    validation.to_csv(out/"validation.csv", index=False)
    test.to_csv(out/"test.csv", index=False)
    summary.to_csv(out/"chronological_split_summary.csv", index=False)
    # Store fold ranges, not huge index arrays, for concise documentation.
    fold_summary = [{k: v for k, v in fold.items() if not k.endswith("_indices")} |
                    {"train_rows": len(fold["train_indices"]),
                     "validation_rows": len(fold["validation_indices"])} for fold in folds]
    (out/"rolling_origin_folds.json").write_text(json.dumps(fold_summary, indent=2, default=str), encoding="utf-8")
