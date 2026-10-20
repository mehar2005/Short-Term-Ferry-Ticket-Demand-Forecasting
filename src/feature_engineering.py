import pandas as pd
import numpy as np
from .data_validation import resolve_columns

DEFAULT_LAGS = (1, 2, 4, 8)
DEFAULT_WINDOWS = (4, 8)

def create_forecasting_features(df, target="sales_count", lags=DEFAULT_LAGS, windows=DEFAULT_WINDOWS,
                                include_other_measure=True):
    """
    Create leak-safe forecasting features for one target.
    One row represents one 15-minute interval. Lag 1 is the prior interval,
    so it is usable only if the forecast is issued after that interval is observed.
    """
    data = df.copy()
    cols = resolve_columns(data)
    if "timestamp" not in cols:
        raise ValueError("A timestamp column is required.")
    if target not in cols:
        raise ValueError(f"Target '{target}' was not detected. Available fields: {list(cols)}")
    tc, target_col = cols["timestamp"], cols[target]
    data[tc] = pd.to_datetime(data[tc], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[tc]).sort_values(tc).drop_duplicates(subset=[tc], keep="last").reset_index(drop=True)

    # Keep source timestamp and target value, then derive features from historical observations only.
    out = pd.DataFrame({"timestamp": data[tc], "target": data[target_col]})
    for lag in lags:
        out[f"{target}_lag_{lag}"] = data[target_col].shift(lag)
    historical = data[target_col].shift(1)
    for window in windows:
        roll = historical.rolling(window=window, min_periods=window)
        out[f"{target}_rolling_mean_{window}"] = roll.mean()
        out[f"{target}_rolling_std_{window}"] = roll.std()
        out[f"{target}_rolling_max_{window}"] = roll.max()

    # Time features are known at forecast creation time.
    ts = data[tc]
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["quarter"] = ts.dt.quarter

    other_key = "redemption_count" if target == "sales_count" else "sales_count"
    if include_other_measure and other_key in cols:
        other_col = cols[other_key]
        data[other_col] = pd.to_numeric(data[other_col], errors="coerce")
        for lag in lags:
            out[f"{other_key}_lag_{lag}"] = data[other_col].shift(lag)

    # Enforce a complete 15-minute grid for trustworthy interval lags.
    # Do not silently interpolate; mark irregular intervals so downstream validation can inspect them.
    delta = out["timestamp"].diff()
    out["interval_is_15min"] = delta.eq(pd.Timedelta(minutes=15)).astype(int)
    out.loc[0, "interval_is_15min"] = 0
    feature_cols = [c for c in out.columns if c not in ["timestamp", "target"]]
    out = out.replace([np.inf, -np.inf], np.nan)
    model_ready = out.dropna(subset=["target"] + feature_cols).copy()
    return model_ready.reset_index(drop=True), {
        "target_requested": target,
        "target_column": target_col,
        "timestamp_column": tc,
        "input_rows": int(len(df)),
        "rows_after_timestamp_sort_dedup": int(len(data)),
        "model_ready_rows": int(len(model_ready)),
        "features": feature_cols,
        "lags": list(lags),
        "rolling_windows": list(windows),
        "note": "Rows with incomplete lag/rolling history are removed. No target imputation is performed."
    }

def save_feature_outputs(feature_df, metadata, output_dir="reports/tables"):
    from pathlib import Path
    import json
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    target = metadata["target_requested"]
    feature_df.to_csv(out / f"features_{target}.csv", index=False)
    (out / f"feature_engineering_{target}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
