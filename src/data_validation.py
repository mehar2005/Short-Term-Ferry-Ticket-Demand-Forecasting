from pathlib import Path
import pandas as pd

EXPECTED_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "date time", "time", "interval_end"],
    "sales_count": ["sales count", "sales_count", "sales", "tickets sold", "ticket_sales"],
    "redemption_count": ["redemption count", "redemption_count", "redemptions", "tickets redeemed", "redeemed"],
    "row_id": ["_id", "id", "row_id", "record_id"]
}

def normalize_name(name):
    return " ".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())

def resolve_columns(df):
    normalized = {normalize_name(c): c for c in df.columns}
    resolved = {}
    for canonical, aliases in EXPECTED_ALIASES.items():
        for alias in aliases:
            key = normalize_name(alias)
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    return resolved

def load_dataset(source):
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place the official CSV there or upload it in the app.")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def validate_dataset(df):
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")
    result = df.copy()
    columns = resolve_columns(result)
    issues, warnings = [], []
    if "timestamp" not in columns:
        issues.append("Required timestamp column was not detected.")
    for field in ["sales_count", "redemption_count"]:
        if field not in columns:
            issues.append(f"Expected measure column was not detected: {field.replace('_', ' ').title()}.")
    report = {
        "rows": int(result.shape[0]), "columns": int(result.shape[1]),
        "column_names": list(map(str, result.columns)),
        "duplicate_rows": int(result.duplicated().sum()),
        "missing_cells": int(result.isna().sum().sum()),
        "columns_with_missing_values": {str(k): int(v) for k, v in result.isna().sum().items() if int(v) > 0},
        "resolved_columns": columns, "issues": issues, "warnings": warnings
    }
    if "timestamp" in columns:
        tc = columns["timestamp"]
        result[tc] = pd.to_datetime(result[tc], errors="coerce")
        valid = result[tc].dropna()
        report["invalid_timestamps"] = int(result[tc].isna().sum())
        report["duplicate_timestamps"] = int(result[tc].duplicated().sum())
        report["timestamp_start"] = str(valid.min()) if len(valid) else None
        report["timestamp_end"] = str(valid.max()) if len(valid) else None
        report["out_of_order_rows_before_sort"] = not valid.is_monotonic_increasing
        gaps = []
        if len(valid.drop_duplicates()) > 1:
            ordered = valid.drop_duplicates().sort_values()
            for i in range(1, len(ordered)):
                prev, curr = ordered.iloc[i-1], ordered.iloc[i]
                delta = curr - prev
                if delta > pd.Timedelta(minutes=15):
                    gaps.append({"previous_timestamp": prev, "next_timestamp": curr,
                                 "gap_minutes": delta.total_seconds()/60,
                                 "missing_15_minute_intervals": max(int(delta / pd.Timedelta(minutes=15))-1, 0)})
        report["gap_rows"] = gaps
        report["gap_events"] = len(gaps)
        report["estimated_missing_15_minute_intervals"] = sum(x["missing_15_minute_intervals"] for x in gaps)
    for field in ["sales_count", "redemption_count"]:
        if field in columns:
            col = columns[field]
            vals = pd.to_numeric(result[col], errors="coerce")
            report[f"{field}_non_numeric_or_missing"] = int(vals.isna().sum())
            report[f"{field}_negative_values"] = int((vals < 0).sum())
            report[f"{field}_zero_values"] = int((vals == 0).sum())
            if (vals < 0).any():
                warnings.append(f"{col} contains negative values; verify whether these represent corrections.")
            result[col] = vals
    report["issues"], report["warnings"] = issues, warnings
    return result, report

def write_validation_outputs(report, output_dir="reports/tables"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["DATA QUALITY REPORT", "="*50]
    for key, value in report.items():
        if key != "gap_rows":
            lines.append(f"{key}: {value}")
    (out / "data_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(report.get("gap_rows", [])).to_csv(out / "interval_gaps.csv", index=False)
