from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .data_validation import resolve_columns

def run_eda(df, output_dir="outputs"):
    output = Path(output_dir)
    figures, reports = output/"figures", output/"tables"
    figures.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    cols = resolve_columns(data)
    if "timestamp" not in cols:
        raise ValueError("EDA requires a recognizable timestamp column.")
    tc = cols["timestamp"]
    data[tc] = pd.to_datetime(data[tc], errors="coerce")
    data = data.dropna(subset=[tc]).sort_values(tc)
    measures = [cols[k] for k in ["sales_count", "redemption_count"] if k in cols]
    if not measures:
        raise ValueError("EDA needs a recognized ticket-count measure.")
    for c in measures: data[c] = pd.to_numeric(data[c], errors="coerce")
    stats = data[measures].describe().T
    stats["median"] = data[measures].median()
    stats["missing"] = data[measures].isna().sum()
    stats["zero_count"] = (data[measures] == 0).sum()
    stats.to_csv(reports/"descriptive_statistics.csv")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,5))
    for c in measures: plt.plot(data[tc], data[c], label=c, linewidth=.9)
    plt.title("Ferry Ticket Activity Over Time"); plt.xlabel("Timestamp"); plt.ylabel("Ticket count"); plt.legend()
    plt.tight_layout(); plt.savefig(figures/"ticket_activity_over_time.png", dpi=150); plt.close()
    data["__hour"] = data[tc].dt.hour
    for c in measures:
        plt.figure(figsize=(8,4))
        sns.histplot(data[c].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {c}"); plt.tight_layout()
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in c).strip("_")
        plt.savefig(figures/f"distribution_{safe}.png", dpi=150); plt.close()
        hourly = data.groupby("__hour")[c].mean().reset_index()
        plt.figure(figsize=(8,4)); sns.lineplot(data=hourly, x="__hour", y=c, marker="o")
        plt.title(f"Average {c} by Hour"); plt.xlabel("Hour of day"); plt.tight_layout()
        plt.savefig(figures/f"hourly_pattern_{safe}.png", dpi=150); plt.close()
    corr = data[measures].corr(numeric_only=True)
    if len(measures) > 1:
        plt.figure(figsize=(6,4)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1)
        plt.title("Ticket Measures Correlation"); plt.tight_layout()
        plt.savefig(figures/"measure_correlation.png", dpi=150); plt.close()
        plt.figure(figsize=(7,4)); sns.scatterplot(data=data, x=measures[0], y=measures[1], alpha=.5)
        plt.title(f"{measures[0]} vs {measures[1]}"); plt.tight_layout()
        plt.savefig(figures/"sales_vs_redemptions.png", dpi=150); plt.close()
    summary = [
        "EXPLORATORY DATA ANALYSIS SUMMARY", "="*38,
        f"Rows used: {len(data)}", f"Period: {data[tc].min()} to {data[tc].max()}",
        f"Measures: {measures}", "", stats.to_string(), "", "Correlation matrix:", corr.to_string(),
        "", "Interpretation: inspect trend, recurring patterns, outliers and sales/redemption relationships.",
        "EDA is descriptive, not causal. Forecasting validation must preserve chronological order."
    ]
    (reports/"eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    return {"statistics": stats, "correlation": corr, "figures_dir": str(figures), "reports_dir": str(reports)}
