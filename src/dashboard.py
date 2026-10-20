\
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_executive_summary(metrics_path="reports/tables/forecast_model_metrics.csv",
                            uncertainty_path="reports/tables/forecast_uncertainty_summary.csv",
                            output_path="reports/executive_summary.txt"):
    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        raise FileNotFoundError("Model metrics are missing. Run model training and evaluation first.")
    metrics = pd.read_csv(metrics_file)
    validation = metrics[metrics["partition"] == "validation"].copy()
    test = metrics[metrics["partition"] == "test"].copy()
    best_rows = validation.sort_values("MAE").groupby("horizon_steps", as_index=False).first()
    lines = [
        "EXECUTIVE SUMMARY — SHORT-TERM FERRY TICKET DEMAND FORECASTING",
        "=" * 68,
        "",
        "Purpose",
        "Develop a short-term forecasting workflow to support ferry ticket demand awareness and operational planning.",
        "",
        "Evaluation approach",
        "Models are selected using chronological validation MAE. Test metrics are reported as held-out performance.",
        "",
        "Best models by horizon (selected using validation MAE):",
    ]
    for _, row in best_rows.iterrows():
        lines.append(
            f"- {int(row['horizon_minutes'])}-minute horizon: {row['model']} "
            f"(validation MAE={row['MAE']:.3f}, validation RMSE={row['RMSE']:.3f})"
        )
    lines += ["", "Held-out test metrics for the validation-selected models:"]
    for _, row in best_rows.iterrows():
        match = test[(test["horizon_steps"] == row["horizon_steps"]) & (test["model"] == row["model"])]
        if not match.empty:
            r = match.iloc[0]
            lines.append(
                f"- {int(r['horizon_minutes'])}-minute horizon: MAE={r['MAE']:.3f}, "
                f"RMSE={r['RMSE']:.3f}, n={int(r['n'])}"
            )
    lines += [
        "",
        "Operational interpretation",
        "Forecasts may help staff anticipate short-term demand changes, but they should be used as decision support, not as an automatic dispatch instruction.",
        "",
        "Limitations",
        "- Forecast quality depends on data completeness, representative operating periods, and stable timestamp intervals.",
        "- Historical sales and redemptions may not directly equal passenger arrivals or vessel occupancy.",
        "- Weather, holidays, service disruptions, special events, and ferry capacity may affect demand but may not be present in the dataset.",
        "- Prediction intervals are approximate and require monitoring before operational use.",
        "",
        "This summary is generated from model output files; it should not be used before running the pipeline on the official dataset."
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\\n".join(lines), encoding="utf-8")
    return out


def plot_model_comparison(metrics_path="reports/tables/forecast_model_metrics.csv",
                          output_path="reports/figures/model_comparison_validation_mae.png"):
    metrics = pd.read_csv(metrics_path)
    data = metrics[metrics["partition"] == "validation"].copy()
    pivot = data.pivot(index="horizon_minutes", columns="model", values="MAE")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_title("Validation MAE by Model and Forecast Horizon")
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (lower is better)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out
