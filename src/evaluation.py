\
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prediction_interval_from_validation_residuals(y_validation, validation_prediction, y_test, test_prediction,
                                                   coverage=0.90):
    """
    Approximate prediction intervals using the empirical absolute validation residual quantile.
    This is a residual-based interval, not a guarantee of calibrated coverage.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1.")
    residuals = np.asarray(y_validation, dtype=float) - np.asarray(validation_prediction, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if len(finite) < 5:
        raise ValueError("At least five finite validation residuals are needed for an interval estimate.")
    q = float(np.quantile(np.abs(finite), coverage))
    pred = np.asarray(test_prediction, dtype=float)
    return pred - q, pred + q, q


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_percent_excluding_zero_actuals": np.nan,
                "zero_actual_count": 0, "n": 0}
    error = y_true - y_pred
    nonzero = np.abs(y_true) > 0
    mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_percent_excluding_zero_actuals": mape,
        "zero_actual_count": int((~nonzero).sum()),
        "n": int(len(y_true))
    }


def evaluate_horizon_results(result_by_horizon, output_dir="reports"):
    output = Path(output_dir)
    reports, figures = output / "tables", output / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    uncertainty_rows = []
    for h, result in result_by_horizon.items():
        for model_name, pred_dict in result["predictions"].items():
            try:
                lower, upper, q = prediction_interval_from_validation_residuals(
                    result["y_validation"], pred_dict["validation"], result["y_test"], pred_dict["test"], coverage=.90
                )
                actual = result["y_test"]
                coverage = float(np.mean((actual >= lower) & (actual <= upper)))
                uncertainty_rows.append({
                    "horizon_steps": h, "horizon_minutes": h*15, "model": model_name,
                    "nominal_interval_coverage": 0.90, "observed_test_coverage": coverage,
                    "mean_interval_width": float(np.mean(upper-lower)),
                    "validation_absolute_residual_quantile": q,
                    "test_rows": len(actual)
                })
                pred_df = pd.DataFrame({
                    "timestamp": result["test_timestamps"], "actual": actual,
                    "prediction": pred_dict["test"], "lower_90": lower, "upper_90": upper
                })
                safe_name = "".join(c.lower() if c.isalnum() else "_" for c in model_name).strip("_")
                pred_df.to_csv(reports / f"uncertainty_{h*15}min_{safe_name}.csv", index=False)
            except ValueError:
                continue

        # Plot validation-selected best model on the test period for this horizon.
        val_rows = result["metrics"].query("partition == 'validation'").sort_values("MAE")
        best = str(val_rows.iloc[0]["model"])
        pred = result["predictions"][best]["test"]
        plt.figure(figsize=(12, 4.5))
        plt.plot(result["test_timestamps"], result["test_actual"], label="Actual", linewidth=1.2)
        plt.plot(result["test_timestamps"], pred, label=f"Prediction — {best}", linewidth=1.1)
        plt.title(f"Test Forecast: {h*15}-Minute Horizon")
        plt.xlabel("Timestamp"); plt.ylabel("Ticket count"); plt.legend(); plt.tight_layout()
        plt.savefig(figures / f"test_forecast_{h*15}min.png", dpi=150)
        plt.close()

    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(reports / "forecast_uncertainty_summary.csv", index=False)
    metrics_path = reports / "forecast_model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        # Compare models by horizon using validation metrics only for selection.
        validation = metrics[metrics["partition"] == "validation"].copy()
        if not validation.empty:
            best = validation.sort_values(["horizon_steps", "MAE"]).groupby("horizon_steps", as_index=False).first()
            best.to_csv(reports / "best_model_by_horizon_validation.csv", index=False)
    notes = {
        "selection_rule": "Select the model with lowest validation MAE at each horizon; report test metrics only after selection.",
        "mape_handling": "MAPE excludes zero-valued actual observations; zero counts are reported separately.",
        "uncertainty_method": "Empirical absolute validation-residual quantile around test predictions; observed test coverage is diagnostic only.",
        "caution": "Residual intervals are approximate and may be miscalibrated under changing demand, seasonality, or distribution shift."
    }
    (reports / "evaluation_methodology.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return uncertainty
