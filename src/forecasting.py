\
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .temporal_split import chronological_split


def build_horizon_dataset(feature_df, horizon_steps=1):
    """
    Align current-time features with a future target.
    horizon_steps=1 means predict the target one 15-minute interval ahead.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")
    data = feature_df.sort_values("timestamp").reset_index(drop=True).copy()
    if "target" not in data:
        raise ValueError("Feature data must contain a 'target' column.")
    data["future_target"] = data["target"].shift(-horizon_steps)
    # The present target is never supplied as a feature; historical lags are.
    excluded = {"timestamp", "target", "future_target"}
    feature_cols = [c for c in data.columns if c not in excluded]
    model_data = data.dropna(subset=feature_cols + ["future_target"]).copy()
    X = model_data[feature_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(model_data["future_target"], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    model_data = model_data.loc[valid].copy()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    return model_data, X, y, feature_cols


def get_models(random_state=42):
    return {
        "Naive (last observed value)": DummyRegressor(strategy="constant", constant=0),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, loss="squared_error",
            random_state=random_state
        ),
    }


def fit_forecasting_models(feature_df, horizon_steps=1, random_state=42):
    """
    Train baseline and ML regressors using chronological train/validation/test partitions.
    The baseline is the last observed target value, supplied through the most recent target lag.
    """
    model_data, X, y, feature_cols = build_horizon_dataset(feature_df, horizon_steps)
    # Keep target and timestamp aligned with the future labels.
    train_end = int(len(model_data) * 0.70)
    val_end = int(len(model_data) * 0.85)
    if train_end < 10 or val_end - train_end < 2 or len(model_data) - val_end < 2:
        raise ValueError(
            f"Insufficient rows after horizon alignment ({len(model_data)}). "
            "Use more data or a shorter horizon."
        )
    train_idx = model_data.index[:train_end]
    val_idx = model_data.index[train_end:val_end]
    test_idx = model_data.index[val_end:]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    predictions = {}
    fitted_models = {}
    metrics = []

    # Persistence baseline: predict the future value using the most recently observed target.
    target_lag_cols = [c for c in X.columns if c.endswith("_lag_1") and "target" not in c]
    if target_lag_cols:
        naive_col = target_lag_cols[0]
        naive_val = X_val[naive_col].to_numpy()
        naive_test = X_test[naive_col].to_numpy()
    else:
        # The feature builder names target lags after the target, e.g. sales_count_lag_1.
        lag_candidates = [c for c in X.columns if c.endswith("_lag_1")]
        if not lag_candidates:
            raise ValueError("A lag-1 feature is required for the persistence baseline.")
        naive_col = lag_candidates[0]
        naive_val = X_val[naive_col].to_numpy()
        naive_test = X_test[naive_col].to_numpy()

    def record(name, y_true, pred, partition):
        mae = mean_absolute_error(y_true, pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        denom = np.abs(np.asarray(y_true, dtype=float))
        mape_mask = denom > 0
        mape = float(np.mean(np.abs((np.asarray(y_true)[mape_mask] - np.asarray(pred)[mape_mask]) /
                                    denom[mape_mask])) * 100) if mape_mask.any() else np.nan
        metrics.append({
            "model": name, "partition": partition, "horizon_steps": horizon_steps,
            "horizon_minutes": horizon_steps * 15, "MAE": float(mae), "RMSE": rmse,
            "MAPE_percent_excluding_zero_actuals": mape,
            "zero_actual_count": int((denom == 0).sum()), "n": int(len(y_true))
        })

    record("Naive persistence", y_val, naive_val, "validation")
    record("Naive persistence", y_test, naive_test, "test")
    predictions["Naive persistence"] = {"validation": naive_val, "test": naive_test}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            loss="squared_error", random_state=random_state
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        fitted_models[name] = model
        predictions[name] = {"validation": val_pred, "test": test_pred}
        record(name, y_val, val_pred, "validation")
        record(name, y_test, test_pred, "test")

    metric_df = pd.DataFrame(metrics)
    context = {
        "horizon_steps": horizon_steps,
        "horizon_minutes": horizon_steps * 15,
        "rows_used": len(model_data),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "feature_columns": list(X.columns),
        "train_start": str(model_data["timestamp"].iloc[0]),
        "train_end": str(model_data["timestamp"].iloc[train_end - 1]),
        "validation_start": str(model_data["timestamp"].iloc[train_end]),
        "validation_end": str(model_data["timestamp"].iloc[val_end - 1]),
        "test_start": str(model_data["timestamp"].iloc[val_end]),
        "test_end": str(model_data["timestamp"].iloc[-1]),
        "naive_feature": naive_col
    }
    return {
        "metrics": metric_df, "models": fitted_models, "predictions": predictions,
        "context": context, "y_validation": y_val.to_numpy(), "y_test": y_test.to_numpy(),
        "validation_timestamps": model_data["timestamp"].iloc[train_end:val_end].to_numpy(),
        "test_timestamps": model_data["timestamp"].iloc[val_end:].to_numpy(),
        "test_actual": y_test.to_numpy()
    }


def run_horizon_experiments(feature_df, horizons=(1, 2, 4, 8), output_dir="reports"):
    output = Path(output_dir)
    reports = output / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_contexts = []
    result_by_horizon = {}
    for h in horizons:
        result = fit_forecasting_models(feature_df, horizon_steps=h)
        result_by_horizon[h] = result
        all_metrics.append(result["metrics"])
        all_contexts.append(result["context"])
        best_val = (result["metrics"].query("partition == 'validation'")
                    .sort_values("MAE").iloc[0]["model"])
        result["context"]["best_model_by_validation_mae"] = str(best_val)
        # Test predictions for each model are stored separately for auditability.
        pred_frame = pd.DataFrame({
            "timestamp": result["test_timestamps"],
            "actual": result["test_actual"]
        })
        for model_name, pred in result["predictions"].items():
            pred_frame[model_name] = pred["test"]
        pred_frame.to_csv(reports / f"test_predictions_{h*15}min.csv", index=False)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(reports / "forecast_model_metrics.csv", index=False)
    (reports / "forecast_experiment_summary.json").write_text(
        json.dumps(all_contexts, indent=2, default=str), encoding="utf-8"
    )
    return result_by_horizon, metrics
