\
from pathlib import Path
from src.data_validation import load_dataset, validate_dataset, write_validation_outputs
from src.eda import run_eda
from src.feature_engineering import create_forecasting_features, save_feature_outputs
from src.temporal_split import chronological_split, rolling_origin_folds, save_split_outputs
from src.forecasting import run_horizon_experiments
from src.evaluation import evaluate_horizon_results

DATA_PATH = Path("data/raw/Toronto Island Ferry Tickets.csv")

def main():
    df = load_dataset(DATA_PATH)
    cleaned, report = validate_dataset(df)
    write_validation_outputs(report, "reports/tables")
    if report["issues"]:
        print("Required columns not detected:", report["issues"])
        return
    run_eda(cleaned, "reports")
    print("Sections 1–2 complete.")
    for target in ["sales_count", "redemption_count"]:
        if target not in report["resolved_columns"]:
            continue
        features, meta = create_forecasting_features(cleaned, target=target)
        save_feature_outputs(features, meta, "reports/tables")
        train, validation, test, summary = chronological_split(features)
        folds = rolling_origin_folds(features)
        save_split_outputs(train, validation, test, summary, folds, "reports/tables")
        print(f"{target}: {len(features)} model-ready rows; chronological splits saved.")
        results, metrics = run_horizon_experiments(features, horizons=(1, 2, 4, 8), output_dir="reports")
        uncertainty = evaluate_horizon_results(results, output_dir="reports")
        print(f"Sections 5–6 complete for {target}.")
        print(metrics.to_string(index=False))
    print("Sections 1–6 completed. Review outputs/reports and outputs/figures.")

if __name__ == "__main__":
    main()
