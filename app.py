from pathlib import Path
import streamlit as st
import pandas as pd

from src.data_validation import load_dataset, validate_dataset, write_validation_outputs
from src.eda import run_eda
from src.feature_engineering import create_forecasting_features, save_feature_outputs
from src.temporal_split import chronological_split, rolling_origin_folds, save_split_outputs
from src.forecasting import run_horizon_experiments
from src.evaluation import evaluate_horizon_results
from src.dashboard import build_executive_summary, plot_model_comparison

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "Toronto Island Ferry Tickets.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(page_title="Ferry Demand Forecasting", page_icon="⛴️", layout="wide", initial_sidebar_state="expanded")
st.title("⛴️ Short-Term Ferry Ticket Demand Forecasting")
st.caption("Predictive Decision Support System | Unified Mentor Internship Project")
st.divider()

with st.sidebar:
    st.header("Project Navigation")
    page = st.radio("Go to", ["Project Overview", "Data Quality", "Exploratory Analysis", "Feature Engineering", "Forecasting & Validation", "Model Evaluation"])
    st.divider()
    st.caption("Forecast horizons: 15 minutes to 2 hours")
    st.caption("Data is processed locally in this app session.")

uploaded = st.sidebar.file_uploader("Upload ferry ticket CSV / Excel", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded) if uploaded.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded)
        st.session_state["source_df"] = df
        st.session_state["source_name"] = uploaded.name
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")
elif "source_df" not in st.session_state and RAW_PATH.exists():
    st.session_state["source_df"] = load_dataset(RAW_PATH)
    st.session_state["source_name"] = RAW_PATH.name

if "source_df" not in st.session_state:
    st.info("Upload the official Toronto Island Ferry Tickets dataset in the sidebar, or place it at `data/raw/Toronto Island Ferry Tickets.csv`.")
    st.markdown("### Project workflow")
    st.markdown("1. Validate timestamp and ticket-count columns.\n2. Explore demand patterns.\n3. Create leak-safe lag, rolling, and calendar features.\n4. Split chronologically.\n5. Train baseline and machine-learning models.\n6. Evaluate multiple forecast horizons and uncertainty.")
    st.stop()

raw_df = st.session_state["source_df"].copy()
cleaned_df, quality = validate_dataset(raw_df)
st.session_state["cleaned_df"] = cleaned_df
st.session_state["quality_report"] = quality

if page == "Project Overview":
    st.subheader("Project Overview")
    st.write("This project analyses 15-minute ferry ticket activity and builds short-term forecasting models to support proactive operational planning.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Observations", f"{quality['rows']:,}")
    c2.metric("Columns", quality['columns'])
    c3.metric("Missing cells", f"{quality['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
    st.markdown("#### Main objectives")
    st.markdown("- Forecast ticket sales and redemptions at 15-, 30-, 60-, and 120-minute horizons.\n- Compare a persistence baseline with regression and tree-based models.\n- Preserve chronological order during validation.\n- Present model error and approximate uncertainty in an accessible dashboard.")
    st.markdown("#### Data preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.caption(f"Loaded source: {st.session_state.get('source_name','uploaded dataset')}")

elif page == "Data Quality":
    st.subheader("Data Quality and Validation")
    st.dataframe(pd.DataFrame({"Column": raw_df.columns.astype(str), "Data type": raw_df.dtypes.astype(str).values}), use_container_width=True)
    if quality.get("gap_rows"):
        st.markdown("#### Timestamp gaps")
        st.dataframe(pd.DataFrame(quality["gap_rows"]), use_container_width=True)
    if st.button("Save data-quality report"):
        write_validation_outputs(quality, str(REPORTS_DIR / "tables"))
        st.success("Data-quality report saved under reports/tables.")
    if quality["issues"]:
        st.error("Required columns are missing or unrecognized. Resolve these before modelling.")

elif page == "Exploratory Analysis":
    st.subheader("Exploratory Data Analysis")
    if st.button("Run EDA", type="primary"):
        with st.spinner("Generating charts and summary tables..."):
            result = run_eda(cleaned_df, output_dir=str(REPORTS_DIR))
        st.session_state["eda_result"] = result
        st.success("EDA complete.")
    if "eda_result" in st.session_state:
        result = st.session_state["eda_result"]
        st.markdown("#### Descriptive statistics")
        st.dataframe(result["statistics"], use_container_width=True)
        st.markdown("#### Correlation")
        st.dataframe(result["correlation"], use_container_width=True)
        for image_path in sorted(FIGURES_DIR.glob("*.png")):
            st.image(str(image_path), caption=image_path.stem.replace("_", " ").title(), use_container_width=True)

elif page == "Feature Engineering":
    st.subheader("Forecasting Feature Engineering")
    targets = [t for t in ["sales_count", "redemption_count"] if t in quality["resolved_columns"]]
    if not targets:
        st.error("Could not identify Sales Count or Redemption Count columns.")
    else:
        target = st.selectbox("Target to forecast", targets, format_func=lambda x:x.replace("_", " ").title())
        st.write("Features include target lags (1, 2, 4, 8 intervals), historical rolling statistics, calendar encodings, and lagged values of the other ticket measure.")
        if st.button("Generate features", type="primary"):
            features, metadata = create_forecasting_features(cleaned_df, target=target)
            save_feature_outputs(features, metadata, str(REPORTS_DIR / "tables"))
            st.session_state["features"] = features
            st.session_state["target"] = target
            st.success(f"Generated {len(metadata['features'])} predictors across {len(features):,} model-ready rows.")
            st.json(metadata)
        if "features" in st.session_state:
            st.dataframe(st.session_state["features"].head(20), use_container_width=True)
            st.download_button("Download feature dataset", st.session_state["features"].to_csv(index=False), file_name=f"features_{st.session_state['target']}.csv", mime="text/csv")

elif page == "Forecasting & Validation":
    st.subheader("Chronological Validation and Forecast Models")
    if "features" not in st.session_state:
        st.warning("Generate features first in the Feature Engineering page.")
    else:
        features = st.session_state["features"]
        a,b,c = st.columns(3)
        train_pct = a.slider("Training share (%)", 60, 80, 70, 5)
        test_pct = b.slider("Test share (%)", 10, 20, 15, 5)
        val_pct = 100 - train_pct - test_pct
        c.metric("Validation share", f"{val_pct}%")
        if st.button("Create chronological partitions"):
            if val_pct < 10:
                st.error("Validation share must be at least 10%.")
            else:
                tr,va,te,summary = chronological_split(features, train_pct/100, val_pct/100, test_pct/100)
                folds = rolling_origin_folds(features)
                save_split_outputs(tr,va,te,summary,folds,str(REPORTS_DIR / "tables"))
                st.dataframe(summary, use_container_width=True)
                st.success("Chronological partitions saved.")
        horizon_options = {"15 minutes":1,"30 minutes":2,"1 hour":4,"2 hours":8}
        selected = st.multiselect("Forecast horizons", list(horizon_options), default=list(horizon_options))
        if st.button("Train and evaluate models", type="primary"):
            if not selected:
                st.warning("Choose at least one horizon.")
            else:
                try:
                    with st.spinner("Training models and evaluating held-out periods..."):
                        results, metrics = run_horizon_experiments(features, tuple(horizon_options[x] for x in selected), output_dir=str(REPORTS_DIR))
                        uncertainty = evaluate_horizon_results(results, output_dir=str(REPORTS_DIR))
                    st.session_state["model_results"] = results
                    st.session_state["model_metrics"] = metrics
                    st.session_state["uncertainty"] = uncertainty
                    st.success("Model experiments completed.")
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

elif page == "Model Evaluation":
    st.subheader("Model Evaluation and Uncertainty")
    metrics = st.session_state.get("model_metrics")
    uncertainty = st.session_state.get("uncertainty")
    if metrics is None:
        metric_path = REPORTS_DIR / "tables" / "forecast_model_metrics.csv"
        if metric_path.exists(): metrics = pd.read_csv(metric_path)
    if metrics is None:
        st.info("Train models from the Forecasting & Validation page to generate metrics.")
    else:
        st.dataframe(metrics.sort_values(["horizon_steps","partition","MAE"]), use_container_width=True)
        st.download_button("Download model metrics", metrics.to_csv(index=False), "forecast_model_metrics.csv", "text/csv")
        val = metrics[metrics["partition"] == "validation"]
        if not val.empty:
            st.markdown("#### Validation MAE comparison (lower is better)")
            st.bar_chart(val.pivot(index="horizon_minutes", columns="model", values="MAE"))
    if uncertainty is None:
        unc_path = REPORTS_DIR / "tables" / "forecast_uncertainty_summary.csv"
        if unc_path.exists(): uncertainty = pd.read_csv(unc_path)
    if uncertainty is not None and len(uncertainty):
        st.markdown("#### Approximate prediction-interval diagnostics")
        st.dataframe(uncertainty, use_container_width=True)
    for image_path in sorted(FIGURES_DIR.glob("test_forecast_*.png")):
        st.image(str(image_path), caption=image_path.stem.replace("_", " ").title(), use_container_width=True)

st.divider()
st.caption("Forecasts are decision-support estimates, not dispatch instructions. Ticket sales and redemptions are not necessarily equivalent to passenger occupancy.")
