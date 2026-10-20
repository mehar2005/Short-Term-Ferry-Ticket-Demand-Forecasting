# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

A modular Streamlit data-science project following the same overall organization as the earlier predictive forecasting project: one root `app.py`, reusable pipeline modules under `src/`, raw and processed data separated, and reports/figures stored independently.

## Project structure

```text
ferry_demand_forecasting/
├── app.py
├── requirements.txt
├── run_analysis.py
├── generate_executive_summary.py
├── data/
│   ├── raw/                 # Original CSV; keep unchanged
│   └── processed/           # Reserved for processed data and exports
├── src/
│   ├── data_validation.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── temporal_split.py
│   ├── forecasting.py
│   ├── evaluation.py
│   └── dashboard.py
├── reports/
│   ├── figures/             # Generated plots
│   ├── tables/              # Metrics, features, splits, reports
│   └── research_paper_template.md
└── notebooks/               # Optional exploratory work
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To run the pipeline from the command line after placing the official file at `data/raw/Toronto Island Ferry Tickets.csv`:

```bash
python run_analysis.py
```

## Dashboard navigation

- Project Overview
- Data Quality
- Exploratory Analysis
- Feature Engineering
- Forecasting & Validation
- Model Evaluation
- Reports & Downloads

Upload the official CSV or Excel file using the sidebar. The dataset was not bundled because the shared WPS URL did not expose raw CSV bytes during project preparation.

## Modelling

The project compares persistence baseline, Linear Regression, Random Forest, and Gradient Boosting at 15-, 30-, 60-, and 120-minute horizons. Splits preserve chronological order. Models are selected using validation metrics; test results are held out for final reporting. MAPE excludes zero actuals and reports their count. Empirical residual-based prediction intervals are approximate and not guaranteed calibrated.

## Submission warning

Run the app and pipeline on the official dataset before using any statistics in the research paper. Do not invent scores or claim the app has been deployed until verified.
