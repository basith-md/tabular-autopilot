"""Run tabular-autopilot on the synthetic retail sales dataset (datetime + time-series)."""

from pathlib import Path

from tabular_autopilot.pipeline import run_pipeline_from_csv
from tabular_autopilot.report import write_report

DATA_PATH = Path(__file__).parent / "data" / "retail_sales.csv"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "retail_sales.html"

if __name__ == "__main__":
    result = run_pipeline_from_csv(DATA_PATH, target="units_sold", dataset_name="retail_sales")
    write_report(result, REPORT_PATH)
    print(f"Task: {result.schema.task}")
    print(f"Best model: {result.modeling.best_model_name}")
    for name, value in result.modeling.metrics.items():
        print(f"  {name}: {value:.4f}")
    if result.timeseries:
        ts = result.timeseries
        print(f"Time series trend R^2: {ts.trend_r_squared:.3f}, ADF p-value: {ts.adf_pvalue:.4f}")
    print(f"Report: {REPORT_PATH}")
