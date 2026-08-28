"""Run tabular-autopilot on the California Housing dataset (regression, geospatial)."""

from pathlib import Path

from tabular_autopilot.pipeline import run_pipeline_from_csv
from tabular_autopilot.report import write_report

DATA_PATH = Path(__file__).parent / "data" / "housing.csv"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "california_housing.html"

if __name__ == "__main__":
    result = run_pipeline_from_csv(DATA_PATH, target="median_house_value", dataset_name="california_housing")
    write_report(result, REPORT_PATH)
    print(f"Task: {result.schema.task}")
    print(f"Best model: {result.modeling.best_model_name}")
    for name, value in result.modeling.metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Report: {REPORT_PATH}")
