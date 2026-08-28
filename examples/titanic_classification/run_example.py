"""Run tabular-autopilot on the Titanic dataset (classification, mixed types, missing data)."""

from pathlib import Path

from tabular_autopilot.pipeline import run_pipeline_from_csv
from tabular_autopilot.report import write_report

DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "titanic.html"

if __name__ == "__main__":
    result = run_pipeline_from_csv(DATA_PATH, target="Survived", dataset_name="titanic")
    write_report(result, REPORT_PATH)
    print(f"Task: {result.schema.task}")
    print(f"Best model: {result.modeling.best_model_name}")
    for name, value in result.modeling.metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Report: {REPORT_PATH}")
