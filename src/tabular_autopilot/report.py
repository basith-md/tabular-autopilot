"""Render an ``AnalysisResult`` into a single self-contained HTML report."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from tabular_autopilot.pipeline import AnalysisResult


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("tabular_autopilot", "templates"),
        autoescape=select_autoescape(["html"]),
    )


def render_html(result: AnalysisResult) -> str:
    env = _env()
    template = env.get_template("report_template.html")
    schema = result.schema
    columns_table = [
        {
            "name": name,
            "role": prof.role.value,
            "dtype": prof.dtype,
            "pct_missing": round(prof.pct_missing * 100, 1),
            "n_unique": prof.n_unique,
        }
        for name, prof in schema.columns.items()
    ]
    return template.render(
        dataset_name=result.dataset_name,
        schema=schema,
        profile=result.profile,
        cleaning=result.cleaning,
        fe=result.feature_engineering,
        modeling=result.modeling,
        charts=result.charts,
        columns_table=columns_table,
        timeseries=result.timeseries,
    )


def write_report(result: AnalysisResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(result), encoding="utf-8")
    return output_path
