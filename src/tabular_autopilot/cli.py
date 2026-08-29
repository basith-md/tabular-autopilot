"""Command-line entry point: ``tabular-autopilot run data.csv --target col``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabular_autopilot.pipeline import run_pipeline_from_csv
from tabular_autopilot.report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabular-autopilot")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the automated pipeline on a dataset.")
    run_p.add_argument("data_path", help="Path to a CSV, Excel, or Parquet file.")
    run_p.add_argument("--target", default=None, help="Target/label column for modeling. Omit for EDA-only.")
    run_p.add_argument("--out", default=None, help="Output HTML report path. Defaults to reports/<name>.html")
    run_p.add_argument("--name", default=None, help="Dataset name shown in the report. Defaults to file stem.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"error: file not found: {data_path}", file=sys.stderr)
            return 1
        name = args.name or data_path.stem
        out_path = Path(args.out) if args.out else Path("reports") / f"{name}.html"

        print(f"Running Tabular Autopilot on {data_path} (target={args.target!r}) ...")
        result = run_pipeline_from_csv(data_path, target=args.target, dataset_name=name)
        written = write_report(result, out_path)

        print(f"Rows x Cols: {result.profile.n_rows} x {result.profile.n_cols}")
        if result.modeling:
            print(f"Task: {result.modeling.task}")
            for metric, value in result.modeling.metrics.items():
                print(f"  {metric}: {value:.4f}")
        print(f"Report written to: {written.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
