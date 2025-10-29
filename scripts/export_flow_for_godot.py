#!/usr/bin/env python
"""
Utility to export process flow data as JSON for the Godot Flow Explorer prototype.

Usage:
    python scripts/export_flow_for_godot.py --input data/sample_order_to_cash.csv --output runtime/flow_payload.json

The resulting JSON contains nodes, edges, and metadata required by the Godot scene.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import process_analysis  # noqa: E402  pylint: disable=wrong-import-position
from app.log_loader import (  # noqa: E402  pylint: disable=wrong-import-position
    load_log_from_csv,
    load_log_from_xes,
    LogFormatError,
    try_auto_detect_columns,
)


def export_flow(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        file_bytes = input_path.read_bytes()
        preview_df = pd.read_csv(io.BytesIO(file_bytes), nrows=200)
        case_col, act_col, ts_col = try_auto_detect_columns(preview_df)
        try:
            container = load_log_from_csv(
                file_bytes,
                case_id_col=case_col,
                activity_col=act_col,
                timestamp_col=ts_col,
            )
        except LogFormatError as exc:
            raise SystemExit(f"Failed to load CSV log: {exc}") from exc
    elif suffix == ".xes":
        loader = load_log_from_xes
        try:
            container = loader(input_path.read_bytes())
        except LogFormatError as exc:
            raise SystemExit(f"Failed to load log: {exc}") from exc
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .xes")

    payload = process_analysis.build_flow_payload(container, max_activities=24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote flow payload to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export flow data for the Godot Flow Explorer prototype.")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to the source log (.csv or .xes).")
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Destination JSON file for the Godot payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_flow(args.input, args.output)


if __name__ == "__main__":
    main()
