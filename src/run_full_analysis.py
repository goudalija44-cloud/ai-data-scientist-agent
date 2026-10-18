
import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis_pipeline import run_analysis
from src.data_loader import load_csv


MAX_SCATTER_POINTS = 500
MAX_CATEGORY_VALUES = 20
HISTOGRAM_BINS = 20


def build_visualization_data(df, analysis):
    """Build compact, JSON-friendly data for frontend charts."""
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    visualization_data = {
        "numerical_columns": [str(column) for column in numerical_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "histograms": {},
        "category_counts": {},
        "scatter": {},
        "correlation": {
            "columns": [],
            "matrix": [],
        },
    }

    # Histogram bin counts for each numerical column.
    for column in numerical_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if values.empty:
            visualization_data["histograms"][str(column)] = {
                "bin_edges": [],
                "counts": [],
            }
            continue

        counts, edges = pd.cut(
            values,
            bins=min(HISTOGRAM_BINS, max(1, values.nunique())),
            include_lowest=True,
            retbins=True,
        ), None

        # Use numpy's histogram to obtain reliable numeric bin edges and counts.
        import numpy as np

        histogram_counts, bin_edges = np.histogram(
            values.to_numpy(),
            bins=min(HISTOGRAM_BINS, max(1, values.nunique())),
        )

        visualization_data["histograms"][str(column)] = {
            "bin_edges": [round(float(edge), 4) for edge in bin_edges],
            "counts": [int(count) for count in histogram_counts],
        }

    # Category frequencies. Keep the top values to limit response size.
    for column in categorical_columns:
        counts = df[column].fillna("(Missing)").astype(str).value_counts()
        top_counts = counts.head(MAX_CATEGORY_VALUES)

        category_data = [
            {"label": str(label), "count": int(count)}
            for label, count in top_counts.items()
        ]

        other_count = int(counts.iloc[MAX_CATEGORY_VALUES:].sum())
        if other_count > 0:
            category_data.append(
                {"label": "Other", "count": other_count}
            )

        visualization_data["category_counts"][str(column)] = category_data

    # Scatter data: first two numerical columns, sampled to a bounded size.
    if len(numerical_columns) >= 2:
        x_column = numerical_columns[0]
        y_column = numerical_columns[1]

        scatter_df = df[[x_column, y_column]].dropna()

        if len(scatter_df) > MAX_SCATTER_POINTS:
            scatter_df = scatter_df.sample(
                n=MAX_SCATTER_POINTS,
                random_state=42,
            )

        visualization_data["scatter"] = {
            "x_column": str(x_column),
            "y_column": str(y_column),
            "x": [float(value) for value in scatter_df[x_column]],
            "y": [float(value) for value in scatter_df[y_column]],
        }

    # Correlation heatmap uses the analysis engine's existing results.
    correlation_data = analysis.get("statistical_analysis", {}).get(
        "correlations", {}
    )
    correlation_columns = list(correlation_data.keys())

    visualization_data["correlation"] = {
        "columns": correlation_columns,
        "matrix": [
            [
                correlation_data.get(row, {}).get(column)
                for column in correlation_columns
            ]
            for row in correlation_columns
        ],
    }

    return visualization_data


def run_full_analysis(csv_file):
    """Combine dataset analysis and chart data into one report."""
    csv_file = Path(csv_file).resolve()
    df = load_csv(csv_file)

    analysis = run_analysis(df)

    columns = {
        str(column): {
            "data_type": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    }

    missing_by_column = {
        name: details["missing_values"]
        for name, details in columns.items()
        if details["missing_values"] > 0
    }

    report = {
        "status": "success",
        "dataset_name": csv_file.stem,
        "source_file": csv_file.name,
        "dataset_overview": {
            "rows": int(df.shape[0]),
            "columns_count": int(df.shape[1]),
            "columns": columns,
            "sample_rows": json.loads(
                df.head(5).to_json(
                    orient="records",
                    date_format="iso",
                )
            ),
        },
        "analysis": analysis,
        "visualization_data": build_visualization_data(df, analysis),
        "ml_readiness": {
            "status": "requires_target_selection",
            "message": (
                "Choose a target column and a task before training "
                "a machine-learning model. Review missing values, "
                "class balance, possible data leakage, and dataset size."
            ),
            "columns_available_as_targets": [
                str(column) for column in df.columns
            ],
        },
        "missing_values_summary": {
            "columns_with_missing_values": missing_by_column,
            "total_missing_values": int(df.isna().sum().sum()),
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate a unified dataset analysis report."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the JSON report.",
    )

    args = parser.parse_args()
    csv_file = Path(args.csv_file).resolve()

    report = run_full_analysis(csv_file)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = (
            Path(__file__).resolve().parent.parent
            / "reports"
            / f"{csv_file.stem}_full_report.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print("Full dataset analysis completed successfully.")
    print(f"Dataset: {report['dataset_name']}")
    print(f"Rows: {report['dataset_overview']['rows']}")
    print(f"Columns: {report['dataset_overview']['columns_count']}")
    print("Insights:", report["analysis"]["insights"]["insight_count"])
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
