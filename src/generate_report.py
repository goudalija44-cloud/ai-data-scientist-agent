import argparse
import json
from pathlib import Path

from src.run_dataset_analysis import analyze_csv_file


def main():
    parser  = argparse.ArgumentParser(
        description="Analyze a CSV file dataset and save a JSON report."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file you want to analyze.",
    )

    parser.add_argument(
        "--output",
        help="Optional path for the JSON report."
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file).resolve()

    if args.output:
        report_path = Path(args.output).resolve()
    else:
        report_path = (
            Path(__file__).resolve().parent.parent
            / "reports"
            / f"{csv_path.stem}_report.json" 
        )

    result = analyze_csv_file(csv_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False, default=str)

    print("Analysis completed successfully.")
    print(f"Source file: {result['source_file']}")
    print(f"Rows: {result['dataset_profile']['rows']}")
    print(f"Columns: {result['dataset_profile']['columns']}")
    print(f"Insights: {result['insights']['insight_count']}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()