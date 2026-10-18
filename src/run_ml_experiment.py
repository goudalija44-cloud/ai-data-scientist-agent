import argparse
import json
from pathlib import Path

from src.data_loader import load_csv
from src.ml_engine import train_baseline_model

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a baseline ML model on a CSV dataset."
    )

    parser.add_argument(
        "csv_file",
        help="Path to the CSV dataset."
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Name of the column to predict."
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="Choose classification or regression."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for testing (default: 0.2). ",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file).resolve()


    # Load the dataset using the project's existing validation.
    df = load_csv(csv_path)

    # Train and evaluate the model.
    result = train_baseline_model(
        df=df,
        target_column=args.target,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    result["dataset_name"] = csv_path.stem
    result["source_file"] = csv_path.name

    # Save a report without overwriting the general analysis report.
    project_root = Path(__file__).resolve().parent.parent
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"{csv_path.stem}_ml_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print(json.dumps(result, indent=4))
    print(f"\nML report saved to: {report_path}")


if __name__ == "__main__":
    main()