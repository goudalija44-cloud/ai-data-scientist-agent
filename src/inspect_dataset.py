import argparse
import json

from src.data_loader import load_csv


def inspect_dataset(file_path):
    """Return a compact, JSON-friendly summary of a csv dataset."""
    df = load_csv(file_path)

    columns = {}

    for column in df.columns:
        columns[str(column)] = {
            "data_type": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique(dropna=True)),
        }

    return {
        "status": "success",
        "dataset_name": file_path.stem,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": columns,
        "sample_rows": json.loads(
            df.head(5).to_json(orient="records", date_format="iso")
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a CSV dataset before analysis or ML training."
    )

    parser.add_argument(
        "csv_file",
        help="Path to the CSV file you want to inspect.",
    )

    args = parser.parse_args()

    from pathlib import Path

    file_path = Path(args.csv_file).resolve()
    result = inspect_dataset(file_path)


    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()