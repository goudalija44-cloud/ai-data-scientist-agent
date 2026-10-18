import pandas as pd
import pytest

from src.data_loader import load_csv
from src.run_dataset_analysis import analyze_csv_file


def test_load_valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    pd.DataFrame({
        "Age": [20, 30],
        "Salary": [25000, 40000],
    }).to_csv(path, index=False)

    df = load_csv(path)

    assert df.shape == (2, 2)
    assert df["Age"].tolist() == [20, 30]


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_non_csv_file_is_rejected(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("example", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSV"):
        load_csv(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv(path)


def test_headers_without_rows_are_rejected(tmp_path):
    path = tmp_path / "headers_only.csv"
    path.write_text("Age,Salary\\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv(path)


def test_end_to_end_analysis(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({
        "Age": [20, 30, 30],
        "City": ["Thane", "Mumbai", "Mumbai"],
    }).to_csv(path, index=False)

    report = analyze_csv_file(path)

    assert report["status"] == "success"
    assert report["source_file"] == "customers.csv"
    assert report["dataset_profile"]["rows"] == 3
    assert report["data_quality"]["duplicate_row_count"] == 1


def test_load_csv_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises((FileNotFoundError, ValueError)):
        load_csv(missing_file)


def test_load_csv_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("name,age\nAlice,25\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv(file_path)


def test_load_csv_rejects_empty_file(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_csv(file_path)


def test_load_csv_rejects_csv_with_no_data_rows(tmp_path):
    file_path = tmp_path / "header_only.csv"
    file_path.write_text("name,age\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv(file_path)