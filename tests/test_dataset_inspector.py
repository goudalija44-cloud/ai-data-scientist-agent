
import pandas as pd
import pytest

from src.inspect_dataset import inspect_dataset


def test_inspector_reports_dimensions_and_column_details(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "Age,Plan\n"
        "22,Basic\n"
        "25,Premium\n"
        "30,Basic\n",
        encoding="utf-8",
    )

    result = inspect_dataset(csv_file)

    assert result["status"] == "success"
    assert result["rows"] == 3
    assert result["columns_count"] == 2
    assert result["columns"]["Age"]["data_type"] == "int64"
    assert result["columns"]["Age"]["missing_values"] == 0
    assert result["columns"]["Age"]["unique_values"] == 3
    assert len(result["sample_rows"]) == 3


def test_inspector_reports_missing_values(tmp_path):
    csv_file = tmp_path / "missing.csv"
    csv_file.write_text(
        "Age,Plan\n"
        "22,Basic\n"
        ",Premium\n"
        "30,Basic\n",
        encoding="utf-8",
    )

    result = inspect_dataset(csv_file)

    assert result["columns"]["Age"]["missing_values"] == 1
    assert result["columns"]["Age"]["unique_values"] == 2


def test_inspector_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_dataset(tmp_path / "does_not_exist.csv")
