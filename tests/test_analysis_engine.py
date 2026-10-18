import pandas as pd
import pytest

from src.analysis_engine import analyze_dataset
from src.data_profiler import profile_dataset
from src.data_quality import check_data_quality


def test_profiler_counts_rows_columns_and_missing_values():
    df = pd.DataFrame({
        "Age": [20, 30, None],
        "City": ["Thane", "Mumbai", "Thane"],
    })

    result = profile_dataset(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["total_missing_values"] == 1


def test_quality_checker_detects_duplicates():
    df = pd.DataFrame({
        "Score": [10, 20, 20],
    })

    result = check_data_quality(df)

    assert result["duplicate_row_count"] == 1


def test_analysis_engine_calculates_mean():
    df = pd.DataFrame({
        "Salary": [20000, 30000, 40000],
    })

    result = analyze_dataset(df)

    assert result["numeric_statistics"]["Salary"]["mean"] == 30000.0


def test_analysis_engine_excludes_missing_values():
    df = pd.DataFrame({
        "Sales": [100, None, 300],
    })

    result = analyze_dataset(df)

    stats = result["numeric_statistics"]["Sales"]

    assert stats["count"] == 2
    assert stats["mean"] == 200.0


def test_empty_dataframe_is_rejected():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        analyze_dataset(df)
