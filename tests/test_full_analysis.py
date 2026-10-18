import pandas as pd

import json

from src.run_full_analysis import run_full_analysis


def test_full_analysis_combines_report_sections(tmp_path):
    csv_file = tmp_path / "customers.csv"
    csv_file.write_text(
        "Age,Monthly_Spend,Plan\n"
        "22,1000,Basic\n"
        "25,1200,Basic\n"
        "28,1400,Premium\n"
        "31,1600,Premium\n"
        "34,1800,Basic\n"
        "37,2000,Premium\n",
        encoding="utf-8",
    )

    result = run_full_analysis(csv_file)

    assert result["status"] == "success"
    assert result["dataset_overview"]["rows"] == 6
    assert result["dataset_overview"]["columns_count"] == 3
    assert "analysis" in result
    assert "dataset_profile" in result["analysis"]
    assert "data_quality" in result["analysis"]
    assert "statistical_analysis" in result["analysis"]
    assert "insights" in result["analysis"]
    assert result["ml_readiness"]["status"] == (
        "requires_target_selection"
    )


def test_full_analysis_report_is_json_serializable(tmp_path):
    csv_file = tmp_path / "numeric.csv"
    csv_file.write_text(
        "Age,Spend\n"
        "20,100\n"
        "30,200\n"
        "40,300\n"
        "50,400\n"
        "60,500\n",
        encoding="utf-8",
    )

    result = run_full_analysis(csv_file)

    serialized = json.dumps(result, default=str)
    restored = json.loads(serialized)

    assert restored["status"] == "success"
    assert restored["dataset_overview"]["rows"] == 5



def test_full_analysis_returns_expected_sections(tmp_path):
    csv_content = (
        "Age,Monthly_Spend,Plan\n"
        "21,200,Basic\n"
        "25,350,Premium\n"
        "32,250,Basic\n"
        "28,400,Premium\n"
        "41,220,Basic\n"
        "36,450,Premium\n"
        "23,180,Basic\n"
        "45,500,Premium\n"
    )

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    report = run_full_analysis(csv_file)

   
    assert report["status"] == "success"
    assert "dataset_overview" in report
    assert "analysis" in report
    assert "visualization_data" in report
    assert "ml_readiness" in report
    assert "missing_values_summary" in report

