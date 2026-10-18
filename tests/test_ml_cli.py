import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_ml_cli_creates_report_for_valid_classification(tmp_path):
    csv_file = tmp_path / "customers_cli_test.csv"
    csv_file.write_text(
         "Age,Monthly_Spend,Plan\n"
        "22,1000,Basic\n"
        "25,1200,Basic\n"
        "28,1400,Basic\n"
        "31,1600,Basic\n"
        "34,1800,Basic\n"
        "37,2000,Basic\n"
        "40,2200,Basic\n"
        "43,2400,Basic\n"
        "24,3000,Premium\n"
        "27,3200,Premium\n"
        "30,3400,Premium\n"
        "33,3600,Premium\n"
        "36,3800,Premium\n"
        "39,4000,Premium\n"
        "42,4200,Premium\n"
        "45,4400,Premium\n",
        encoding="utf-8",
    )

    report_path = (
        PROJECT_ROOT / "reports" / "customers_cli_test_ml_report.json"
    )

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.run_ml_experiment",
                str(csv_file),
                "--target",
                "Plan",
                "--task",
                "classification",
            ],

            cwd = PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert completed.returncode == 0, completed.stderr
        assert "ML report saved to:" in completed.stdout
        assert report_path.exists()

        with report_path.open(encoding="utf-8") as file:
            report = json.load(file)

        assert report["status"] == "success"
        assert report["target_column"] == "Plan"
        assert report["task"] == "classification"
        assert "confusion_matrix" in report
        assert "model_configuration" in report

    finally:
        if report_path.exists():
            report_path.unlink()


def test_ml_cli_rejects_missing_target_argument(tmp_path):
    csv_file = tmp_path / "customers_invalid_cli.csv"
    csv_file.write_text(
        "Age,Plan\n22,Basic\n25,Premium\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_ml_experiment",
            str(csv_file),
            "--task",
            "classification",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "--target" in completed.stderr