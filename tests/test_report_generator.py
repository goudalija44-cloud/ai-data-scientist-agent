import json

from src.generate_report import main

def test_report_generator_creates_json_report(tmp_path, monkeypatch):
    csv_file = tmp_path / "customers.csv"
    csv_file.write_text(
        "Customer_ID, Age, Monthly_Spend, Plan\n"
        "101, 25, 1200, Basic\n"
        "102, 25, 2500, Premium\n"
        "103, , 1800, Basic\n",
        encoding="utf-8",    
    )

    report_file = tmp_path / "output" / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_report",
            str(csv_file),
            "--output",
            str(report_file),
        ],
    )

    main()

    assert report_file.exists()

    with report_file.open("r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["status"] == "success"
    assert report["source_file"] == "customers.csv"
    assert report["dataset_profile"]["rows"] == 3
    assert report["dataset_profile"]["columns"] == 4
    assert "insights" in report