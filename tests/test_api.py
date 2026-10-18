from fastapi.testclient import TestClient

from src.api import app 


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_csv_upload():
    csv_content = (
        "age,income,purchased\n"
        "22,25000,0\n"
        "25,30000,0\n"
        "28,35000,1\n"
        "31,40000,1\n"
        "35,45000,1\n"
        "40,50000,0\n"
        "45,55000,1\n"
        "50,60000,1\n"
        "55,65000,0\n"
        "60,70000,1\n"
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.csv",
                csv_content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    report = response.json()

    assert report["status"] == "success"
    assert "dataset_overview" in report
    assert "analysis" in report
    assert "ml_readiness" in report



def test_reject_non_csv_upload():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.txt",
                "this is not a CSV upload",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a CSV file."



def test_train_rejects_missing_target_column():
    response = client.post(
        "/train",
        data={
            "target_column": "DoesNotExist",
            "task": "classification",
            "test_size": "0.2",
            "random_state": "42",
        },
        files={
            "file": (
                "sample.csv",
                b"Age,Plan\n21,Basic\n25,Premium\n32,Basic\n"
                b"28,Premium\n41,Basic\n36,Premium\n",
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 422)


def test_train_rejects_invalid_task():
    response = client.post(
        "/train",
        data={
            "target_column": "Plan",
            "task": "clustering",
            "test_size": "0.2",
            "random_state": "42",
        },
        files={
            "file": (
                "sample.csv",
                b"Age,Plan\n21,Basic\n25,Premium\n32,Basic\n"
                b"28,Premium\n41,Basic\n36,Premium\n",
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 422)


def test_train_rejects_invalid_test_size():
    response = client.post(
        "/train",
        data={
            "target_column": "Plan",
            "task": "classification",
            "test_size": "1.5",
            "random_state": "42",
        },
        files={
            "file": (
                "sample.csv",
                b"Age,Plan\n21,Basic\n25,Premium\n32,Basic\n"
                b"28,Premium\n41,Basic\n36,Premium\n",
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 422)


    
def test_train_classification_success():
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
        "31,270,Basic\n"
        "27,420,Premium\n"
        "39,230,Basic\n"
        "22,380,Premium\n"
        "34,260,Basic\n"
        "43,480,Premium\n"
        "29,210,Basic\n"
        "38,430,Premium\n"
    )

    response = client.post(
        "/train",
        data={
            "target_column": "Plan",
            "task": "classification",
            "test_size": "0.25",
            "random_state": "42",
        },
        files={
            "file": (
                "sample.csv",
                csv_content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["task"] == "classification"
    assert result["target_column"] == "Plan"
    assert result["train_rows"] == 12
    assert result["test_rows"] == 4

    assert "accuracy" in result["metrics"]
    assert "f1_weighted" in result["metrics"]
    assert len(result["class_metrics"]) > 0
    assert len(result["confusion_matrix"]) > 0
    assert len(result["confusion_matrix_labels"]) > 0
    assert result["warning"]
