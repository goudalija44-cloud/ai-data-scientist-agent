import pandas as pd
import pytest
import pandas as pd
from src.ml_engine import train_baseline_model

@pytest.fixture
def classification_data():
    return pd.DataFrame({
        "Customer_ID": list(range(100, 140)),
        "Age": list(range(20, 60)),
        "Monthly_Spend": [500 + i * 20 for i in range(40)],
        "Plan": ["Basic" if i % 2 == 0 else "Premium" for i in range(40)], 
    })

@pytest.fixture
def regression_data():
    return pd.DataFrame({
        "Age": list(range(20, 60)),
        "Monthly_Spend": [500 + i * 20 for i in range(40)],
        "Annual_Spend": [6000 + i * 240 for i in range(40)],
    })

def test_classification_model_returns_metrics(classification_data):
    result = train_baseline_model(
        classification_data,
        target_column="Plan",
        task="classification",
    )

    assert result["status"] == "success"
    assert result["task"] == "classification"
    assert result["target_column"] == "Plan"
    assert "accuracy" in result["metrics"]
    assert "f1_weighted" in result["metrics"]
    assert "Customer_ID" in result["excluded_identifier_columns"]


def test_regression_model_returns_metrics(regression_data):
    result = train_baseline_model(
        regression_data,
        target_column="Annual_Spend",
        task="regression"
    )

    assert result["status"] == "success"
    assert result["task"] == "regression"
    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]


def test_missing_target_column_is_rejected(classification_data):
    with pytest.raises(ValueError, match="does not exist"):
        train_baseline_model(
            classification_data,
            target_column="Unknown",
            task="classification",
        )

def test_invalid_task_is_rejected(classification_data):
    with pytest.raises(ValueError, match="task must be"):
        train_baseline_model(
            classification_data,
            target_column="Plan",
            task="clustering",
        )        


def test_missing_target_values_are_rejected(classification_data):
    classification_data.loc[0, "Plan"] = None

    with pytest.raises(ValueError, match="missing values"):
        train_baseline_model(
            classification_data,
            target_column="Plan",
            task="classification",
        )

def test_regression_requires_numeric_target(classification_data):
    with pytest.raises(ValueError, match="numeric target"):
        train_baseline_model(
            classification_data,
            target_column="Plan",
            task="regression"
        )

def test_dataset_with_too_few_rows_is_rejected():
    tiny_data = pd.DataFrame({
        "Feature": [1,2,3,4],
        "Target": [0,1,0,1],
    })

    with pytest.raises(ValueError, match="At least 5 rows"):
        train_baseline_model(
            tiny_data,
            target_column="Target",
            task="classification",
        )

def test_classification_returns_per_class_metrics_and_confusion_matrix():
    df = pd.DataFrame({
        "Age": list(range(20, 60)),
        "Monthly_Spend": [i * 100 for i in range(40)],
        "Plan": ["Basic", "Premium"] * 20,
        "target": [0, 1] * 20,
    })

    result = train_baseline_model(
        df=df,
        target_column="target",
        task="classification",   
    )

    assert "class_metrics" in result
    assert "confusion_matrix" in result
    assert "confusion_matrix_labels" in result

    assert len(result["class_metrics"]) == 2
    assert len(result["confusion_matrix"]) == 2
    assert all(
        len(row) == 2 for row in result["confusion_matrix"]
    )

    assert sum(
        sum(row) for row in result["confusion_matrix"]
    ) == result["test_rows"]


def test_regression_does_not_return_classification_details():
    df = pd.DataFrame({
        "Age": list(range(20, 60)),
        "Monthly_Spend": [i * 100 for i in range(40)],
        "Annual_Spend": [i * 1200 for i in range(40)],
    })

    result = train_baseline_model(
        df=df,
        target_column="Annual_Spend",
        task="regression",
    )

    assert result["status"] == "success"
    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "class_metrics" not in result
    assert "confusion_matrix" not in result


def test_classification_returns_evaluation_metadata():
    df=pd.DataFrame({
        "Age": list(range(20, 60)),
        "Monthly_Spend": [i * 100 for i in range(40)],
        "Plan": ["Basic", "Premium"] * 20,
        "target": [0, 1] * 20, 
    })

    result = train_baseline_model(
        df=df,
        target_column="target",
        task="classification",
        test_size=0.2,
        random_state=42,
    )

    evaluation = result["evaluation"]
    configuration = result["model_configuration"]

    assert evaluation["method"] == "train_test_split"
    assert evaluation["test_size"] == 0.2
    assert evaluation["random_state"] == 42
    assert evaluation["train_rows"] == result["train_rows"]
    assert evaluation["test_rows"] == result["test_rows"]

    assert configuration["algorithm"] == "LogisticRegression"
    assert "median_imputation" in (
        configuration["preprocessing"]["numeric_features"]
    )

    assert "one_hot_encoding" in (
        configuration["preprocessing"]["categorical_features"]
    )


def test_classification_handles_categorical_features():
    df = pd.DataFrame({
        "Age": [
            21, 25, 32, 28, 41, 36, 23, 45,
            31, 27, 39, 22, 34, 43, 29, 38,
            24, 42, 33, 26
        ],
        "City": [
            "Mumbai", "Pune", "Thane", "Mumbai", "Pune",
            "Thane", "Mumbai", "Pune", "Thane", "Mumbai",
            "Pune", "Thane", "Mumbai", "Pune", "Thane",
            "Mumbai", "Pune", "Thane", "Mumbai", "Pune"
        ],
        "Plan": [
            "Basic", "Premium", "Basic", "Premium", "Basic",
            "Premium", "Basic", "Premium", "Basic", "Premium",
            "Basic", "Premium", "Basic", "Premium", "Basic",
            "Premium", "Basic", "Premium", "Basic", "Premium"
        ]
    })

    result = train_baseline_model(
        df,
        target_column="Plan",
        task="classification",
        test_size=0.25,
        random_state=42
    )

    assert result["status"] == "success"
    assert "Age" in result["feature_columns"]
    assert "City" in result["feature_columns"]

    preprocessing = result["model_configuration"]["preprocessing"]

    assert "one_hot_encoding" in preprocessing["categorical_features"]
    assert "standard_scaling" in preprocessing["numeric_features"]

    assert "accuracy" in result["metrics"]
    assert "f1_weighted" in result["metrics"]



def test_regression_returns_expected_metrics():
    df = pd.DataFrame({
        "Area": [
            500, 600, 700, 800, 900, 1000, 1100, 1200,
            1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
        ],
        "Bedrooms": [
            1, 1, 2, 2, 2, 2, 3, 3,
            3, 3, 3, 4, 4, 4, 4, 5
        ],
        "Price": [
            20, 25, 30, 35, 40, 45, 50, 55,
            60, 65, 70, 75, 80, 85, 90, 95
        ]
    })

    result = train_baseline_model(
        df,
        target_column="Price",
        task="regression",
        test_size=0.25,
        random_state=42
    )

    assert result["status"] == "success"
    assert result["task"] == "regression"
    assert result["target_column"] == "Price"
    assert result["train_rows"] == 12
    assert result["test_rows"] == 4

    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]

    assert result["model_configuration"]["algorithm"] == "Ridge"


def test_regression_rejects_non_numeric_target():
    df = pd.DataFrame({
        "Age": [21, 25, 32, 28, 41, 36],
        "Plan": ["Basic", "Premium", "Basic",
                 "Premium", "Basic", "Premium"]
    })

    try:
        train_baseline_model(
            df,
            target_column="Plan",
            task="regression",
            test_size=0.25,
            random_state=42
        )
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError(
            "Regression should reject a non-numeric target."
        )
