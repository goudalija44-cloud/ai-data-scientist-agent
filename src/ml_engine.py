import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge 
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_baseline_model(
        df: pd.DataFrame,
        target_column: str,
        task: str,
        test_size: float = 0.2,
        random_state: int = 42
) -> dict:
    """Train and evaluate a simple classification or regression model."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    if task not in {"classification", "regression"}:
        raise ValueError(
            "task must be 'classification' or 'regression'."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if df[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values."
            "Clean or remove these rows before training."
        )


    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    # Avoid using an identifier as a predictive feature.
    identifier_columns = [
        col for col in X.columns
        if "id" in str(col).lower()
        and(
            X[col].nunique(dropna=True) == len(X)
            or str(col).lower() == "id"
            or str(col).lower().endswith("_id")
        )
    ]

    X = X.drop(columns=identifier_columns)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable features remain after excluding identifier columns."
        )

    # Convert numeric-looking data only when its actual dtype is numeric

    numeric_columns = X.select_dtypes(include="number").columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]


    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target.")

    else:
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

    if len(df) < 5:
        raise ValueError(
            "At least 5 rows are required for a baseline train/test split."
        )    

    # Stratification is used for classification only when every class has
    # enough examples to appear in both training and test partitions.

    stratify = None
    if task == "classification":
        class_counts = y.value_counts()
        n_test = max(1, int(len(y) * test_size))
        n_train = len(y) - n_test
        if(
            class_counts.min() >= 2
            and n_test >= y.nunique()
            and n_train >= y.nunique()
        ):
            stratify = y 

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    except ValueError as exc:
        raise ValueError(f"Unable to create train/test split: {exc}")

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    estimator = (
        LogisticRegression(max_iter=1000)
        if task == "classification"
        else Ridge()
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except ValueError as exc:
        raise ValueError(f"Model training failed: {exc}") from exc 

    
    if task == "classification":
        labels = model.named_steps["estimator"].classes_

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                labels=labels,
                zero_division=0,
            )
        )

        class_metrics = [
            {
                "class_label": str(label),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(score), 4),
                "support": int(count),
            }
            for label, p, r, score, count in zip(
                labels, precision, recall, f1, support
            )
        ]

        metrics = {
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "f1_weighted": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                4,
            ),
        }

        evaluation_details = {
            "class_metrics": class_metrics,
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels
            ).tolist(),
            "confusion_matrix_labels": [
                str(label) for label in labels
            ],
        }

    else:
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "mae": round(
                float(mean_absolute_error(y_test, predictions)), 4
            ),
            "rmse": round(float(mse ** 0.5), 4),
            "r2": (
                round(float(r2_score(y_test, predictions)), 4)
                if len(y_test) >= 2
                else None
            ),
        }
        evaluation_details = {}

    return {
        "status": "success",
        "task": task,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "excluded_identifier_columns": identifier_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        **evaluation_details,
        "evaluation": {
            "method": "train_test_split",
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "metrics_reported": (
                ["accuracy", "f1_weighted", "per_class_precision",
                 "per_class_recall", "per_class_f1", "confusion_matrix"]
                 if task == "classification"
                 else ["mae", "rmse", "r2"]
            ),
        },
        "model_configuration":{
            "algorithm":(
                "LogisticRegression"
                if task == "classification"
                else "Ridge"
            ),
            "preprocessing": {
                "numeric_features": (
                    ["median_imputation", "standard_scaling"]
                    if numeric_columns else []
                ),
                "categorical_features": (
                    ["most_frequent_imputation", "one_hot_encoding"]
                    if categorical_columns else []
                ),
            },
        },
        "warning": (
            "This is a baseline model. Test performance on a small dataset "
            "is unstable and should not be treated as evidence of "
            "real-world predictive performance."
        ),
    }





            



    