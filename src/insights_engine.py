
import pandas as pd


def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generate simple, evidence-backed descriptive insights.
    Does not infer causation or make predictions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    insights = []

    # Dataset size
    insights.append({
        "type": "dataset_overview",
        "message": f"Dataset contains {len(df)} rows and {len(df.columns)} columns.",
        "evidence": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
    })

    # Missing values
    missing = df.isna().sum()
    for column, count in missing.items():
        if count > 0:
            insights.append({
                "type": "missing_values",
                "message": f"Column '{column}' contains {int(count)} missing values.",
                "evidence": {
                    "column": str(column),
                    "missing_count": int(count),
                    "missing_percentage": round(
                        float(count / len(df) * 100), 2
                    ),
                },
            })

    # Numerical patterns
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            continue

        if series.nunique() == 1:
            insights.append({
                "type": "constant_numeric_column",
                "message": f"Column '{column}' has only one observed numeric value.",
                "evidence": {
                    "column": str(column),
                    "value": float(series.iloc[0]),
                },
            })
            continue

        mean_value = float(series.mean())
        median_value = float(series.median())

        if mean_value > median_value:
            pattern = "mean_above_median"
        elif mean_value < median_value:
            pattern = "mean_below_median"
        else:
            pattern = "mean_equals_median"

        insights.append({
            "type": "numeric_distribution",
            "message": (
                f"Column '{column}': mean is {mean_value:.2f} "
                f"and median is {median_value:.2f}."
            ),
            "evidence": {
                "column": str(column),
                "count": int(series.count()),
                "mean": round(mean_value, 4),
                "median": round(median_value, 4),
                "pattern": pattern,
                "min": float(series.min()),
                "max": float(series.max()),
            },
        })

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        insights.append({
            "type": "duplicate_rows",
            "message": f"Dataset contains {duplicate_count} duplicate rows.",
            "evidence": {
                "duplicate_count": duplicate_count,
            },
        })

    return {
        "insight_count": len(insights),
        "insights": insights,
    }
