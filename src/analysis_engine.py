import pandas as pd


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Calculate verified descriptive statistics and correlations."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    numeric_df = df.select_dtypes(include="number")
    results = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_df.columns.tolist(),
        "numeric_statistics": {},
        "categorical_statistics": {},
        "correlations": {},
    }

    # Calculate numeric statistics while excluding missing values.
    for column in numeric_df.columns:
        series = numeric_df[column]
        results["numeric_statistics"][str(column)] = {
            "count": int(series.count()),
            "mean": (
                round(float(series.mean()), 4)
                if series.count() else None
            ),
            "median": (
                round(float(series.median()), 4)
                if series.count() else None
            ),
            "min": (
                float(series.min())
                if series.count() else None
            ),
            "max": (
                float(series.max())
                if series.count() else None
            ),
            "standard_deviation": (
                round(float(series.std()), 4)
                if series.count() > 1 else None
            ),
        }

    # Summarize categorical columns.
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    for column in categorical_df.columns:
        counts = df[column].value_counts(dropna=True)
        results["categorical_statistics"][str(column)] = {
            "unique_values": int(df[column].nunique(dropna=True)),
            "top_value": str(counts.index[0]) if not counts.empty else None,
            "top_value_count": int(counts.iloc[0]) if not counts.empty else 0,
        }

    # Pearson correlations are only calculated between numeric columns.
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr(method="pearson")
        results["correlations"] = {
            str(col): {
                str(other): (
                    round(float(corr.loc[col, other]), 4)
                    if pd.notna(corr.loc[col, other]) else None
                )
                for other in corr.columns
            }
            for col in corr.columns
        }

    return results
