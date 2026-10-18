
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a reliable summary of a Pandas DataFrame.

    Returns dataset dimensions, column types, missing values,
    duplicate-row count, and numerical summaries.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    missing_counts = df.isna().sum()

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            str(column): int(count)
            for column, count in missing_counts.items()
        },
        "total_missing_values": int(missing_counts.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_summary": {},
        "categorical_summary": {},
    }

    numerical_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    if not numerical_df.empty:
        profile["numerical_summary"] = (
            numerical_df.describe().round(2).to_dict()
        )

    for column in categorical_df.columns:
        profile["categorical_summary"][str(column)] = {
            "unique_values": int(df[column].nunique(dropna=True)),
            "most_frequent": (
                str(df[column].mode(dropna=True).iloc[0])
                if not df[column].mode(dropna=True).empty
                else None
            ),
        }

    return profile
