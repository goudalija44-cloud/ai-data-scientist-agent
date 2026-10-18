
import pandas as pd


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check common data-quality issues without modifying the dataset.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    missing = df.isna().sum()
    missing_columns = {
        str(column): int(count)
        for column, count in missing.items()
        if count > 0
    }

    duplicate_count = int(df.duplicated().sum())

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    potential_id_columns = [
        str(column)
        for column in df.columns
        if (
            df[column].notna().all()
            and df[column].is_unique
            and (
                "id" in str(column).lower()
                or str(column).lower().endswith("_key")
            )
        )
    ]

    findings = []

    if missing_columns:
        findings.append({
            "issue": "missing_values",
            "severity": "warning",
            "details": missing_columns,
        })

    if duplicate_count > 0:
        findings.append({
            "issue": "duplicate_rows",
            "severity": "warning",
            "details": {"count": duplicate_count},
        })

    if constant_columns:
        findings.append({
            "issue": "constant_columns",
            "severity": "info",
            "details": {"columns": constant_columns},
        })

    if potential_id_columns:
        findings.append({
            "issue": "potential_identifier_columns",
            "severity": "info",
            "details": {"columns": potential_id_columns},
        })

    return {
        "rows_checked": int(len(df)),
        "columns_checked": int(len(df.columns)),
        "missing_value_columns": missing_columns,
        "duplicate_row_count": duplicate_count,
        "constant_columns": constant_columns,
        "potential_identifier_columns": potential_id_columns,
        "findings": findings,
    }
