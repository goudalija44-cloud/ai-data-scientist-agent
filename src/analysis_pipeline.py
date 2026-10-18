
import pandas as pd

from src.data_profiler import profile_dataset
from src.data_quality import check_data_quality
from src.analysis_engine import analyze_dataset
from src.insights_engine import generate_insights


def run_analysis(df: pd.DataFrame) -> dict:
    """
    Run the complete dataset analysis pipeline.

    This function does not modify the input DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    return {
        "status": "success",
        "dataset_profile": profile_dataset(df),
        "data_quality": check_data_quality(df),
        "statistical_analysis": analyze_dataset(df),
        "insights": generate_insights(df),
    }
