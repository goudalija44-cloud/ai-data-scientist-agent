
from pathlib import Path

from src.data_loader import load_csv
from src.analysis_pipeline import run_analysis

def analyze_csv_file(file_path: str | Path) -> dict:
  """
  Load a csv file and return a complete analysis report.

  file-loading and analysis errrors are raised to the caller.
  """

  df = load_csv(file_path)
  report = run_analysis(df)

  report["source_file"] = Path(file_path).name
  return report
