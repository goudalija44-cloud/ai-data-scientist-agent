
from pathlib import Path
import pandas as pd

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

def load_csv(file_path: str | Path) -> pd.DataFrame:
  """Load and validate a CSV file without modifying its contents."""

  path = Path(file_path)

  if not path.exists():
    raise FileNotFoundError(f"File Not Found: {file_path}")

  if not path.is_file():
    raise ValueError("The provided path is not a file.")

  if path.suffix.lower() != '.csv':
    raise ValueError("Only CSV files are supported.")

  if path.stat().st_size > MAX_FILE_SIZE_BYTES:
    raise ValueError("File size exceeds the limit of 50 MB.")

  if path.stat().st_size == 0:
    raise ValueError("File is empty.")

  try:
    df = pd.read_csv(file_path)
  except pd.errors.EmptyDataError as exc:
    raise ValueError("CSV contains no tabular data.") from exc
  except (pd.errors.ParserError, UnicodeDecodeError) as exc:
    raise ValueError("CSV file could not be parsed.") from exc

  if df.empty:
    raise ValueError("CSV contains headers but no data rows.")

  if len(df.columns) == 0:
    raise ValueError("CSV contains no columns.")

  if df.columns.duplicated().any():
    raise ValueError("CSV contains duplicate column names.")

  if any(not str(col).strip() for col in df.columns):
    raise ValueError("CSV contains an empty column name.")

  return df
