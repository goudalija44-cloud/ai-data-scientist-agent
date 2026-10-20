
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.data_loader import load_csv
from src.ml_engine import train_baseline_model
from src.run_full_analysis import run_full_analysis


app = FastAPI(
    title="AI Data Scientist Agent",
    description="API for CSV dataset inspection, analysis, and ML experiments.",
    version="0.2.0",
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://127.0.0.1:5500",
#         "http://localhost:5500",
#     ],
#     allow_credentials=False,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://ai-data-scientist-agent-frontend-my5quuoje-goudalija44-cloud.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def validate_csv_upload(filename, contents):
    """Validate a CSV upload before processing."""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The upload exceeds the 50 MB limit.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "AI Data Scientist Agent",
    }


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        filename = file.filename or ""
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        validate_csv_upload(filename, contents)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "uploaded_dataset.csv"
            csv_path.write_bytes(contents)
            report = run_full_analysis(csv_path)

        return JSONResponse(content=report)

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The dataset could not be analyzed.",
        ) from exc
    finally:
        await file.close()


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task: str = Form(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
):
    try:
        filename = file.filename or ""
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        validate_csv_upload(filename, contents)

        if task not in {"classification", "regression"}:
            raise HTTPException(
                status_code=400,
                detail="Task must be classification or regression.",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "uploaded_dataset.csv"
            csv_path.write_bytes(contents)

            df = load_csv(csv_path)

            result = train_baseline_model(
                df=df,
                target_column=target_column,
                task=task,
                test_size=test_size,
                random_state=random_state,
            )

        result["dataset_name"] = Path(filename).stem
        result["source_file"] = Path(filename).name

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The machine-learning experiment could not be completed.",
        ) from exc
    finally:
        await file.close()
