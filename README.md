# AI Data Scientist Agent

An interactive data analysis application that helps users explore CSV datasets, identify data-quality issues, discover patterns, visualize results, and train baseline machine learning models.

The project combines data analysis, machine learning, and a web-based interface, with plans to add AI-assisted explanations grounded in computed evidence.

## Features

### Data Analysis
- Automated dataset profiling
- Missing-value and duplicate detection
- Constant-column and potential identifier detection
- Numeric and categorical statistical summaries
- Evidence-based analytical insights
- Correlation analysis

### Data Visualization
- Interactive data exploration through a web interface
- Numeric distribution charts
- Categorical breakdown charts
- Correlation heatmaps using Plotly

### Machine Learning
- Baseline classification using Logistic Regression
- Baseline regression using Ridge Regression
- Numeric and categorical preprocessing
- Missing-value imputation
- Feature scaling and categorical encoding
- Classification and regression evaluation metrics
- Confusion matrix and per-class classification metrics
- Warnings about limitations on small datasets

### API and Interface
- FastAPI backend
- CSV upload and analysis endpoint
- Machine learning training endpoint
- HTML, CSS, and JavaScript frontend

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Pandas and NumPy | Data processing and numerical analysis |
| Scikit-learn | Preprocessing and baseline machine learning |
| FastAPI | Backend API |
| Uvicorn | Local API server |
| HTML, CSS, JavaScript | Frontend interface |
| Plotly | Interactive visualizations |
| Pytest | Automated testing |

## Project Structure

```text
ai-data-scientist-agent/
├── data/
│   ├── sample_customers.csv
│   └── sample_ml_classification.csv
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── notebooks/
│   └── AI_Data_Scientist_Agent_Phase1.ipynb
├── src/
│   ├── analysis_engine.py
│   ├── analysis_pipeline.py
│   ├── api.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_quality.py
│   ├── generate_report.py
│   ├── insights_engine.py
│   ├── inspect_dataset.py
│   ├── ml_engine.py
│   ├── run_dataset_analysis.py
│   ├── run_full_analysis.py
│   └── run_ml_experiment.py
├── tests/
├── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python installed on your system
- Git
- Visual Studio Code or another Python-compatible editor

### 1. Clone the repository

```bash
git clone https://github.com/goudalija44-cloud/ai-data-scientist-agent.git
cd ai-data-scientist-agent
```

### 2. Create a virtual environment

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

If the `python` command is unavailable, use your installed Python executable's full path.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the repository root:

```bash
python -m uvicorn src.api:app --reload
```

The API will normally be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 5. Open the frontend

With the backend running, open `frontend/index.html` in your browser or serve the `frontend` directory using a local static web server.

If the browser blocks API requests, check the backend's CORS configuration and serve the frontend from a local HTTP origin permitted by the API.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/analyze` | Upload and analyze a CSV dataset |
| POST | `/train` | Train and evaluate a baseline ML model |

The `/train` endpoint accepts a CSV file, target column, task type, test size, and random state.

## Running Tests

From the project root, run:

```bash
python -m pytest -q
```

The current development test run passed **44 tests**. Re-run the tests after making changes to confirm the current result.

## Current Project Status

**Implemented in the current development version**
- CSV loading and validation
- Dataset profiling and data-quality checks
- Statistical analysis and analytical insights
- Visualization data and frontend charts
- Baseline classification and regression
- FastAPI endpoints
- Automated tests

**Planned improvements**
- AI-assisted explanations grounded in computed results
- More robust model evaluation and experiment tracking
- Additional datasets and edge-case testing
- Deployment and a publicly accessible demo
- Improved documentation, screenshots, and portfolio presentation

## Limitations

- Baseline models are intended for initial evaluation, not production use.
- Results depend on dataset quality, target selection, feature suitability, and dataset size.
- Model metrics alone do not establish that a model generalizes to unseen real-world data.
- The planned AI-assisted explanation layer and public deployment should not be considered complete until implemented and tested.

## Disclaimer

This project is under development. Features and results may change as the application evolves. Verify analytical findings and model performance before using them to make consequential decisions.
