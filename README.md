# AI Data Scientist Agent

An end-to-end machine learning platform for exploring datasets, identifying data-quality issues, generating visual insights, and training baseline ML models through a web interface.

## Live Demo

- **Frontend:** [Try the AI Data Scientist Agent](https://ai-data-scientist-agent-frontend-my5quuoje-goudalija44-cloud.vercel.app/)
- **Backend API:** [Render API](https://ai-data-scientist-agent-api.onrender.com)
- **Health Check:** [Check API Status](https://ai-data-scientist-agent-api.onrender.com/health)
- **GitHub Repository:** [View Source Code](https://github.com/goudalija44-cloud/ai-data-scientist-agent)

## Features

- Upload CSV datasets for analysis.
- Inspect dataset structure, columns, data types, and summary statistics.
- Identify data-quality issues and missing values.
- Generate visualizations to explore data distributions and relationships.
- Run baseline machine learning workflows for classification and regression.
- Evaluate regression models using MAE, RMSE, and R².
- Evaluate classification models using relevant classification metrics and a confusion matrix.
- Access analysis and training functionality through a FastAPI backend.
- Use a browser-based frontend connected to a separately deployed API.
- Validate uploaded data and handle invalid inputs.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Data analysis | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| API testing | HTTPX, Pytest |
| Frontend hosting | Vercel |
| Backend hosting | Render |
| Version control | Git, GitHub |

## Architecture

1. **Frontend:** The user uploads a CSV file and selects an analysis or training task.
2. **API:** The frontend sends the request to the FastAPI backend.
3. **Data processing:** The backend validates the uploaded data and performs the requested workflow.
4. **Machine learning:** Supported classification or regression workflows train and evaluate baseline models.
5. **Results:** The frontend presents analysis summaries, metrics, and visualizations.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check API health |
| `/analyze` | POST | Analyze an uploaded dataset |
| `/train` | POST | Run a supported model-training workflow |

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/goudalija44-cloud/ai-data-scientist-agent.git
cd ai-data-scientist-agent
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Open the frontend

Serve the `frontend` directory using a local development server, such as the VS Code Live Server extension. Open the local frontend URL in your browser.

## Testing

Run the automated test suite with:

```bash
python -m pytest -q
```

The project previously passed 44 tests. Run the command again after changes to verify the current version.

## Deployment

- The frontend is hosted on Vercel.
- The FastAPI backend is hosted on Render.
- Cross-Origin Resource Sharing (CORS) is configured to allow the deployed frontend origin.
- The frontend communicates with the backend through HTTP API requests.

## Project Goals

This project demonstrates practical skills in:

- Data analysis and preprocessing
- Data-quality validation
- Baseline machine learning
- Model evaluation
- REST API development
- Frontend-to-backend integration
- Automated testing
- Cloud deployment and Git-based development

## Future Improvements

- Add more configurable machine learning algorithms.
- Improve automated model selection and comparison.
- Add explainability tools for model predictions.
- Introduce experiment tracking and downloadable reports.
- Expand testing and production monitoring.

## Disclaimer

This project provides exploratory data analysis and baseline machine learning workflows. Model results depend on dataset quality, feature selection, and the suitability of the selected algorithm. It is intended for learning and demonstration, not as a substitute for expert review in high-stakes applications.