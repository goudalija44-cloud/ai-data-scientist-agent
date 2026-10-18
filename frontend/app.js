const API_BASE_URL = "http://127.0.0.1:8000";
const MAX_FILE_SIZE = 50 * 1024 * 1024;

// DOM elements
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const removeFileButton = document.getElementById("removeFile");
const analyzeButton = document.getElementById("analyzeButton");
const uploadMessage = document.getElementById("uploadMessage");
const apiStatus = document.getElementById("apiStatus");

const resultsSection = document.getElementById("resultsSection");
const datasetName = document.getElementById("datasetName");
const overviewCards = document.getElementById("overviewCards");
const columnsContainer = document.getElementById("columnsContainer");
const qualityContainer = document.getElementById("qualityContainer");
const insightsContainer = document.getElementById("insightsContainer");
const rawReport = document.getElementById("rawReport");
const downloadButton = document.getElementById("downloadButton");

let currentReport = null;


let currentFile = null;

const mlExperimentForm = document.getElementById("mlExperimentForm");
const mlTargetColumn = document.getElementById("mlTargetColumn");
const mlTask = document.getElementById("mlTask");
const mlTestSize = document.getElementById("mlTestSize");
const runMLButton = document.getElementById("runMLButton");
const mlStatus = document.getElementById("mlStatus");
const mlResults = document.getElementById("mlResults");
const mlMetrics = document.getElementById("mlMetrics");
const mlDetails = document.getElementById("mlDetails");


// Safely create an element and insert plain text.
function createElement(tag, className = "", text = "") {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    element.textContent = String(text);
    return element;
}

function showMessage(message, type = "") {
    uploadMessage.textContent = message;
    uploadMessage.className = "message";

    if (type) {
        uploadMessage.classList.add(type);
    }
}

function setApiStatus(connected) {
    apiStatus.textContent = connected ? "API connected" : "API unavailable";

    const statusDot = document.querySelector(".status-dot");

    if (statusDot) {
        statusDot.style.background = connected ? "#18845b" : "#c0392b";
    }
}

function updateFileDisplay() {
    const file = fileInput.files[0];

    selectedFile.hidden = !file;
    fileName.textContent = file ? file.name : "";
}

function validateFile(file) {
    if (!file) {
        return "Please select a CSV file first.";
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        return "Please select a file with the .csv extension.";
    }

    if (file.size === 0) {
        return "The selected file is empty.";
    }

    if (file.size > MAX_FILE_SIZE) {
        return "The file exceeds the 50 MB limit.";
    }

    return null;
}

async function checkApiHealth() {
    apiStatus.textContent = "Checking API...";

    try {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.status}`);
        }

        const result = await response.json();

        if (result.status !== "healthy") {
            throw new Error("API did not report a healthy status.");
        }

        setApiStatus(true);
    } catch (error) {
        setApiStatus(false);
        console.error("API health check failed:", error);
    }
}

// Safely format numbers for display.
function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    if (typeof value === "number" && Number.isFinite(value)) {
        return value.toLocaleString();
    }

    return String(value);
}

// Render a single overview metric.
function renderMetric(label, value) {
    const card = createElement("article", "metric-card");

    card.appendChild(createElement("div", "metric-label", label));
    card.appendChild(
        createElement("div", "metric-value", formatNumber(value))
    );

    return card;
}

// Get the column information regardless of whether it is an array or object.
function normalizeColumns(columns) {
    if (Array.isArray(columns)) {
        return columns.map((column, index) => {
            if (typeof column === "string") {
                return { name: column };
            }

            return {
                ...(column && typeof column === "object" ? column : {}),
                name:
                    column?.name ??
                    column?.column_name ??
                    `Column ${index + 1}`,
            };
        });
    }

    if (columns && typeof columns === "object") {
        return Object.entries(columns).map(([name, details]) => ({
            name,
            ...(details && typeof details === "object" ? details : {}),
        }));
    }

    return [];
}

// Calculate missing values using the available report fields.
function getTotalMissingValues(report) {
    const candidates = [
        report?.missing_values_summary?.total_missing_values,
        report?.analysis?.dataset_profile?.total_missing_values,
        report?.dataset_profile?.total_missing_values,
    ];

    for (const value of candidates) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return value;
        }
    }

    const columns = normalizeColumns(
        report?.dataset_overview?.columns ??
        report?.analysis?.dataset_profile?.missing_values
    );

    if (columns.length > 0) {
        return columns.reduce((total, column) => {
            const value =
                column.missing_values ??
                column.missing ??
                column.count ??
                0;

            return total + (Number(value) || 0);
        }, 0);
    }

    return 0;
}

// Render dataset overview cards.
function renderOverview(report) {
    overviewCards.replaceChildren();

    const overview = report.dataset_overview || {};
    const analysis = report.analysis || {};
    const profile = analysis.dataset_profile || {};

    const columns = normalizeColumns(
        overview.columns ?? profile.column_names
    );

    const totalColumns =
        overview.columns_count ??
        overview.columns ??
        profile.columns ??
        columns.length;

    overviewCards.append(
        renderMetric("Total rows", overview.rows ?? profile.rows),
        renderMetric("Total columns", totalColumns),
        renderMetric("Missing values", getTotalMissingValues(report))
    );
}

// Render dataset columns in a table.
function renderColumns(columns) {
    columnsContainer.replaceChildren();

    const normalizedColumns = normalizeColumns(columns);

    if (normalizedColumns.length === 0) {
        columnsContainer.textContent = "No column information is available.";
        return;
    }

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    ["Column", "Data Type", "Missing Values", "Unique Values"].forEach(
        (label) => {
            headerRow.appendChild(createElement("th", "", label));
        }
    );

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    normalizedColumns.forEach((column, index) => {
        const row = document.createElement("tr");

        const values = [
            column.name ?? `Column ${index + 1}`,
            column.dtype ?? column.data_type ?? "Unknown",
            column.missing_values ?? column.missing ?? 0,
            column.unique_values ?? column.unique ?? "—",
        ];

        values.forEach((value) => {
            row.appendChild(createElement("td", "", formatNumber(value)));
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);

    const wrapper = createElement("div", "table-wrapper");
    wrapper.appendChild(table);
    columnsContainer.appendChild(wrapper);
}

// Safely display an individual data quality finding.
function renderFinding(finding, fallbackTitle = "Finding") {
    const card = createElement("div", "finding");

    if (typeof finding === "string") {
        card.appendChild(createElement("p", "", finding));
        return card;
    }

    if (!finding || typeof finding !== "object") {
        card.appendChild(createElement("p", "", String(finding)));
        return card;
    }

    const title =
        finding.title ??
        finding.type ??
        finding.category ??
        finding.severity ??
        fallbackTitle;

    card.appendChild(createElement("strong", "", title));

    const details =
        finding.details ??
        finding.description ??
        finding.message ??
        finding.explanation;

    if (details !== undefined) {
        card.appendChild(createElement("p", "", details));
    } else {
        const detailEntries = Object.entries(finding).filter(
            ([key]) =>
                !["title", "type", "category", "severity"].includes(key)
        );

        if (detailEntries.length > 0) {
            const detailsList = document.createElement("ul");

            detailEntries.forEach(([key, value]) => {
                const readableValue =
                    value && typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value);

                detailsList.appendChild(
                    createElement("li", "", `${key}: ${readableValue}`)
                );
            });

            card.appendChild(detailsList);
        }
    }

    return card;
}

// Render data quality results.
function renderQuality(quality) {
    qualityContainer.replaceChildren();

    if (!quality || typeof quality !== "object") {
        qualityContainer.textContent =
            "Data quality information is unavailable.";
        return;
    }

    const duplicateRows = quality.duplicate_rows ?? 0;

    const missingValues =
        quality.missing_values &&
        typeof quality.missing_values === "object"
            ? quality.missing_values
            : {};

    const missingCount = Object.values(missingValues).reduce(
        (total, value) => total + (Number(value) || 0),
        0
    );

    const constantColumns = Array.isArray(quality.constant_columns)
        ? quality.constant_columns
        : [];

    const identifierColumns = Array.isArray(
        quality.potential_identifier_columns
    )
        ? quality.potential_identifier_columns
        : [];

    const summary = createElement(
        "p",
        "",
        `Duplicate rows: ${formatNumber(duplicateRows)} | ` +
        `Columns with missing values: ${formatNumber(
            Object.keys(missingValues).length
        )} | ` +
        `Total missing values: ${formatNumber(missingCount)} | ` +
        `Constant columns: ${formatNumber(constantColumns.length)}`
    );

    qualityContainer.appendChild(summary);

    const findings = Array.isArray(quality.findings)
        ? quality.findings
        : [];

    findings.forEach((finding, index) => {
        qualityContainer.appendChild(
            renderFinding(finding, `Quality finding ${index + 1}`)
        );
    });

    if (Object.keys(missingValues).length > 0) {
        qualityContainer.appendChild(
            createElement("h3", "", "Columns with missing values")
        );

        Object.entries(missingValues).forEach(([name, count]) => {
            qualityContainer.appendChild(
                createElement(
                    "p",
                    "",
                    `${name}: ${formatNumber(count)} missing value(s)`
                )
            );
        });
    }

    if (constantColumns.length > 0) {
        qualityContainer.appendChild(
            createElement(
                "p",
                "",
                `Constant columns: ${constantColumns.join(", ")}`
            )
        );
    }

    if (identifierColumns.length > 0) {
        qualityContainer.appendChild(
            createElement(
                "p",
                "",
                `Potential identifier columns: ${identifierColumns.join(", ")}`
            )
        );
    }

    if (
        findings.length === 0 &&
        Object.keys(missingValues).length === 0 &&
        constantColumns.length === 0 &&
        identifierColumns.length === 0 &&
        Number(duplicateRows) === 0
    ) {
        qualityContainer.appendChild(
            createElement(
                "p",
                "",
                "No issues were reported by the data quality checks."
            )
        );
    }
}

// Render generated insights.
function renderInsights(insightsData) {
    insightsContainer.replaceChildren();

    let insights = [];

    if (Array.isArray(insightsData)) {
        insights = insightsData;
    } else if (
        insightsData &&
        typeof insightsData === "object" &&
        Array.isArray(insightsData.insights)
    ) {
        insights = insightsData.insights;
    }

    if (insights.length === 0) {
        insightsContainer.textContent =
            "No insights were generated for this dataset.";
        return;
    }

    insights.forEach((insight, index) => {
        insightsContainer.appendChild(
            renderFinding(insight, `Insight ${index + 1}`)
        );
    });
}

// Render the complete API report.
function renderReport(report) {
    if (!report || typeof report !== "object") {
        throw new Error("The API returned an invalid report.");
    }

    currentReport = report;

    populateMLTargets(report);

    datasetName.textContent =
        `Source: ${
            report.source_file ??
            report.dataset_name ??
            "Uploaded dataset"
        }`;

    const analysis = report.analysis || {};

    // Pass each renderer the correct section of the report.
    renderOverview(report);

    renderColumns(
        report.dataset_overview?.columns ??
        analysis.dataset_profile?.column_names ??
        analysis.dataset_profile?.data_types
    );

    renderQuality(analysis.data_quality);

    renderInsights(analysis.insights);

    initializeVisualizations(report);

    rawReport.textContent = JSON.stringify(report, null, 2);
    resultsSection.hidden = false;

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}

// Submit a CSV file to the API.
uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    const validationError = validateFile(file);

    if (validationError) {
        showMessage(validationError, "error");
        return;
    }

    currentFile = file;

    const formData = new FormData();
    formData.append("file", file);

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing dataset...";
    showMessage("Uploading your file and analyzing its contents...");

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            const detail =
                typeof result.detail === "string"
                    ? result.detail
                    : "The dataset could not be analyzed.";

            throw new Error(detail);
        }

        renderReport(result);
        showMessage("Analysis completed successfully.", "success");
    } catch (error) {
        console.error("Dataset analysis failed:", error);

        showMessage(
            error.message === "Failed to fetch"
                ? "Cannot reach the API. Make sure FastAPI is running at 127.0.0.1:8000."
                : error.message || "An unexpected error occurred.",
            "error"
        );
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.replaceChildren(
            document.createTextNode("Analyze dataset "),
            createElement("span", "", "→")
        );
    }
});

// Update the selected file display.
fileInput.addEventListener("change", () => {
    updateFileDisplay();

    const file = fileInput.files[0];

    currentFile = file || null;

    if (runMLButton) {
        runMLButton.disabled = !currentFile || !currentReport;
    }

    if (!file) {
        showMessage("");
        return;
    }

    const validationError = validateFile(file);

    if (validationError) {
        showMessage(validationError, "error");
    } else {
        showMessage("File selected. Ready to analyze.", "success");
    }
});

// Remove the selected file.
removeFileButton.addEventListener("click", () => {
    fileInput.value = "";
    currentFile = null;

    if (runMLButton) runMLButton.disabled = true;
    if (mlResults) mlResults.hidden = true;
    updateFileDisplay();
    showMessage("File removed.");
});

// Drag-and-drop support.
uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    const validationError = validateFile(file);

    if (validationError) {
        showMessage(validationError, "error");
        return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;

    updateFileDisplay();
    showMessage("File selected. Ready to analyze.", "success");
});

// Download the full report as JSON.
downloadButton.addEventListener("click", () => {
    if (!currentReport) {
        showMessage("There is no report available to download.", "error");
        return;
    }

    const blob = new Blob(
        [JSON.stringify(currentReport, null, 2)],
        { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "dataset_analysis_report.json";

    document.body.appendChild(link);
    link.click();
    link.remove();

    // Release the temporary download URL after the click.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// Check whether the backend is running.
checkApiHealth();



function initializeVisualizations(report) {
    const data = report.visualization_data;

    if (!data || typeof Plotly === "undefined") {
        console.warn(
            "Visualization data or Plotly is unavailable."
        );
        return;
    }

    const histogramSelect = document.getElementById("histogramColumn");
    const categorySelect = document.getElementById("categoryColumn");

    const histogramElement = document.getElementById("histogramChart");
    const categoryElement = document.getElementById("categoryChart");
    const correlationElement = document.getElementById("correlationChart");

    if (
        !histogramSelect ||
        !categorySelect ||
        !histogramElement ||
        !categoryElement ||
        !correlationElement
    ) {
        console.warn("One or more chart HTML elements are missing.");
        return;
    }

    // Populate the numerical column selector.
    histogramSelect.replaceChildren(
        new Option("Choose a column", "")
    );

    (data.numerical_columns || []).forEach((column) => {
        histogramSelect.add(new Option(column, column));
    });

    // Populate the categorical column selector.
    categorySelect.replaceChildren(
        new Option("Choose a column", "")
    );

    (data.categorical_columns || []).forEach((column) => {
        categorySelect.add(new Option(column, column));
    });

    // Draw a histogram using server-computed bin counts.
    function drawHistogram(column) {
        const histogram = data.histograms?.[column];

        if (
            !histogram ||
            !histogram.counts?.length ||
            !histogram.bin_edges?.length
        ) {
            Plotly.purge(histogramElement);
            return;
        }

        const labels = histogram.counts.map((_, index) => {
            const start = histogram.bin_edges[index];
            const end = histogram.bin_edges[index + 1];

            return `${start} – ${end}`;
        });

        Plotly.react(
            histogramElement,
            [{
                type: "bar",
                x: labels,
                y: histogram.counts,
                hovertemplate:
                    "Range: %{x}<br>Rows: %{y}<extra></extra>"
            }],
            {
                title: `${column} Distribution`,
                xaxis: { title: column, automargin: true },
                yaxis: { title: "Number of rows", rangemode: "tozero" },
                height: 280,
                margin: { t: 45, r: 15, b: 65, l: 50 },
                autosize: true
            },
            { responsive: true }
        );
    }

    // Draw category frequencies as a bar chart.
    function drawCategories(column) {
        const categories = data.category_counts?.[column];

        if (!categories?.length) {
            Plotly.purge(categoryElement);
            return;
        }

        Plotly.react(
            categoryElement,
            [{
                type: "bar",
                x: categories.map((item) => item.label),
                y: categories.map((item) => item.count),
                hovertemplate:
                    "Category: %{x}<br>Rows: %{y}<extra></extra>"
            }],
            {
                title: `${column} Category Breakdown`,
                xaxis: { title: column, automargin: true },
                yaxis: { title: "Number of rows", rangemode: "tozero" },
                height: 280,
                margin: { t: 45, r: 15, b: 65, l: 50 },
                autosize: true
            },
            { responsive: true }
        );
    }

    // Redraw charts when the user changes a selection.
    histogramSelect.onchange = () => {
        drawHistogram(histogramSelect.value);
    };

    categorySelect.onchange = () => {
        drawCategories(categorySelect.value);
    };

    // Draw the correlation heatmap.
    const correlation = data.correlation;
    const matrix = correlation?.matrix || [];
    const columns = correlation?.columns || [];

    if (
        columns.length > 0 &&
        matrix.length === columns.length
    ) {
        Plotly.react(
            correlationElement,
            [{
                type: "heatmap",
                x: columns,
                y: columns,
                z: matrix.map((row) =>
                    row.map((value) =>
                        value === null ? NaN : value
                    )
                ),
                zmin: -1,
                zmax: 1,
                colorscale: "RdBu",
                reversescale: true,
                colorbar: { title: "Correlation" },
                hovertemplate:
                    "%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>"
            }],
            {
                title: "Numerical Feature Correlations",
                xaxis: { side: "bottom", automargin: true },
                yaxis: { autorange: "reversed", automargin: true },
                height: 340,
                margin: { t: 45, r: 15, b: 65, l: 50 },
                autosize: true
            },
            { responsive: true }
        );
    } else {
        correlationElement.textContent =
            "At least two numerical columns are needed for a correlation heatmap.";
    }

    // Select the first available columns by default.
    if (data.numerical_columns?.length) {
        histogramSelect.value = data.numerical_columns[0];
        drawHistogram(histogramSelect.value);
    } else {
        histogramElement.textContent =
            "No numerical columns are available for a histogram.";
    }

    if (data.categorical_columns?.length) {
        categorySelect.value = data.categorical_columns[0];
        drawCategories(categorySelect.value);
    } else {
        categoryElement.textContent =
            "No categorical columns are available for a category chart.";
    }
}



function populateMLTargets(report) {
    if (!mlTargetColumn || !runMLButton) return;

    const columns = Object.keys(
        report.dataset_overview?.columns || {}
    );

    mlTargetColumn.replaceChildren(
        new Option("Choose a target column", "")
    );

    columns.forEach((column) => {
        mlTargetColumn.add(new Option(column, column));
    });

    runMLButton.disabled = !currentFile || columns.length === 0;

    if (mlStatus) {
        mlStatus.textContent = columns.length
            ? "Choose a target and task, then run the experiment."
            : "No columns are available as targets.";
    }

    if (mlResults) mlResults.hidden = true;
}

function renderMLMetric(label, value) {
    const card = createElement("div", "ml-metric");
    card.appendChild(createElement("div", "ml-metric-label", label));
    card.appendChild(createElement("div", "ml-metric-value", value));
    return card;
}


function renderMLResults(result) {
    mlMetrics.replaceChildren();
    mlDetails.replaceChildren();

    const metrics = result.metrics || {};

    // Display evaluation metrics.
    Object.entries(metrics).forEach(([name, value]) => {
        if (typeof value === "number" || value === null) {
            const isPercentage = ["accuracy", "f1_weighted", "r2"].includes(name);

            const displayValue =
                value === null
                    ? "N/A"
                    : isPercentage
                        ? `${(value * 100).toFixed(2)}%`
                        : value.toFixed(3);

            mlMetrics.appendChild(
                renderMLMetric(
                    name.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()),
                    displayValue
                )
            );
        }
    });

    // Display experiment summary.
    const summary = document.createElement("p");
    summary.textContent =
        `Task: ${result.task || "—"} | Target: ${result.target_column || "—"} | ` +
        `Training rows: ${result.train_rows ?? "—"} | Test rows: ${result.test_rows ?? "—"}`;

    mlDetails.appendChild(summary);

    // Display model configuration.
    const configuration = result.model_configuration || {};
    const modelHeading = createElement("h4", "", "Model Configuration");
    mlDetails.appendChild(modelHeading);

    const modelInfo = document.createElement("p");
    modelInfo.textContent =
        `Algorithm: ${configuration.algorithm || "—"}`;

    mlDetails.appendChild(modelInfo);

    const evaluation = result.evaluation || {};
    const evaluationInfo = document.createElement("p");
    evaluationInfo.textContent =
        `Evaluation: ${evaluation.method || "—"} | ` +
        `Test proportion: ${
            typeof evaluation.test_size === "number"
                ? `${(evaluation.test_size * 100).toFixed(0)}%`
                : "—"
        } | Random state: ${evaluation.random_state ?? "—"}`;

    mlDetails.appendChild(evaluationInfo);

    // Display classification confusion matrix when available.
    if (
        result.task === "classification" &&
        Array.isArray(result.confusion_matrix) &&
        Array.isArray(result.confusion_matrix_labels)
    ) {
        const heading = createElement("h4", "", "Confusion Matrix");
        mlDetails.appendChild(heading);

        const wrapper = createElement("div", "ml-result-details");
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");

        const corner = document.createElement("th");
        corner.textContent = "Actual \\ Predicted";
        headerRow.appendChild(corner);

        result.confusion_matrix_labels.forEach(label => {
            const th = document.createElement("th");
            th.textContent = String(label);
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");

        result.confusion_matrix.forEach((row, index) => {
            const tr = document.createElement("tr");
            const labelCell = document.createElement("th");
            labelCell.textContent =
                String(result.confusion_matrix_labels[index] ?? `Class ${index + 1}`);

            tr.appendChild(labelCell);

            row.forEach(value => {
                const td = document.createElement("td");
                td.textContent = String(value);
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        wrapper.appendChild(table);
        mlDetails.appendChild(wrapper);

        const explanation = document.createElement("p");
        explanation.className = "chart-description";
        explanation.textContent =
            "Rows represent actual classes; columns represent predicted classes. " +
            "Diagonal values are correct predictions; off-diagonal values are errors.";

        mlDetails.appendChild(explanation);

        // Display per-class metrics.
        if (Array.isArray(result.class_metrics)) {
            mlDetails.appendChild(
                createElement("h4", "", "Per-Class Performance")
            );

            const classWrapper = createElement("div", "ml-result-details");
            const classTable = document.createElement("table");
            const classHead = document.createElement("thead");
            const classHeader = document.createElement("tr");

            ["Class", "Precision", "Recall", "F1-score", "Support"].forEach(name => {
                const th = document.createElement("th");
                th.textContent = name;
                classHeader.appendChild(th);
            });

            classHead.appendChild(classHeader);
            classTable.appendChild(classHead);

            const classBody = document.createElement("tbody");

            result.class_metrics.forEach(item => {
                const tr = document.createElement("tr");

                [
                    item.class_label,
                    item.precision,
                    item.recall,
                    item.f1,
                    item.support
                ].forEach(value => {
                    const td = document.createElement("td");
                    td.textContent = String(value ?? "—");
                    tr.appendChild(td);
                });

                classBody.appendChild(tr);
            });

            classTable.appendChild(classBody);
            classWrapper.appendChild(classTable);
            mlDetails.appendChild(classWrapper);
        }
    }

    // Show any baseline-model warning.
    if (result.warning) {
        mlDetails.appendChild(
            createElement("p", "message", `Warning: ${result.warning}`)
        );
    }

    mlResults.hidden = false;
}


if (mlExperimentForm) {
    mlExperimentForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!currentFile) {
            mlStatus.textContent = "Please upload and analyze a CSV first.";
            return;
        }

        const validationError = validateFile(currentFile);

        if (validationError) {
            mlStatus.textContent = validationError;
            return;
        }

        const target = mlTargetColumn.value;

        if (!target) {
            mlStatus.textContent = "Please select a target column.";
            return;
        }

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("target_column", target);
        formData.append("task", mlTask.value);
        formData.append("test_size", mlTestSize.value);
        formData.append("random_state", "42");

        runMLButton.disabled = true;
        runMLButton.textContent = "Training model...";
        mlStatus.textContent = "Running the baseline ML experiment...";
        mlResults.hidden = true;

        try {
            const response = await fetch(`${API_BASE_URL}/train`, {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(
                    typeof result.detail === "string"
                        ? result.detail
                        : "The ML experiment failed."
                );
            }

            renderMLResults(result);
            mlStatus.textContent = "Experiment completed successfully.";
        } catch (error) {
            console.error("ML experiment failed:", error);
            mlStatus.textContent = error.message || "The ML experiment failed.";
        } finally {
            runMLButton.disabled = !currentFile;
            runMLButton.textContent = "Run ML Experiment";
        }
    });
}

