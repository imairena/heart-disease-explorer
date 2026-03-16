# Heart Disease Explorer

A Streamlit app for exploring the UCI Heart Disease dataset with visualizations, data analysis, and predictive What-If scenarios.

## Features

- **Data Cleaning**: Automatically handles nulls (`?`, `-9`), invalid values, and encodes categorical variables.
- **4 Visualizations**: Correlation heatmap, feature distributions by target, overall target breakdown, and risk factor box plots.
- **What-If Sliders**: Adjust patient parameters dynamically across demographics, symptoms, lab results, and ECG to determine risk.
- **Machine Learning Predictions**: Toggle between a simple distance-based heuristic (centroid calculation) and a trained Linear Regression model to predict patient risk outcomes.
- **Automated CI/CD**: Uses GitHub Actions for continuous integration, validating code quality with `pytest` and `flake8` on push/PR.

## Quick Start

```bash
# 1. Setup a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup and Clean data
python3 setup_data.py
python3 data_cleaning.py

# 4. (Optional) Run tests and linters
flake8 . --exclude=venv --max-line-length=120 --extend-ignore=E1,E2,E3,W,E501
pytest

# 5. Run the app
streamlit run app.py
```

## Project Structure

```text
heart-disease-app/
├── .github/workflows/  # CI pipeline config (linting & testing)
├── .streamlit/         # Streamlit custom styling & configuration
├── tests/              # Pytest unit and integration tests
├── app.py              # Main Streamlit application entrypoint
├── data_cleaning.py    # Data standardisation and processing
├── setup_data.py       # Helper utility to retrieve raw data
├── visualizations.py   # Modular plotting and charting functions
├── what_if.py          # ML Risk Prediction UI and algorithm logic
├── pytest.ini          # Pytest configuration variables
├── requirements.txt    # Required python packages
├── data/
│   ├── processed.*.data           # Raw data files
│   └── heart_disease_cleaned.csv  # Final cleaned output
├── DEPLOYMENT.md       # Streamlit Cloud deployment steps
└── README.md
```

## Deploy to Streamlit Community Cloud

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed step-by-step instructions.

## Data Source

UCI Machine Learning Repository - Heart Disease Dataset  
[https://archive.ics.uci.edu/ml/datasets/heart+disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
