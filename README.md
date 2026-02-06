# Heart Disease Explorer

A Streamlit app for exploring the UCI Heart Disease dataset with visualizations and What-If analysis.

## Features

- **Data Cleaning**: Handles nulls (`?`, `-9`), invalid values, and encodes categorical variables
- **4 Visualizations**: Correlation heatmap, feature distributions, target breakdown, risk factor box plots
- **What-If Sliders**: Adjust patient parameters to see estimated heart disease risk

## Quick Start

```bash
# 1. Copy data (if you have it in Downloads)
python3 setup_data.py

# 2. Clean data
python3 data_cleaning.py

# 3. Run the app
streamlit run app.py
```

Or with a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 setup_data.py
python3 data_cleaning.py
streamlit run app.py
```

## Project Structure

```
heart-disease-app/
├── app.py              # Streamlit app
├── data_cleaning.py    # Data cleaning script
├── setup_data.py       # Copies data from Downloads
├── requirements.txt
├── data/
│   ├── processed.*.data    # Raw data files
│   └── heart_disease_cleaned.csv  # Cleaned output
├── DEPLOYMENT.md       # Streamlit Cloud deployment steps
└── README.md
```

## Deploy to Streamlit Community Cloud

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions.

## Data Source

UCI Machine Learning Repository - Heart Disease Dataset  
[https://archive.ics.uci.edu/ml/datasets/heart+disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
