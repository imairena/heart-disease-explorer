"""
Heart Disease Data Cleaning Script

This script processes raw heart disease data from the UCI Machine Learning Repository.
It handles various data quality issues including:
- Missing values represented as '?' or -9
- Invalid zero values for certain numeric fields (biologically implausible)
- Encoding categorical variables
- Creating a binary target variable from the original multi-class target

The cleaned dataset is saved as a CSV file for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Opt-in to future pandas behavior to avoid Downcasting FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Column names for the 14 used attributes (from UCI documentation)
# These match the standard UCI Heart Disease dataset format
COLUMNS = [
    'age',        # Age in years
    'sex',        # Sex (0 = female, 1 = male)
    'cp',         # Chest pain type (1-4)
    'trestbps',   # Resting blood pressure (mm Hg)
    'chol',       # Serum cholesterol (mg/dl)
    'fbs',        # Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)
    'restecg',    # Resting electrocardiographic results (0-2)
    'thalach',    # Maximum heart rate achieved
    'exang',      # Exercise induced angina (0 = no, 1 = yes)
    'oldpeak',    # ST depression induced by exercise relative to rest
    'slope',      # Slope of the peak exercise ST segment (1-3)
    'ca',         # Number of major vessels colored by flourosopy (0-4)
    'thal',       # Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    'num'         # Target: diagnosis of heart disease (0 = no, 1-4 = yes, severity)
]

# Numeric columns that can have missing values and need imputation
# These are continuous variables that can be filled with median values
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Categorical columns that need special handling
# These are discrete variables that should be filled with mode (most common value)
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Target column name - this is what we're trying to predict
TARGET_COL = 'num'


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine all processed heart disease data files.
    
    The UCI Heart Disease dataset comes from 4 different locations:
    - Cleveland (USA)
    - Hungary
    - Switzerland
    - VA Long Beach (USA)
    
    This function reads all available files and combines them into a single DataFrame.
    
    Args:
        data_dir: Path to the directory containing the data files
        
    Returns:
        A pandas DataFrame containing all combined data with column names assigned
    """
    # List of all data files we expect to find
    files = [
        data_dir / 'processed.cleveland.data',
        data_dir / 'processed.hungarian.data',
        data_dir / 'processed.switzerland.data',
        data_dir / 'processed.va.data',
    ]
    
    # Collect DataFrames from each file that exists
    dfs = []
    for f in files:
        if f.exists():
            # Read CSV without header row (raw data has no column names)
            # Assign column names from our COLUMNS constant
            df = pd.read_csv(f, header=None, names=COLUMNS)
            dfs.append(df)
    
    # Combine all DataFrames into one, resetting the index
    # ignore_index=True ensures we get a continuous index (0, 1, 2, ...)
    return pd.concat(dfs, ignore_index=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the heart disease dataset through multiple steps.
    
    This function performs comprehensive data cleaning:
    1. Replace missing value indicators (? and -9) with NaN
    2. Handle invalid zeros (e.g., chol=0, trestbps=0) that are biologically implausible
    3. Drop rows with critical missing values (target variable)
    4. Convert multi-class target to binary (0 = no disease, 1 = disease)
    5. Impute remaining missing values using median (numeric) or mode (categorical)
    6. Ensure proper data types and valid value ranges
    
    Args:
        df: Raw DataFrame with uncleaned data
        
    Returns:
        Cleaned DataFrame ready for analysis
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Step 1: Replace missing value indicators with NaN
    # The UCI dataset uses '?' and -9 to indicate missing values
    df = df.replace('?', np.nan)
    df = df.replace(-9, np.nan)      # Integer -9
    df = df.replace(-9.0, np.nan)    # Float -9.0 (in case it was converted)
    
    # Step 2: Convert all columns to numeric where possible
    # errors='coerce' converts non-numeric values to NaN instead of raising an error
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 3: Handle invalid zeros - these are biologically implausible
    # Cholesterol and resting blood pressure cannot be 0 in real patients
    # These are likely data entry errors or missing value codes
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    
    # Step 4: Drop rows where target variable is missing
    # We can't use rows where we don't know the outcome (heart disease status)
    df = df.dropna(subset=[TARGET_COL])
    
    # Step 5: Convert multi-class target to binary classification
    # Original: 0 = no disease, 1-4 = different severity levels of disease
    # Binary: 0 = no disease, 1 = disease (any severity)
    # This simplifies the problem to "disease present" vs "disease absent"
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    
    # Step 6: Impute missing values in numeric columns using median
    # Median is robust to outliers and works well for continuous variables
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        if col in df.columns and df[col].isna().any():
            # Fill missing values with the median of that column
            df[col] = df[col].fillna(df[col].median())
    
    # Step 7: Impute missing values in categorical columns using mode (most common value)
    # Mode is appropriate for discrete categorical variables
    for col in ['slope', 'ca', 'thal']:
        if col in df.columns and df[col].isna().any():
            # Get the most common value (mode)
            mode_val = df[col].mode()
            # Use the first mode value if it exists, otherwise default to 0
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 0)
    
    # Step 8: Drop any remaining rows with NaN values
    # After imputation, there should be very few (if any) remaining NaNs
    # But we drop them to ensure a completely clean dataset
    df = df.dropna()
    
    # Step 9: Ensure categorical columns are integers (not floats)
    # This is important for consistency and to avoid issues with comparisons
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Step 10: Clip values to valid ranges based on UCI dataset documentation
    # 'ca' (number of major vessels) should be 0-4
    if 'ca' in df.columns:
        df['ca'] = df['ca'].clip(0, 4)
    
    # 'thal' (thalassemia) should be 3, 6, or 7
    # Sometimes 0 is used incorrectly to mean "normal" (which is 3)
    if 'thal' in df.columns:
        df['thal'] = df['thal'].replace(0, 3)  # Replace 0 with 3 (normal)
    
    return df


def get_feature_ranges(df: pd.DataFrame) -> dict:
    """
    Get min/max/median ranges for each feature.
    
    This function extracts the statistical ranges of each feature in the dataset.
    These ranges are used in the Streamlit app to set up sliders for the What-If analysis.
    Users can adjust feature values within these ranges to see how they affect risk predictions.
    
    Args:
        df: Cleaned DataFrame with all features
        
    Returns:
        Dictionary mapping feature names to their min, max, and median values
        Example: {'age': {'min': 29.0, 'max': 77.0, 'median': 56.0}, ...}
    """
    ranges = {}
    
    # Iterate through all columns except the target variable
    for col in df.columns:
        if col != TARGET_COL:
            # Only process numeric columns (int64 or float64)
            # Categorical columns will be handled separately in the UI
            if df[col].dtype in ['int64', 'float64']:
                ranges[col] = {
                    'min': float(df[col].min()),      # Minimum value in dataset
                    'max': float(df[col].max()),      # Maximum value in dataset
                    'median': float(df[col].median()), # Median value (good default for sliders)
                }
    
    return ranges


def main():
    """
    Main function to run the data cleaning pipeline.
    
    This function:
    1. Locates the raw data files (checks multiple possible locations)
    2. Loads all raw data files
    3. Cleans the data using the clean_data() function
    4. Saves the cleaned data to a CSV file
    5. Prints summary statistics
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # First, try to find data in the 'data' subdirectory of the project
    data_dir = script_dir / 'data'
    
    # If data folder doesn't exist in project, try the Downloads folder
    # This is a common location where users might have downloaded the dataset
    if not (data_dir / 'processed.cleveland.data').exists():
        data_dir = Path.home() / 'Downloads' / 'heart+disease'
    
    # If still not found, raise an error with helpful instructions
    if not (data_dir / 'processed.cleveland.data').exists():
        raise FileNotFoundError(
            f"Data not found. Please copy the heart+disease folder to {script_dir / 'data'} "
            f"or ensure it exists at {data_dir}"
        )
    
    # Step 1: Load raw data from all available files
    print("Loading raw data...")
    df_raw = load_raw_data(data_dir)
    print(f"Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # Step 2: Clean the data
    print("\nCleaning data...")
    df_clean = clean_data(df_raw)
    print(f"Cleaned data: {len(df_clean)} rows")
    # Note: The number of rows may decrease if rows with missing target values were dropped
    
    # Step 3: Save cleaned data to CSV file
    # This CSV will be used by the Streamlit app for faster loading
    output_path = script_dir / 'data' / 'heart_disease_cleaned.csv'
    output_path.parent.mkdir(exist_ok=True)  # Create 'data' directory if it doesn't exist
    df_clean.to_csv(output_path, index=False)  # index=False prevents saving row numbers
    print(f"\nSaved cleaned data to {output_path}")
    
    # Step 4: Print summary statistics for verification
    print("\n--- Data Summary ---")
    print(df_clean.describe())  # Shows count, mean, std, min, 25%, 50%, 75%, max for each column
    
    print("\n--- Target Distribution ---")
    print(df_clean[TARGET_COL].value_counts())  # Shows how many cases have disease vs no disease
    
    return df_clean


if __name__ == '__main__':
    main()