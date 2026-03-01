"""
Heart Disease Explorer - Streamlit App

This is the main Streamlit application for exploring the UCI Heart Disease dataset.
It provides:
- Interactive visualizations (correlation heatmap, distributions, risk factors)
- What-If analysis tool (adjust patient parameters to see risk predictions)
- Data summary and statistics

The app loads cleaned data from a CSV file (if available) or processes raw data files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import functions from our data cleaning module
# These handle loading, cleaning, and getting feature ranges from the dataset
from data_cleaning import load_raw_data, clean_data, get_feature_ranges

# Import visualizations
from visualizations import plot_correlation_heatmap, plot_feature_distributions, plot_target_breakdown, plot_risk_factors

# Configure the Streamlit page settings
# This must be called before any other Streamlit commands
st.set_page_config(
    page_title="Heart Disease Explorer",      # Title shown in browser tab
    page_icon="❤️",                           # Heart emoji as favicon
    layout="wide",                            # Use wide layout (more horizontal space)
    initial_sidebar_state="expanded"          # Sidebar starts expanded (not collapsed)
)

# Custom styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #e74c3c; margin-bottom: 0.5rem; }
    .sub-header { color: #7f8c8d; margin-bottom: 2rem; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 1rem; border-radius: 10px; 
        color: white; margin: 0.5rem 0;
    }
    /* Slider - teal accent for readability on dark background */
    .stSlider [data-baseweb="thumb"] { background-color: #1abc9c !important; }
    .stSlider div[data-baseweb="slider"] > div > div { background-color: #1abc9c !important; }
    
    /* Slider range labels - always visible, black text */
    .slider-range-label { color: #ffffff !important; font-size: 0.85rem; opacity: 0.9; }
    
    /* Slider hover tooltip - simple, matches dark theme (tooltips render in portal) */
    [data-baseweb="popover"],
    [role="tooltip"] {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: none !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        padding: 4px 8px !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_clean_data():
    """
    Load and clean heart disease data (cached for performance).
    
    This function is decorated with @st.cache_data, which means Streamlit will:
    - Cache the result after the first call
    - Reuse the cached result on subsequent calls (faster loading)
    - Only re-run if the function code or inputs change
    
    The function tries multiple strategies to find data:
    1. First, look for pre-cleaned CSV (fastest option)
    2. If not found, look for raw data files and clean them
    3. If neither found, create sample data for demonstration
    
    Returns:
        Tuple of (DataFrame, data_dir):
        - DataFrame: The cleaned dataset
        - data_dir: Path to data directory (None if using CSV or sample data)
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    
    # Strategy 1: Prefer cleaned CSV if available (fastest, used for deployment)
    # This avoids re-running the cleaning process every time the app loads
    if (data_dir / 'heart_disease_cleaned.csv').exists():
        return pd.read_csv(data_dir / 'heart_disease_cleaned.csv'), None
    
    # Strategy 2: Load from raw files if CSV doesn't exist
    # Check if raw data files exist in the project's data directory
    if not (data_dir / 'processed.cleveland.data').exists():
        # If not in project directory, try Downloads folder
        data_dir = Path.home() / 'Downloads' / 'heart+disease'
    
    # Strategy 3: If still not found, use sample data
    if not (data_dir / 'processed.cleveland.data').exists():
        st.warning("Data not found. Using sample data. Copy heart+disease folder to app/data/ for full dataset.")
        return _create_sample_data(), None
    
    # Load and clean raw data files
    df_raw = load_raw_data(data_dir)
    df_clean = clean_data(df_raw)
    return df_clean, data_dir


def _create_sample_data():
    """
    Create minimal sample data for demo when real data is unavailable.
    
    This function generates synthetic data with realistic ranges based on the
    UCI Heart Disease dataset characteristics. It's used when the actual dataset
    files cannot be found, allowing users to still explore the app's functionality.
    
    Returns:
        DataFrame with synthetic heart disease data matching the expected schema
    """
    # Set random seed for reproducibility (same data generated each time)
    np.random.seed(42)
    n = 200  # Number of sample records to generate
    
    # Generate synthetic data with realistic ranges for each feature
    return pd.DataFrame({
        'age': np.random.randint(29, 78, n),           # Age: 29-77 years
        'sex': np.random.randint(0, 2, n),             # Sex: 0 (female) or 1 (male)
        'cp': np.random.randint(1, 5, n),              # Chest pain: 1-4
        'trestbps': np.random.randint(94, 200, n),     # Resting BP: 94-199 mm Hg
        'chol': np.random.randint(126, 564, n),        # Cholesterol: 126-563 mg/dl
        'fbs': np.random.randint(0, 2, n),             # Fasting blood sugar: 0 or 1
        'restecg': np.random.randint(0, 3, n),         # Rest ECG: 0, 1, or 2
        'thalach': np.random.randint(71, 202, n),      # Max heart rate: 71-201
        'exang': np.random.randint(0, 2, n),           # Exercise angina: 0 or 1
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),  # ST depression: 0.0-6.2 (rounded to 1 decimal)
        'slope': np.random.randint(1, 4, n),           # ST slope: 1, 2, or 3
        'ca': np.random.randint(0, 4, n),              # Major vessels: 0-3
        'thal': np.random.choice([3, 6, 7], n),        # Thalassemia: 3 (normal), 6, or 7
        'num': np.random.randint(0, 2, n)              # Target: 0 (no disease) or 1 (disease)
    })





def predict_risk_simple(features: dict, df: pd.DataFrame) -> float:
    """
    Calculate a simple risk score using centroid distance method.
    
    This is a simplified prediction algorithm that:
    1. Calculates the "centroid" (average) of all disease cases
    2. Calculates the "centroid" (average) of all no-disease cases
    3. Measures how close the input features are to each centroid
    4. Returns risk based on which centroid is closer
    
    Note: This is a simple heuristic, not a trained machine learning model.
    For production use, you'd want to train a proper classifier (e.g., logistic regression, random forest).
    
    Args:
        features: Dictionary mapping feature names to values (from user input)
        df: Training DataFrame with all features and target variable
        
    Returns:
        Estimated probability of heart disease (0.0 to 1.0)
        - 0.0 = very low risk (very close to no-disease centroid)
        - 1.0 = very high risk (very close to disease centroid)
    """
    # Separate data into disease and no-disease groups
    disease = df[df['num'] == 1].drop('num', axis=1)      # All patients with disease
    no_disease = df[df['num'] == 0].drop('num', axis=1)    # All patients without disease
    
    # Calculate centroids (mean of all features for each group)
    # This gives us the "average" patient profile for each group
    centroid_disease = disease.mean()                      # Average feature values for disease group
    centroid_no_disease = no_disease.mean()                # Average feature values for no-disease group
    
    # Build feature vector from user input
    # For each feature in the centroid, get the value from user input
    # If a feature is missing, use the disease centroid value as default
    x = np.array([features.get(col, centroid_disease[col]) for col in centroid_disease.index])
    
    # Calculate Euclidean distance from input to each centroid
    # Euclidean distance = sqrt(sum of squared differences)
    # Smaller distance = more similar to that group
    dist_disease = np.linalg.norm(x - centroid_disease.values)        # Distance to disease centroid
    dist_no_disease = np.linalg.norm(x - centroid_no_disease.values)  # Distance to no-disease centroid
    
    # Calculate risk using inverse distance weighting
    # If closer to no-disease centroid → lower risk
    # If closer to disease centroid → higher risk
    total = dist_disease + dist_no_disease
    
    # Handle edge case: if both distances are 0 (shouldn't happen, but safety check)
    if total == 0:
        return 0.5  # Neutral risk if exactly at both centroids
    
    # Risk is proportional to distance from no-disease centroid
    # If dist_no_disease is large (far from no-disease) → high risk
    # If dist_no_disease is small (close to no-disease) → low risk
    risk = dist_no_disease / total
    
    # Ensure risk is between 0 and 1
    return float(np.clip(risk, 0, 1))


def main():
    """
    Main function that runs the Streamlit app.
    
    This function:
    1. Sets up the page header
    2. Loads the data
    3. Creates the navigation sidebar
    4. Displays the appropriate page based on user selection
    """
    # Display main header using custom CSS styling
    st.markdown('<p class="main-header">❤️ Heart Disease Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the UCI Heart Disease dataset • What-If Analysis • Risk Insights</p>', unsafe_allow_html=True)
    
    # Load and clean the data (cached, so only runs once)
    df, data_dir = load_and_clean_data()
    
    # Sidebar - Navigation menu
    st.sidebar.title("Navigation")
    # Radio buttons for page selection (user can choose which page to view)
    page = st.sidebar.radio(
        "Go to",
        ["📊 Overview & Visualizations", "🎯 What-If Analysis", "📁 Data Summary"]
    )
    
    # Key metrics displayed in sidebar (always visible)
    st.sidebar.markdown("---")  # Horizontal divider
    st.sidebar.markdown("### Dataset Stats")
    
    # Display total number of records in the dataset
    st.sidebar.metric("Total Records", len(df))
    
    # Calculate and display disease prevalence
    # Mean of 'num' (0 or 1) gives the percentage of cases with disease
    disease_pct = df['num'].mean() * 100
    st.sidebar.metric("Disease Prevalence", f"{disease_pct:.1f}%")
    
    # Page 1: Overview & Visualizations
    if page == "📊 Overview & Visualizations":
        st.header("High-Impact Visualizations")
        
        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)
        
        # Left column: Correlation heatmap and target breakdown
        with col1:
            st.subheader("1. Correlation Heatmap")
            st.markdown(
                "This chart shows how pairs of measurements move together. "
                "Dark red squares mean two measurements tend to be high or low at the same time "
                "(they increase together), blue squares mean when one is high the other tends to be low."
            )
            # Create and display correlation heatmap
            fig1 = plot_correlation_heatmap(df)
            st.pyplot(fig1)  # Display the matplotlib figure in Streamlit
            plt.close()  # Close figure to free memory
            
            st.subheader("3. Target & Age Breakdown")
            st.markdown(
                "The left pie chart shows what fraction of people in the dataset have signs of heart disease. "
                "The right bar chart shows how common heart disease is in each age group, "
                "so you can see how risk changes as people get older."
            )
            # Create and display target breakdown (pie chart + age groups)
            fig3 = plot_target_breakdown(df)
            st.pyplot(fig3)
            plt.close()
        
        # Right column: Feature distributions and risk factors
        with col2:
            st.subheader("2. Feature Distributions by Disease Status")
            st.markdown(
                "Each small chart compares people **with** and **without** heart disease for one measurement. "
                "The two colors show how common each value is in each group, "
                "so you can see, for example, whether higher cholesterol or blood pressure appears more often in the disease group."
            )
            # Create and display feature distribution histograms
            fig2 = plot_feature_distributions(df)
            st.pyplot(fig2)
            plt.close()
            
            st.subheader("4. Risk Factors Comparison")
            st.markdown(
                "These box plots compare key risk factors between people with and without heart disease. "
                "Each box shows the typical range of values, and points outside the box are unusual values. "
                "Taller or higher boxes in the disease group suggest that factor tends to be worse when disease is present."
            )
            # Create and display risk factor box plots
            fig4 = plot_risk_factors(df)
            st.pyplot(fig4)
            plt.close()
    
    # Page 2: What-If Analysis
    elif page == "🎯 What-If Analysis":
        st.header("What-If Analysis")
        st.markdown("Adjust the patient parameters below to see how they affect the estimated heart disease risk.")
        
        # Get min/max/median ranges for each feature (used to set slider bounds)
        ranges = get_feature_ranges(df)
        
        # Create a two-column layout: Inputs on the left, Risk Score on the right
        input_col, result_col = st.columns([2, 1])
        
        with input_col:
            st.subheader("Patient Parameters")
            
            # Organize inputs into logical medical categories using tabs
            tab_demo, tab_symp, tab_lab, tab_ecg, tab_img = st.tabs([
                "👤 Demographics & Vitals", 
                "🩺 Symptoms", 
                "🩸 Lab Results", 
                "🏃‍♂️ Stress Test & ECG",
                "🩻 Advanced Imaging"
            ])
            
            with tab_demo:
                age = st.slider(
                    "Age (years)", 
                    int(ranges['age']['min']), int(ranges['age']['max']), int(ranges['age']['median']), 1,
                    help="Patient age in years. In this dataset most people are between about 30 and 80 years old."
                )
                st.markdown('<p class="slider-range-label">Range: %d – %d</p>' % (int(ranges['age']['min']), int(ranges['age']['max'])), unsafe_allow_html=True)
                
                sex = st.selectbox(
                    "Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                    help="Biological sex of the patient (0 = female, 1 = male)."
                )
                
                trestbps = st.slider(
                    "Resting Blood Pressure (mm Hg)", 
                    int(ranges['trestbps']['min']), int(ranges['trestbps']['max']), int(ranges['trestbps']['median']), 1,
                    help="Blood pressure measured at rest before exercise."
                )
                st.markdown('<p class="slider-range-label">Range: %d – %d mm Hg</p>' % (int(ranges['trestbps']['min']), int(ranges['trestbps']['max'])), unsafe_allow_html=True)
                
            with tab_symp:
                cp = st.selectbox(
                    "Chest Pain Type", [1, 2, 3, 4], 
                    format_func=lambda x: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal", 4: "Asymptomatic"}[x],
                    help="Type of chest pain reported."
                )
                
                exang = st.selectbox(
                    "Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                    help="Chest pain brought on by exercise (1 = yes, 0 = no)."
                )
                
            with tab_lab:
                chol = st.slider(
                    "Cholesterol (mg/dl)", 
                    int(ranges['chol']['min']), int(ranges['chol']['max']), int(ranges['chol']['median']), 1,
                    help="Total cholesterol level in the blood."
                )
                st.markdown('<p class="slider-range-label">Range: %d – %d mg/dl</p>' % (int(ranges['chol']['min']), int(ranges['chol']['max'])), unsafe_allow_html=True)
                
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                    help="Whether fasting blood sugar is greater than 120 mg/dl."
                )
                
            with tab_ecg:
                thalach = st.slider(
                    "Max Heart Rate Achieved", 
                    int(ranges['thalach']['min']), int(ranges['thalach']['max']), int(ranges['thalach']['median']), 1,
                    help="Highest heart rate reached during an exercise test."
                )
                st.markdown('<p class="slider-range-label">Range: %d – %d</p>' % (int(ranges['thalach']['min']), int(ranges['thalach']['max'])), unsafe_allow_html=True)
                
                restecg = st.selectbox(
                    "Rest ECG", [0, 1, 2], 
                    format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[x],
                    help="Result of a resting electrocardiogram (ECG)."
                )
                
                oldpeak = st.slider(
                    "ST Depression (oldpeak)", 
                    float(ranges['oldpeak']['min']), float(ranges['oldpeak']['max']), float(ranges['oldpeak']['median']), 0.1,
                    help="Change in a specific part of the ECG (ST segment) during exercise."
                )
                st.markdown('<p class="slider-range-label">Range: %.1f – %.1f</p>' % (float(ranges['oldpeak']['min']), float(ranges['oldpeak']['max'])), unsafe_allow_html=True)
                
                slope = st.selectbox(
                    "ST Slope", [1, 2, 3], 
                    format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
                    help="Shape (slope) of the ST segment on the ECG during peak exercise."
                )
                
            with tab_img:
                ca = st.slider(
                    "Number of Major Vessels (0-4)", 0, 4, 0, 1,
                    help="Number of major blood vessels seen as open on a special heart imaging test (0–4)."
                )
                st.markdown('<p class="slider-range-label">Range: 0 – 4</p>', unsafe_allow_html=True)
                
                thal = st.selectbox(
                    "Thalassemia", [3, 6, 7], 
                    format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x],
                    help="Result of a thallium heart scan."
                )
                
        # Collect all user inputs into a dictionary
        features = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Calculate risk score based on user inputs
        risk = predict_risk_simple(features, df)
        
        # Display results on the right column
        with result_col:
            st.subheader("Risk Assessment")
            
            # Convert risk (0-1) to percentage (0-100%)
            risk_pct = risk * 100
            
            # Show warning or success message based on risk level
            if risk_pct > 50:
                st.error("⚠️ **High Risk Profile**\n\nConsider consulting a healthcare provider or doctor.")
            else:
                st.success("✅ **Lower Risk Profile**\n\nLooks good based on these parameters.")
                
            st.markdown("---")
            
            # Display risk as a metric with help text
            st.metric(
                "Estimated Heart Disease Risk", 
                f"{risk_pct:.1f}%", 
                help="Based on centroid distance from disease vs no-disease patient profiles."
            )
            
            # Display progress bar (visual indicator of risk level)
            st.progress(risk)
    
    # Page 3: Data Summary
    else:
        st.header("Data Summary")
        
        # Display first 100 rows of the dataset in an interactive table
        # use_container_width=True makes the table use the full width of the page
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Statistics")
        # Display descriptive statistics (count, mean, std, min, max, quartiles)
        # This gives users a quick overview of the data distribution
        st.dataframe(df.describe(), use_container_width=True)


if __name__ == '__main__':
    main()