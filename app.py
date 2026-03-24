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

from pathlib import Path

# Import functions from our data cleaning module
# These handle loading, cleaning, and getting feature ranges from the dataset
from data_cleaning import load_raw_data, clean_data

# Import visualizations
from visualizations import plot_correlation_heatmap, plot_feature_distributions, plot_target_breakdown, plot_risk_factors

# Import what-if analysis
from what_if import render_what_if_analysis
from copilot import render_global_copilot

# Configure the Streamlit page settings
# This must be called before any other Streamlit commands
st.set_page_config(
    page_title="Heart Disease Explorer",      # Title shown in browser tab
    page_icon="❤️",                           # Heart emoji as favicon
    layout="wide",                            # Use wide layout (more horizontal space)
    initial_sidebar_state="collapsed"         # Sidebar starts collapsed
)

# Custom styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #E06D53; margin-bottom: 0.5rem; transition: color 0.3s ease; }
    .main-header:hover { color: #E6F1FF; }
    .sub-header { color: #8892b0; margin-bottom: 2rem; }
    .metric-card { 
        background: #112240; 
        padding: 1rem; border-radius: 10px; 
        color: #E6F1FF; margin: 0.5rem 0;
        border: 1px solid #233554;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        border-color: #E06D53;
    }
    
    /* Button hover effects */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(224, 109, 83, 0.2);
    }
    
    /* Tab hover effects */
    .stTabs [data-baseweb="tab-list"] button {
        transition: color 0.3s ease, background-color 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #E06D53 !important;
        background-color: rgba(224, 109, 83, 0.1);
        border-radius: 4px;
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
    
    # Create horizontal tabs instead of sidebar navigation
    tab1, tab2, tab3 = st.tabs(["📊 Overview & Visualizations", "🎯 What-If Analysis", "📁 Data Summary"])
    
    # Page 1: Overview & Visualizations
    with tab1:
        st.header("High-Impact Visualizations")
        
        # Dynamic Filtering in an Expander
        with st.expander("🔍︎ Filter Visualizations", expanded=True):
            st.markdown("Adjust these filters to focus the charts on specific patient groups.")
            f_col1, f_col2, f_col3 = st.columns(3)
            
            with f_col1:
                age_min, age_max = int(df['age'].min()), int(df['age'].max())
                selected_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))
                
            with f_col2:
                sex_filter = st.multiselect("Sex", ["Female (0)", "Male (1)"], ["Female (0)", "Male (1)"])
                sex_map = {"Female (0)": 0, "Male (1)": 1}
                selected_sex = [sex_map[s] for s in sex_filter] if sex_filter else [0, 1]
                
            with f_col3:
                chest_pain = st.multiselect(
                    "Chest Pain Type", 
                    ["Typical angina (1)", "Atypical angina (2)", "Non-anginal (3)", "Asymptomatic (4)"],
                    ["Typical angina (1)", "Atypical angina (2)", "Non-anginal (3)", "Asymptomatic (4)"]
                )
                cp_map = {"Typical angina (1)": 1, "Atypical angina (2)": 2, "Non-anginal (3)": 3, "Asymptomatic (4)": 4}
                selected_cp = [cp_map[c] for c in chest_pain] if chest_pain else [1, 2, 3, 4]

        filtered_df = df[
            (df['age'] >= selected_age[0]) & 
            (df['age'] <= selected_age[1]) & 
            (df['sex'].isin(selected_sex)) &
            (df['cp'].isin(selected_cp))
        ]
        
        if filtered_df.empty:
            st.warning("No records match these filters. Showing all data instead.")
            filtered_df = df
            
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
            fig1 = plot_correlation_heatmap(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("3. Target & Age Breakdown")
            st.markdown(
                "The left pie chart shows what fraction of people in the dataset have signs of heart disease. "
                "The right bar chart shows how common heart disease is in each age group, "
                "so you can see how risk changes as people get older."
            )
            # Create and display target breakdown (pie chart + age groups)
            fig3 = plot_target_breakdown(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Right column: Feature distributions and risk factors
        with col2:
            st.subheader("2. Feature Distributions by Disease Status")
            st.markdown(
                "Each small chart compares people **with** and **without** heart disease for one measurement. "
                "The two colors show how common each value is in each group, "
                "so you can see, for example, whether higher cholesterol or blood pressure appears more often in the disease group."
            )
            # Create and display feature distribution histograms
            fig2 = plot_feature_distributions(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("4. Risk Factors Comparison")
            st.markdown(
                "These box plots compare key risk factors between people with and without heart disease. "
                "Each box shows the typical range of values, and points outside the box are unusual values. "
                "Taller or higher boxes in the disease group suggest that factor tends to be worse when disease is present."
            )
            # Create and display risk factor box plots
            fig4 = plot_risk_factors(filtered_df)
            st.plotly_chart(fig4, use_container_width=True)
    
    # Page 2: What-If Analysis
    with tab2:
        render_what_if_analysis(df)
    
    # Page 3: Data Summary
    with tab3:
        st.header("Data Summary")
        
        # Key metrics displayed in Data Summary
        st.markdown("### Dataset Stats (All Records)")
        col_stats1, col_stats2 = st.columns(2)
        
        # Display total number of records in the dataset
        with col_stats1:
            st.metric("Total Records", len(df))
        
        # Calculate and display disease prevalence
        with col_stats2:
            disease_pct = df['num'].mean() * 100 if len(df) > 0 else 0
            st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
        
        st.markdown("---")
        
        # Add a download button for the dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv,
            file_name="heart_disease_data.csv",
            mime="text/csv",
        )
        
        # Display first 100 rows of the dataset in an interactive table
        # use_container_width=True makes the table use the full width of the page
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Statistics")
        # Display descriptive statistics (count, mean, std, min, max, quartiles)
        # This gives users a quick overview of the data distribution
        st.dataframe(df.describe(), use_container_width=True)

    # Global AI copilot (shared across all pages/tabs)
    render_global_copilot(df)

if __name__ == '__main__':
    main()