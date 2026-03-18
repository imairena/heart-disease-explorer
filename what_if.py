import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from data_cleaning import get_feature_ranges
import google.generativeai as genai

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = ""

@st.cache_data
def get_centroids(df: pd.DataFrame):
    """Cache the centroid calculations so they don't re-run on every slider change."""
    disease = df[df['num'] == 1].drop('num', axis=1)
    no_disease = df[df['num'] == 0].drop('num', axis=1)
    return disease.mean(), no_disease.mean()

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
    # Get cached centroids (average of all features for each group)
    # This gives us the "average" patient profile for each group
    centroid_disease, centroid_no_disease = get_centroids(df)
    
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

@st.cache_resource
def train_linear_regression(df: pd.DataFrame):
    """Train and cache a simple Linear Regression model."""
    X = df.drop('num', axis=1)
    y = df['num']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_risk_ml(features: dict, df: pd.DataFrame) -> float:
    """Calculate risk score using a trained Linear Regression model."""
    model = train_linear_regression(df)
    
    # Ensure features are in the exact order as training data
    X_train_columns = df.drop('num', axis=1).columns
    x_array = np.array([features.get(col, 0) for col in X_train_columns]).reshape(1, -1)
    
    # Predict risk and clip to [0, 1] range
    risk = model.predict(x_array)[0]
    return float(np.clip(risk, 0, 1))

def render_what_if_analysis(df: pd.DataFrame):
    """
    Render the What-If Analysis page.
    """
    st.header("What-If Analysis")
    st.markdown("Adjust the patient parameters below to see how they affect the estimated heart disease risk.")
    
    # Get min/max/median ranges for each feature (used to set slider bounds)
    ranges = get_feature_ranges(df)
    
    # Create a two-column layout: Inputs on the left, Risk Score on the right
    input_col, result_col = st.columns([2, 1])
    
    with input_col:
        st.subheader("Prediction Model")
        prediction_model = st.radio(
            "Select Risk Prediction Method:",
            ["Simple Heuristic (Centroid)", "Machine Learning (Linear Regression)"],
            help="Choose between a simple distance-based heuristic or a trained Linear Regression ML model."
        )
        st.markdown("---")
        
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
                help="Patient age in years.\n\nRisk broadly increases with age."
            )
            st.markdown('<p class="slider-range-label">Range: %d – %d</p>' % (int(ranges['age']['min']), int(ranges['age']['max'])), unsafe_allow_html=True)
            
            sex = st.selectbox(
                "Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                help="Biological sex of the patient (0 = female, 1 = male)."
            )
            
            trestbps = st.slider(
                "Resting Blood Pressure (mm Hg)", 
                int(ranges['trestbps']['min']), int(ranges['trestbps']['max']), int(ranges['trestbps']['median']), 1,
                help="Blood pressure measured at rest before exercise.\n\nHealthy: < 120 mm Hg\nElevated: 120-129 mm Hg\nHigh: ≥ 130 mm Hg\n\nSymptoms of high BP: often none, but can include headaches, shortness of breath, or nosebleeds."
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
                help="Total cholesterol level in the blood.\n\nHealthy: < 200 mg/dl\nBorderline high: 200-239 mg/dl\nHigh: ≥ 240 mg/dl\n\nSymptoms: High cholesterol has no symptoms but increases risk of heart disease."
            )
            st.markdown('<p class="slider-range-label">Range: %d – %d mg/dl</p>' % (int(ranges['chol']['min']), int(ranges['chol']['max'])), unsafe_allow_html=True)
            
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether fasting blood sugar is greater than 120 mg/dl.\n\nHealthy: < 100 mg/dl\nPrediabetes: 100-125 mg/dl\nDiabetes: ≥ 126 mg/dl\n\nSymptoms of high sugar: increased thirst, frequent urination, fatigue, blurred vision."
            )
            
        with tab_ecg:
            thalach = st.slider(
                "Max Heart Rate Achieved", 
                int(ranges['thalach']['min']), int(ranges['thalach']['max']), int(ranges['thalach']['median']), 1,
                help="Highest heart rate reached during an exercise test.\n\nHealthy max roughly: 220 minus your age.\nLower achieved max HR can indicate poor fitness or heart problems.\n\nSymptoms of abnormal HR: palpitations, dizziness, shortness of breath."
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
    
    # Calculate risk score based on user inputs and chosen model
    if "Linear Regression" in prediction_model:
        risk = predict_risk_ml(features, df)
        model_desc = "Based on a Linear Regression machine learning model."
    else:
        risk = predict_risk_simple(features, df)
        model_desc = "Based on centroid distance from disease vs no-disease patient profiles."
    
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
            help=model_desc
        )
        
        # Display progress bar (visual indicator of risk level)
        st.progress(risk)

    # --- AI Chatbot Section ---
    st.markdown("---")
    st.subheader("🤖 AI Medical Assistant")
    st.markdown("Ask the AI to explain your risk score or how to improve your health based on your inputs.")
    
    if not GEMINI_API_KEY:
        st.info("💡 **Tip:** To activate the AI Assistant, open `.streamlit/secrets.toml` and paste your free Google Gemini API Key.")
        
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input
    if prompt := st.chat_input("Ask about your risk score or how to lower it..."):
        if not GEMINI_API_KEY:
            st.error("API Key missing! Please add your Gemini API Key to `.streamlit/secrets.toml`.")
        else:
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                # Configure API
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Build context
                context = f"""
You are a helpful medical AI assistant integrated into a Heart Disease Explorer app.
Current patient parameters:
- Age: {age} | Sex: {"Male" if sex == 1 else "Female"}
- Blood Pressure: {trestbps} mm Hg | Cholesterol: {chol} mg/dl
- Fasting Blood Sugar > 120: {"Yes" if fbs == 1 else "No"}
- Max Heart Rate: {thalach} | Chest Pain Type: {cp}
Model predicted Heart Disease Risk: {risk_pct:.1f}%

User question: {prompt}

Answer briefly, empathetically, and accurately based on their parameters.
Always include a short disclaimer that you are an AI, not a doctor.
"""
                with st.spinner("Analyzing..."):
                    response = model.generate_content(context)
                
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
