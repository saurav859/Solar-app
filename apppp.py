# 📦 Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration ---
# 📍 Model save path
MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')

# Define the input features, using the EXACT feature names (keys) the model expects.
# I've added a 'display' key to show user-friendly text in the sidebar.
INPUT_FEATURES = {
    # Key (Model Name) : {'default': X, 'step': Y, 'display': 'User-Friendly Label'}
    'distance-to-solar-noon': {'default': 0.50, 'step': 0.01, 'display': 'Distance to Solar Noon [0-1]'},
    'temperature': {'default': 70.0, 'step': 1.0, 'display': 'Temperature (°F)'},
    'wind-direction': {'default': 90.0, 'step': 1.0, 'display': 'Wind Direction (deg)'},
    'wind-speed': {'default': 5.00, 'step': 0.1, 'display': 'Wind Speed (mph)'},
    'sky-cover': {'default': 2.0, 'step': 1.0, 'display': 'Sky Cover [0-10]'},
    'visibility': {'default': 10.00, 'step': 0.01, 'display': 'Visibility (miles)'},
    'humidity': {'default': 50.0, 'step': 1.0, 'display': 'Humidity (%)'},
    
    # 🚨 CRITICAL FIX: These keys now match the 'Feature names seen at fit time'
    'average-wind-speed-(period)': {'default': 5.00, 'step': 0.01, 'display': 'Average Wind Speed (period)'},
    'average-pressure-(period)': {'default': 29.80, 'step': 0.01, 'display': 'Average Pressure (period)'},
}

# The list of feature keys, in the exact order the model expects.
FEATURE_KEYS_ORDERED = list(INPUT_FEATURES.keys())
# --- End Configuration ---

# 📌 Streamlit UI - Main App
st.title("☀️ Solar Power Generation Predictor")
st.write("Adjust the weather parameters in the sidebar to get a prediction.")

# --- Model Loading (Load once) ---
try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        st.info("Please ensure you have run your training script to generate a valid `model.pkl`.")
        st.stop()
        
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    scaler = saved['scaler']
    st.sidebar.success("✅ Model and Scaler Loaded!")
    
except Exception as e:
    st.error(f"❌ Error loading model components: {e}")
    st.info("The `model.pkl` file might be corrupted or in an incorrect format.")
    st.stop()

# 1. Sidebar/Input Parameters Section
user_inputs = {}
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    # Collect user inputs using st.number_input
    for key_name, params in INPUT_FEATURES.items():
        # Use the 'display' name for the label, but the 'key_name' (model feature) for the dict/key
        user_inputs[key_name] = st.number_input(
            label=params['display'], 
            value=params['default'], 
            step=params['step'],
            key=key_name # Unique key for the widget
        )

# Convert inputs to a DataFrame for prediction
input_df = pd.DataFrame([user_inputs])
# Ensure the columns are in the exact order the model expects
X_input_ordered = input_df[FEATURE_KEYS_ORDERED]


# 2. Prediction Logic and Output Display
if st.button("🔮 Predict Solar Power Generated", type="primary"):
    
    with st.spinner('Calculating prediction...'):
        try:
            # Prepare data: Scale the user input
            X_scaled = scaler.transform(X_input_ordered)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            st.subheader("✅ Prediction Result")
            st.metric(
                label="Predicted Solar Power Generated (kW)", 
                value=f"{prediction:,.2f}"
            )

            # Display the input as a table
            st.subheader("Input Parameters Used")
            
            # Create a display-friendly DataFrame: transpose and rename index
            display_data = X_input_ordered.T.rename(
                columns={0: 'Value'}
            )
            # Use the full, descriptive names for the index/row headers
            display_data.index = [d['display'] for d in INPUT_FEATURES.values()]
            
            st.dataframe(display_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please verify the column names used in your model training script.")

# Add a reset button at the bottom of the sidebar
with st.sidebar:
    if st.button("🔄 Reset to Defaults"):
         st.experimental_rerun()