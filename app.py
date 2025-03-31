import streamlit as st
import joblib
import numpy as np

# Load the saved Decision Tree Regressor model
model_filename = 'decision_tree_regressor_model.pkl'
regressor = joblib.load(model_filename)

st.title("Salary Prediction App")
st.markdown("""
This app predicts the salary based on Gender, Experience (Years), and Position.
""")

# --- User Inputs ---
# Gender input (assuming "F" is encoded as 0 and "M" as 1)
gender = st.selectbox("Select Gender", ("F", "M"))
gender_encoded = 0 if gender == "F" else 1

# Experience input in years
experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=5)

# Position input
position = st.selectbox("Select Position", ("DevOps Engineer", "Systems Administrator", "Web Developer"))
# Mapping based on LabelEncoder used during training
position_mapping = {"DevOps Engineer": 0, "Systems Administrator": 1, "Web Developer": 2}
position_encoded = position_mapping[position]

# Prepare input features (2D array for prediction)
input_features = np.array([[gender_encoded, experience, position_encoded]])

# --- Prediction ---
if st.button("Predict Salary"):
    predicted_salary = regressor.predict(input_features)
    st.success(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
