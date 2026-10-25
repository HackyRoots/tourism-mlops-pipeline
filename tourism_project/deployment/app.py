# app.py
# Streamlit app for the Wellness Tourism Package prediction.
# Loads the model committed to this folder by the GitHub Actions pipeline, collects the customer
# details, stores them in a dataframe and shows the predicted purchase likelihood.
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_model_v1.joblib"
THRESHOLD = 0.5   # same classification threshold that was used when the model was evaluated

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Wellness Tourism Package Prediction")
st.write("Enter the customer details to estimate whether the customer is likely to purchase the Wellness Tourism Package.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits the trained model to this folder.")
    st.stop()

model = load_model()

# ---------------- Customer details ----------------
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=0, max_value=500000, value=22000, step=500)
with col2:
    number_of_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    number_of_children = st.number_input("Number of children visiting (below 5)", min_value=0, max_value=5, value=1)
    preferred_star = st.selectbox("Preferred property star rating", [3, 4, 5])
    number_of_trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])

st.subheader("Sales interaction details")
col3, col4 = st.columns(2)
with col3:
    product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=150, value=15)
with col4:
    number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    pitch_satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)

# Save the inputs into a dataframe (column names must match the training data)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_persons,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

st.subheader("Input summary")
st.write(input_df)

if st.button("Predict"):
    probability = float(model.predict_proba(input_df)[0, 1])
    st.subheader("Prediction")
    st.write(f"Probability of purchasing the package: **{probability:.1%}**")
    if probability >= THRESHOLD:
        st.success("Likely to purchase the Wellness Tourism Package. Worth prioritising for the campaign.")
    else:
        st.warning("Unlikely to purchase the Wellness Tourism Package.")
