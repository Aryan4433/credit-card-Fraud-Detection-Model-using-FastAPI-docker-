import streamlit as st
import requests
from datetime import datetime, date

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("Detect suspicious transactions using Machine Learning")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Transaction Details")

category = st.sidebar.selectbox(
    "Category",
    [
        "travel",
        "shopping_pos",
        "shopping_net",
        "misc_net",
        "misc_pos",
        "entertainment",
        "food_dining",
        "personal_care",
        "kids_pets",
        "home",
        "health_fitness",
        "grocery_pos",
        "grocery_net",
        "gas_transport"
    ]
)

gender = st.sidebar.selectbox("Gender", ["male", "female"])

amt = st.sidebar.slider("Transaction Amount", 1.0, 10000.0, 500.0)

city_pop = st.sidebar.number_input("City Population", value=500000)

lat = st.sidebar.number_input("User Latitude", value=28.6139)
long = st.sidebar.number_input("User Longitude", value=77.2090)

merch_lat = st.sidebar.number_input("Merchant Latitude", value=28.7041)
merch_long = st.sidebar.number_input("Merchant Longitude", value=77.1025)

trans_date = st.sidebar.date_input("Transaction Date", date.today())
trans_time = st.sidebar.time_input("Transaction Time", datetime.now().time())

dob = st.sidebar.date_input("Date of Birth", date(1995, 1, 1))

# -----------------------------
# Button
# -----------------------------
if st.button("🔍 Check Fraud"):

    payload = {
        "category": category,
        "amt": amt,
        "gender": gender,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "trans_datetime": f"{trans_date}T{trans_time}",
        "dob": str(dob)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        st.divider()

        if "error" in result:
            st.error(result["error"])

        else:
            fraud = result["fraud"]
            prob = result["probability"]
            label = result["label"]

            # Result UI
            if fraud == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

            # Probability
            st.subheader("Fraud Probability")
            st.progress(prob)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", label)

            with col2:
                st.metric("Confidence", f"{prob:.2%}")

    except Exception as e:
        st.error(f"API Connection Error: {e}")