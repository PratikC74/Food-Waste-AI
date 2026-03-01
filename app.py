import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split



# 1️⃣ Page Configuration (FIRST)
st.set_page_config(
    page_title="Food Waste AI",
    page_icon="🍱",
    layout="wide"
)

# 2️⃣ 👉 ADD UI STYLING HERE 👇
st.markdown("""
<style>
/* Background */
.main {
    background-color: #f5f7fa;
}

/* Buttons */
.stButton>button {
    background-color: #2E8B57;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# 3️⃣ THEN start your app UI
st.title("🍱 AI-Based Food Waste Optimization System")
st.markdown("Smart Prediction & Redistribution Platform")


# -----------------------------
# NGO Static Location Data
# -----------------------------
ngo_locations = pd.DataFrame({
    "ngo_name": ["Helping Hands", "Food For All", "Care & Share"],
    "lat": [19.0760, 19.2183, 19.0330],
    "lon": [72.8777, 72.9781, 72.8650],
    "phone": ["+91 9876543210", "+91 9123456780", "+91 9988776655"]
})

# -----------------------------
# Load Datasets
# -----------------------------
food = pd.read_csv("food_data.csv")
ngo = pd.read_csv("ngo_data.csv")

# -----------------------------
# Preprocessing
# -----------------------------
food_encoded = pd.get_dummies(food, columns=["Event_Type"])

X = food_encoded.drop(["Surplus_kg", "Spoilage"], axis=1)
y_reg = food_encoded["Surplus_kg"]
y_clf = food_encoded["Spoilage"]

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)

clf_model = DecisionTreeClassifier()
clf_model.fit(X_train, y_train_clf)

# -----------------------------
# KMeans for NGO Clustering
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
ngo["Cluster"] = kmeans.fit_predict(ngo[["Latitude", "Longitude"]])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Food Waste AI", layout="centered")

# -----------------------------
# User Inputs
# -----------------------------
event = st.selectbox("Event Type", food["Event_Type"].unique())
guests = st.number_input("Number of Guests", min_value=50, max_value=1000)
food_prepared = st.number_input("Food Prepared (kg)", min_value=10, max_value=500)
temp = st.number_input("Temperature (°C)", min_value=20, max_value=45)
hours = st.number_input("Hours Passed", min_value=1, max_value=12)

event_lat = st.number_input("📍 Event Latitude", value=19.1000)
event_lon = st.number_input("📍 Event Longitude", value=72.9000)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # Prepare input dictionary
    input_dict = {
        "Guests": guests,
        "Food_Prepared_kg": food_prepared,
        "Temperature": temp,
        "Hours_Passed": hours
    }

    # Encode event type
    for col in X.columns:
        if col.startswith("Event_Type_"):
            input_dict[col] = 1 if col == f"Event_Type_{event}" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[X.columns]

    # Predictions
    surplus = reg_model.predict(input_df)[0]
    spoilage = clf_model.predict(input_df)[0]

    # -----------------------------
    # Display Results
    # -----------------------------
    col1, col2 = st.columns(2)
    col1.metric("📈 Predicted Surplus (kg)", f"{surplus:.2f}")
    risk_text = "High" if spoilage == 1 else "Low"
    col2.metric("⚠ Spoilage Risk", risk_text)

    st.markdown("### 📊 Surplus Level Indicator")
    st.progress(min(int(surplus), 100))

    # Bar Chart
    st.markdown("### 📉 Surplus Visualization")
    fig, ax = plt.subplots()
    ax.bar(["Predicted Surplus"], [surplus])
    ax.set_ylabel("Kg")
    st.pyplot(fig)

    # -----------------------------
    # Find Nearest NGO
    # -----------------------------
    st.markdown("### 🏢 Recommended NGO Details")

    user_location = (event_lat, event_lon)

    def find_nearest_ngo(user_location):
        lat1, lon1 = user_location

        ngo_locations["distance"] = np.sqrt(
            (ngo_locations["lat"] - lat1) ** 2 +
            (ngo_locations["lon"] - lon1) ** 2
        )

        nearest = ngo_locations.loc[ngo_locations["distance"].idxmin()]
        return nearest

    nearest_ngo = find_nearest_ngo(user_location)

    st.success(f"🏢 NGO Name: {nearest_ngo['ngo_name']}")
    st.info(f"📞 Contact Number: {nearest_ngo['phone']}")
    st.write(f"📍 Distance: {nearest_ngo['distance']:.4f} units")

    # Map - Nearest NGO
    nearest_map = pd.DataFrame({
        "latitude": [nearest_ngo["lat"]],
        "longitude": [nearest_ngo["lon"]]
    })

    st.markdown("### 📍 Nearest NGO Location")
    st.map(nearest_map)

    # Map - All NGOs
    st.markdown("### 📍 All NGO Locations")
    st.map(ngo_locations.rename(columns={"lat": "latitude", "lon": "longitude"}))

    # -----------------------------
    # Action Suggestions
    # -----------------------------
    if spoilage == 1:
        st.error("🚨 High Spoilage Risk! Immediate Redistribution Required.")
        st.markdown("### 🚚 Suggested Action:")
        st.markdown("- Contact nearest NGO immediately")
        st.markdown("- Arrange transport within 1 hour")
        st.markdown("- Avoid storing at room temperature")
    else:
        st.success("✅ Low Spoilage Risk. Safe for Redistribution.")
        st.markdown("### 📦 Suggested Action:")
        st.markdown("- Safe to distribute within next few hours")
        st.markdown("- Maintain hygienic storage conditions")

        st.subheader("📍 NGO Clusters:")
        st.write(ngo.groupby("Cluster")["NGO_Name"].apply(list))



