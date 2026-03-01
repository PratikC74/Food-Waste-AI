import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# 1️⃣ Page Configuration (FIRST) - ONLY ONCE
st.set_page_config(
    page_title="Food Waste AI",
    page_icon="🍱",
    layout="wide"
)

# 2️⃣ ADD UI STYLING HERE
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

# 3️⃣ Start app UI
st.title("🍱 AI-Based Food Waste Optimization System")
st.markdown("Smart Prediction & Redistribution Platform")


# -----------------------------
# NGO Static Location Data
# -----------------------------
go_locations = pd.DataFrame({
    "ngo_name": ["Helping Hands", "Food For All", "Care & Share"],
    "lat": [19.0760, 19.2183, 19.0330],
    "lon": [72.8777, 72.9781, 72.8650],
    "phone": ["+91 9876543210", "+91 9123456780", "+91 9988776655"]
})

# -----------------------------
# Load Datasets with Error Handling
# -----------------------------
try:
    food = pd.read_csv("food_data.csv")
    ngo = pd.read_csv("ngo_data.csv")
except FileNotFoundError as e:
    st.error(f"❌ Error: Required data files not found - {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading datasets: {e}")
    st.stop()

# Validate required columns
required_food_cols = ["Event_Type", "Surplus_kg", "Spoilage"]
if not all(col in food.columns for col in required_food_cols):
    st.error(f"❌ Error: food_data.csv missing required columns: {required_food_cols}")
    st.stop()

# Check for latitude/longitude columns (case-insensitive)
lat_col = next((col for col in ngo.columns if col.lower() == "latitude"), None)
lon_col = next((col for col in ngo.columns if col.lower() == "longitude"), None)

if lat_col is None or lon_col is None:
    st.error("❌ Error: ngo_data.csv must contain 'latitude' and 'longitude' columns")
    st.stop()

# Normalize column names for consistency
ngo.rename(columns={lat_col: "latitude", lon_col: "longitude"}, inplace=True)

# Check for NGO name column
ngo_name_col = next((col for col in ngo.columns if col.lower() in ["ngo_name", "name"]), None)
if ngo_name_col:
    ngo.rename(columns={ngo_name_col: "ngo_name"}, inplace=True)

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
# Train Models with Caching
# -----------------------------
@st.cache_resource
def train_models():
    """Train ML models with caching to avoid retraining on every rerun"""
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train_reg)
    
    clf_model = DecisionTreeClassifier()
    clf_model.fit(X_train, y_train_clf)
    
    return reg_model, clf_model

reg_model, clf_model = train_models()

# -----------------------------
# KMeans for NGO Clustering
# -----------------------------
@st.cache_resource
def cluster_ngos():
    """Cluster NGO locations with caching"""
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(ngo[["latitude", "longitude"]])
    return clusters

ngo["Cluster"] = cluster_ngos()

# -----------------------------
# User Inputs
# -----------------------------
st.markdown("### 📝 Enter Event Details")
event = st.selectbox("Event Type", food["Event_Type"].unique())
guests = st.number_input("Number of Guests", min_value=50, max_value=1000)
food_prepared = st.number_input("Food Prepared (kg)", min_value=10, max_value=500)
temp = st.number_input("Temperature (°C)", min_value=20, max_value=45)
hours = st.number_input("Hours Passed", min_value=1, max_value=12)

st.markdown("### 📍 Enter Event Location")
event_lat = st.number_input("Event Latitude", value=19.1000, format="%.4f")
event_lon = st.number_input("Event Longitude", value=72.9000, format="%.4f")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict", use_container_width=True):

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
    try:
        surplus = reg_model.predict(input_df)[0]
        spoilage = clf_model.predict(input_df)[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

    # -----------------------------
    # Display Results
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Predicted Surplus (kg)", f"{surplus:.2f}")
    with col2:
        risk_text = "🔴 High" if spoilage == 1 else "🟢 Low"
        st.metric("⚠️ Spoilage Risk", risk_text)

    st.markdown("### 📊 Surplus Level Indicator")
    progress_val = min(max(int(surplus), 0), 100)
    st.progress(progress_val / 100.0)

    # Bar Chart
    st.markdown("### 📊 Surplus Visualization")
    st.bar_chart(pd.DataFrame({"Surplus (kg)": [surplus]}))

    # -----------------------------
    # Find Nearest NGO
    # -----------------------------
    st.markdown("### 🏢 Recommended NGO Details")

    user_location = (event_lat, event_lon)

    def find_nearest_ngo(user_location):
        """Find the nearest NGO based on user location"""
        lat1, lon1 = user_location
        
        ngo_locations_copy = ngo_locations.copy()
        ngo_locations_copy["distance"] = np.sqrt(
            (ngo_locations_copy["lat"] - lat1) ** 2 +
            (ngo_locations_copy["lon"] - lon1) ** 2
        )

        nearest = ngo_locations_copy.loc[ngo_locations_copy["distance"].idxmin()]
        return nearest

    nearest_ngo = find_nearest_ngo(user_location)

    st.success(f"🏢 NGO Name: **{nearest_ngo['ngo_name']}**")
    st.info(f"📞 Contact Number: **{nearest_ngo['phone']}**")
    st.write(f"📍 Distance: **{nearest_ngo['distance']:.4f}** units")

    # Map - Nearest NGO
    nearest_map = pd.DataFrame({
        "latitude": [nearest_ngo["lat}],
        "longitude": [nearest_ngo["lon"]],
        "name": ["Nearest NGO"]
    })

    st.markdown("### 📍 Nearest NGO Location")
    st.map(nearest_map)

    # Map - All NGOs
    st.markdown("### 📍 All NGO Locations")
    all_ngos_map = ngo_locations.copy()
    all_ngos_map.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
    st.map(all_ngos_map)

    # -----------------------------
    # Action Suggestions
    # -----------------------------
    st.markdown("---")
    if spoilage == 1:
        st.error("🚨 HIGH SPOILAGE RISK! IMMEDIATE REDISTRIBUTION REQUIRED")
        st.markdown("### 🚚 Suggested Action:")
        st.markdown("- ⏰ Contact nearest NGO **immediately**")
        st.markdown("- 🚗 Arrange transport within **1 hour**")
        st.markdown("- ❄️ Avoid storing at **room temperature**")
        st.markdown("- 📦 Use **refrigerated containers** if available")
    else:
        st.success("✅ LOW SPOILAGE RISK - Safe for Redistribution")
        st.markdown("### 📦 Suggested Action:")
        st.markdown("- ✓ Safe to distribute within the **next few hours**")
        st.markdown("- ✓ Maintain **hygienic storage conditions**")
        st.markdown("- ✓ Keep in **cool, shaded area**")
        st.markdown("- ✓ Schedule pickup with NGO at your convenience")

    # Display NGO Clusters if data exists
    if "ngo_name" in ngo.columns:
        st.subheader("📍 NGO Clusters:")
        try:
            cluster_groups = ngo.groupby("Cluster")["ngo_name"].apply(list).to_dict()
            for cluster_id, ngo_names in cluster_groups.items():
                st.write(f"**Cluster {cluster_id}:** {', '.join(ngo_names)}")
        except Exception as e:
            st.warning(f"⚠️ Could not display NGO clusters: {e}")
