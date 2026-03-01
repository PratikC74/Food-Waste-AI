import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------
# Create Food Dataset
# -----------------------------
n = 500

event_types = ["Wedding", "Corporate", "Birthday", "Festival"]

food_data = {
    "Event_Type": np.random.choice(event_types, n),
    "Guests": np.random.randint(50, 1000, n),
    "Food_Prepared_kg": np.random.randint(20, 500, n),
    "Temperature": np.random.randint(20, 45, n),
    "Hours_Passed": np.random.randint(1, 12, n),
}

df = pd.DataFrame(food_data)

# Create realistic surplus logic
df["Surplus_kg"] = (
    df["Food_Prepared_kg"]
    - (df["Guests"] * 0.4)
    + np.random.randint(-20, 20, n)
)

df["Surplus_kg"] = df["Surplus_kg"].apply(lambda x: max(0, x))

# Spoilage rule
df["Spoilage"] = np.where(
    (df["Temperature"] > 35) & (df["Hours_Passed"] > 6),
    1,
    0
)

df.to_csv("food_data.csv", index=False)

print("food_data.csv created successfully!")


# -----------------------------
# Create NGO Dataset
# -----------------------------
ngo_data = {
    "NGO_Name": ["Helping Hands", "FoodCare", "HungerFree",
                 "SevaTrust", "CareFoundation", "MealSupport"],
    "Area": ["Andheri", "Dadar", "Bandra", "Kurla", "Borivali", "Chembur"],
    "Latitude": np.random.uniform(19.0, 19.3, 6),
    "Longitude": np.random.uniform(72.8, 73.0, 6),
    "Capacity_kg": np.random.randint(50, 300, 6)
}

ngo_df = pd.DataFrame(ngo_data)
ngo_df.to_csv("ngo_data.csv", index=False)

print("ngo_data.csv created successfully!")
