

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Phone Price Predictor", page_icon="📱", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(45deg, #ff4b4b, #ff7676);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    .prediction-card {
        padding: 30px;
        background-color: white;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #eee;
        margin-top: 20px;
    }
    .price-text {
        color: #ff4b4b;
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_assets():
    # Make sure these files are in your project folder!
    model = pickle.load(open('model_phone.pkl', 'rb'))
    df = pd.read_csv("used_phone.csv")
    return model, df

model, df = load_assets()



# Preparation for Selectboxes
brand_model_map = df.groupby("brand")["model"].unique().to_dict()
brand_list = sorted(list(brand_model_map.keys()))


# st.markdown("""
# <style>
# .title {
#     text-align: center;
#     font-size: 38px;
#     font-weight: 700;
#     color: red;
#     background:black;
#     padding: 20px;
#     border-radius: 15px;
#     box-shadow: 0 8px 25px rgba(560, 124, 255, 0.5);
#     text-shadow: 0 0 10px #00ffff,
#                 0 0 20px #00ffff,
#                 0 0 30px #00ffff;
# }
# </style>

# <div class="title">
#     🚀 USED PHONE VALUE ESTIMATIOR 🚀
# </div>
# """, unsafe_allow_html=True)






# --- UI LAYOUT ---
st.title("📱 USED PHONE VALUE ESTIMATIOR")
st.markdown("Enter the device details below to get an instant AI-powered valuation.")
st.write("---")



# Main Container for inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Identity")
    selected_brand = st.selectbox("Brand Name", brand_list)
    selected_model = st.selectbox("Model Name", sorted(brand_model_map[selected_brand]))
    condition = st.selectbox("Physical Condition", options=sorted(df["condition"].unique()))
    age = st.selectbox("Age of Device (Years)", options=sorted(df["age_years"].unique()))

with col2:
    st.subheader("⚙️ Technical Specs")
    ram = st.selectbox("RAM Capacity (GB)", options=sorted(df["ram_gb"].unique()))
    storage = st.selectbox("Storage Capacity (GB)", options=sorted(df["storage_gb"].unique()))
    battery_health = st.selectbox("Battery Health (%)", options=sorted(df["battery_health"].unique(), reverse=True))
    original_price = st.number_input("Original Purchase Price ($)", min_value=100, value=15000, step=100)



st.markdown("---")



# --- PREDICTION LOGIC ---
# Fitting Encoders (In production, these should be pickled alongside the model)
le_brand = LabelEncoder().fit(df["brand"])
le_model = LabelEncoder().fit(df["model"])
le_condition = LabelEncoder().fit(df["condition"])



# Center the button using columns
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    if st.button("Calculate Market Value"):
        # Encoding inputs
        brand_encoded = le_brand.transform([selected_brand])[0]
        model_encoded = le_model.transform([selected_model])[0]
        condition_encoded = le_condition.transform([condition])[0]

        input_data = pd.DataFrame({
            "brand": [brand_encoded],
            "model": [model_encoded],
            "ram_gb": [ram],
            "storage_gb": [storage],
            "condition": [condition_encoded],
            "battery_health": [battery_health],
            "age_years": [age],
            "original_price": [original_price]
        })


        # Get Prediction
        prediction = model.predict(input_data)[0]


        # Display Result
        st.balloons()
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: #555;'>Suggested Listing Price</h3>
                <div class="price-text">${int(prediction):,}</div>
                <p style='color: #888;'>Estimated for a {selected_brand} {selected_model} in {condition} condition.</p>
            </div>
        """, unsafe_allow_html=True)


st.markdown("""
<style>
.developer {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    background: #1e1e1e;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0, 255, 255, 0.5);
    text-shadow: 0 0 10px #00ffff,
                0 0 20px #00ffff,
                0 0 30px #00ffff;
}
</style>

<div class="developer">
    🚀 Developed by Yananguntikar Meesal 🚀
</div>
""", unsafe_allow_html=True)
