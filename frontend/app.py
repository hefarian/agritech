import pandas as pd
import requests
import streamlit as st

API_URL = st.sidebar.text_input("URL API", value="http://localhost:8000")

st.set_page_config(page_title="Agritech Answers", layout="wide")
st.title("Agritech Answers")
st.caption("Prediction de rendement et recommandation de culture")


@st.cache_data(ttl=60)
def load_metadata() -> dict:
    response = requests.get(f"{API_URL}/metadata", timeout=10)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


try:
    metadata = load_metadata()
    crop_options = metadata["crops"]
    area_options = metadata["areas"]
except requests.RequestException:
    metadata = None
    crop_options = ["Maize", "Wheat", "Rice, paddy"]
    area_options = ["Albania", "France", "India"]
    st.warning("API indisponible ou modele non charge. Utilisation d'options par defaut.")

mode = st.radio("Mode", options=["Prediction", "Recommendation"], horizontal=True)

left_column, right_column = st.columns(2)
with left_column:
    area = st.selectbox("Pays / zone", area_options)
    year = st.slider("Annee", min_value=1990, max_value=2035, value=2013)
    rainfall = st.slider("Pluie moyenne annuelle (mm)", min_value=0.0, max_value=4000.0, value=1200.0)
with right_column:
    pesticides = st.slider("Pesticides (tonnes)", min_value=0.0, max_value=100000.0, value=1000.0)
    avg_temp = st.slider("Temperature moyenne (C)", min_value=-5.0, max_value=40.0, value=18.0)

context = {
    "Area": area,
    "Year": year,
    "average_rain_fall_mm_per_year": rainfall,
    "pesticides_tonnes": pesticides,
    "avg_temp": avg_temp,
}

if mode == "Prediction":
    crop = st.selectbox("Culture", crop_options)
    if st.button("Predire le rendement", type="primary"):
        try:
            result = post_json("/predict", {**context, "Item": crop})
            st.metric("Rendement predit", f"{result['predicted_yield']:.0f} hg/ha")
        except requests.RequestException as exc:
            st.error(f"Echec de la requete API: {exc}")
else:
    if st.button("Recommander une culture", type="primary"):
        try:
            result = post_json("/recommend", context)
            ranking = pd.DataFrame(result["recommendations"])
            st.bar_chart(ranking.set_index("Item")["predicted_yield"])
            st.dataframe(ranking, use_container_width=True)
        except requests.RequestException as exc:
            st.error(f"Echec de la requete API: {exc}")
