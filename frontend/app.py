import os
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

PROJECT_NAME = os.getenv("COMPOSE_PROJECT_NAME", "local")

def normalize_api_url(raw_value: str) -> str:
    # Le front tourne en conteneur : "localhost" doit pointer vers le service Docker "api".
    default_port = os.getenv("API_CONTAINER_PORT", "8000")
    value = (raw_value or "").strip()
    if not value:
        value = f"http://api:{default_port}"
    if "://" not in value:
        value = f"http://{value}"

    parsed = urlparse(value)
    host = parsed.hostname or "api"
    port = parsed.port
    scheme = parsed.scheme or "http"

    if host in {"localhost", "127.0.0.1", "0.0.0.0", "api"}:
        host = "api"
        if port is None:
            port = int(default_port)

    if port is None:
        return f"{scheme}://{host}".rstrip("/")
    return f"{scheme}://{host}:{port}".rstrip("/")


# Permet a l'utilisateur de viser une API locale ou distante.
default_api_url = os.getenv("API_URL", f"http://api:{os.getenv('API_CONTAINER_PORT', '8000')}")
raw_api_url = st.sidebar.text_input("URL API", value=default_api_url)
API_URL = normalize_api_url(raw_api_url)
st.sidebar.caption(f"URL API effective: {API_URL}")

st.set_page_config(page_title=f"Agritech Answers - {PROJECT_NAME}", layout="wide")
st.title(f"Agritech Answers - {PROJECT_NAME}")
st.caption("Prediction de rendement et recommandation de culture")


@st.cache_data(ttl=60)
def load_metadata(api_url: str) -> dict:
    # On garde les metadonnees en cache un court instant pour eviter
    # d'appeler l'API a chaque rafraichissement des widgets.
    response = requests.get(f"{api_url}/metadata", timeout=10)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict) -> dict:
    # Petit utilitaire pour centraliser les requetes POST et les erreurs associees.
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


try:
    # Si l'API est disponible, on recupere les vraies cultures et zones.
    metadata = load_metadata(API_URL)
    crop_options = metadata["crops"]
    area_options = metadata["areas"]
except requests.RequestException:
    # Ces valeurs de secours gardent l'interface utilisable meme si l'API n'est pas prete.
    metadata = None
    crop_options = ["Maize", "Wheat", "Rice, paddy"]
    area_options = ["Albania", "France", "India"]
    st.warning("API indisponible ou modele non charge. Utilisation d'options par defaut.")

mode = st.radio("Mode", options=["Prediction", "Recommendation"], horizontal=True)

# On coupe le formulaire en deux colonnes pour garder une page lisible.
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
    # En mode prediction, l'utilisateur choisit une culture et recoit un seul resultat.
    crop = st.selectbox("Culture", crop_options)
    if st.button("Predire le rendement", type="primary"):
        try:
            result = post_json("/predict", {**context, "Item": crop})
            st.metric("Rendement predit", f"{result['predicted_yield']:.0f} hg/ha")
        except requests.RequestException as exc:
            st.error(f"Echec de la requete API: {exc}")
else:
    # En mode recommandation, on envoie seulement le contexte et on affiche un classement.
    if st.button("Recommander une culture", type="primary"):
        try:
            result = post_json("/recommend", context)
            ranking = pd.DataFrame(result["recommendations"])
            st.bar_chart(ranking.set_index("Item")["predicted_yield"])
            st.dataframe(ranking, use_container_width=True)
        except requests.RequestException as exc:
            st.error(f"Echec de la requete API: {exc}")
