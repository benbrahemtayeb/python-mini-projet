import requests
import streamlit as st

st.set_page_config(page_title="Sentiment Multi-source", layout="centered")

STYLE = {
    "positif": (":)", "#2e9e5b"),
    "negatif": (":(", "#d64545"),
    "neutre": (":|", "#7a869a"),
}

st.title("Customer Review Prediction")
st.caption("Texte, image (OCR) ou audio (ASR)")

api_url = st.sidebar.text_input("URL de l'API", "http://127.0.0.1:8000")
if st.sidebar.button("Tester la connexion"):
    try:
        st.sidebar.success(requests.get(f"{api_url}/health", timeout=5).json())
    except Exception as exc:
        st.sidebar.error(f"API injoignable : {exc}")


def show_result(data: dict) -> None:
    emoji, color = STYLE[data["sentiment_predit"]]
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;background:{color}22;"
        f"border:2px solid {color};text-align:center;'>"
        f"<span style='font-size:44px'>{emoji}</span><br>"
        f"<span style='font-size:28px;font-weight:700;color:{color}'>"
        f"{data['sentiment_predit'].upper()}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(f"**Confiance : {data['confiance']:.0%}**")
    st.progress(min(max(data["confiance"], 0.0), 1.0))
    st.write(f"**Source :** {data['source']}")
    st.text_area("Texte extrait / analysé", data["texte_extrait"], height=100, disabled=True)
    if data.get("probabilites"):
        st.bar_chart(data["probabilites"])


def call_api(endpoint: str, **kwargs) -> None:
    try:
        resp = requests.post(f"{api_url}{endpoint}", timeout=180, **kwargs)
    except requests.exceptions.RequestException as exc:
        st.error(f"Impossible de joindre l'API : {exc}")
        return
    if resp.status_code == 200:
        show_result(resp.json())
    else:
        st.error(f"Erreur {resp.status_code} : {resp.json().get('detail', resp.text)}")


tab_text, tab_img, tab_audio = st.tabs([" Texte", " Image", " Audio"])

with tab_text:
    txt = st.text_area("Saisissez un avis client", height=130, key="txt")
    if st.button("Analyser le texte", type="primary"):
        call_api("/predict/text", json={"texte": txt})

with tab_img:
    img = st.file_uploader("Importez une capture d'écran", type=["png", "jpg", "jpeg"], key="img")
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyser l'image", type="primary"):
            with st.spinner("OCR en cours..."):
                call_api("/predict/image", files={"file": (img.name, img.getvalue(), img.type)})

with tab_audio:
    aud = st.file_uploader("Importez un message vocal", type=["wav", "mp3"], key="aud")
    if aud:
        st.audio(aud.getvalue())
        if st.button("Analyser l'audio", type="primary"):
            with st.spinner("Transcription en cours (le 1er appel télécharge le modèle)..."):
                call_api("/predict/audio", files={"file": (aud.name, aud.getvalue(), aud.type)})
