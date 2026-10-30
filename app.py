import base64
import io
import os
import traceback

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from model_utils import (
    build_overlay_with_contours,
    build_summary_figure,
    heatmap_to_rgb,
    load_ulcare_model,
    make_gradcam_plusplus,
    predict_with_tta,
    to_data_url,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ULCARE_V3.keras")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STAGE_NAMES = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]

st.set_page_config(
    page_title="ULCARE - AI Diabetic Foot Ulcer Screening",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's own chrome (menu/header/footer/padding) and pin the component's
# iframe to the full browser viewport. index.html's internal CSS (html/body height:100%
# + overflow-y:auto) is what makes the page scroll *inside* that pinned iframe — see the
# comment above reportFrame() in index.html for the full explanation of why this matters
# for the mobile sidebar's position:fixed overlay.
st.markdown(
    """
    <style>
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
        div.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
        iframe[title="ulcare_app"] {
            position: fixed !important;
            top: 0; left: 0;
            width: 100vw !important;
            height: 100dvh !important;
            border: none !important;
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model ULCARE...")
def get_model():
    return load_ulcare_model(MODEL_PATH)


_component_func = components.declare_component("ulcare_app", path=FRONTEND_DIR)


def ulcare_component(result=None, key=None):
    return _component_func(result=result, key=key, default=None)


def decode_data_url(data_url):
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    raw = base64.b64decode(b64data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def run_prediction(model, pil_img):
    probs, stage = predict_with_tta(model, pil_img)
    heatmap = make_gradcam_plusplus(model, pil_img, stage)
    overlay_img = build_overlay_with_contours(pil_img, heatmap)
    figure_img = build_summary_figure(pil_img, heatmap, overlay_img, probs, stage, STAGE_NAMES)
    return {
        "stage": int(stage),
        "probabilities": [float(p) for p in probs],
        "photo": to_data_url(pil_img),
        "heatmap": to_data_url(heatmap_to_rgb(heatmap)),
        "overlay": to_data_url(overlay_img),
        "figure": to_data_url(figure_img),
    }


def main():
    if "ulcare_result" not in st.session_state:
        st.session_state.ulcare_result = None
    if "ulcare_last_id" not in st.session_state:
        st.session_state.ulcare_last_id = None

    model = get_model()

    value = ulcare_component(result=st.session_state.ulcare_result, key="ulcare_main")

    if isinstance(value, dict) and value.get("action") == "predict":
        req_id = value.get("id")
        if req_id and req_id != st.session_state.ulcare_last_id:
            st.session_state.ulcare_last_id = req_id
            try:
                pil_img = decode_data_url(value["image"])
                payload = run_prediction(model, pil_img)
                result = {"id": req_id, "action": "predict", **payload}
            except Exception:
                traceback.print_exc()  # tampil di terminal / Manage app > Logs
                result = {
                    "id": req_id,
                    "action": "predict",
                    "error": "Analisis gagal di server. Coba lagi atau periksa log aplikasi.",
                }
            st.session_state.ulcare_result = result
            st.rerun()


if __name__ == "__main__":
    main()
