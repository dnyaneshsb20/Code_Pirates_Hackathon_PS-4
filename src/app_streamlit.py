# src/app_streamlit.py
import streamlit as st
import json
import os
from PIL import Image

st.set_page_config(page_title="Airdopes Assembly Verifier", layout="wide")
st.title("Airdopes Assembly Verifier — Demo")

uploaded = st.file_uploader("Upload verification_result.json (from pipeline)", type=["json"])
if uploaded:
    data = json.load(uploaded)
    st.markdown(f"**Video analyzed:** `{data.get('video')}`")
    ver = data.get("verification", {})
    frames = data.get("frames", [])
    vtexts = data.get("vllm_texts", {})

    # Show each step in grid
    steps = list(ver.items())
    cols = st.columns(2)
    idx = 0
    for step_idx, info in steps:
        col = cols[idx % 2]
        with col:
            st.subheader(f"Step {step_idx}: {info['expected']}")
            status = info.get("status", "uncertain")
            if status == "done":
                st.success("COMPLETED")
            elif status == "missing":
                st.error("MISSING")
            else:
                st.warning("UNCERTAIN")
            ef = info.get("evidence_frame")
            if ef and os.path.exists(ef):
                try:
                    img = Image.open(ef)
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.write("Could not load evidence image:", ef)
            st.write("VLLM note:", info.get("note"))
        idx += 1

    st.markdown("---")
    st.subheader("Raw VLLM responses (sample frames)")
    for fpath, txt in list(vtexts.items())[:10]:
        st.markdown(f"**{os.path.basename(fpath)}**: {txt}")

else:
    st.info("Run the pipeline first to create verification_result.json, then upload it here.")
