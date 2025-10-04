# app_streamlit.py
import streamlit as st
import os
from src.pipeline import run_pipeline

st.set_page_config(page_title="Assembly Verification", layout="wide")
st.title("🎥 Assembly Verification from Video Upload")

# Golden steps
golden_steps = [
    "Step 1: Preparation – place all items on table",
    "Step 2: Open the charging case fully",
    "Step 3: Insert left earbud into left slot",
    "Step 4: Insert right earbud into right slot",
    "Step 5: Close the charging case completely",
    "Step 6: Connect charging cable; LED should light"
]

st.info("""
👷 **Instructions for Demo**
1. Upload an assembly process video.
2. The system will analyze the video step by step.
3. Verification results will be displayed below.
""")

# ---- File Uploader ----
uploaded_video = st.file_uploader("📂 Upload an assembly video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save video to local file
    os.makedirs("data", exist_ok=True)
    video_path = "data/uploaded.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    # Run pipeline on uploaded video
    with st.spinner("🔍 Analyzing video, please wait..."):
        result = run_pipeline(video_path, "out_uploaded", golden_steps)

    st.success("✅ Analysis complete!")

    # ---- Show Results ----
    st.subheader("📊 Verification Results")

    for step, info in result["verification"].items():
        status = info["status"]
        expected = info["expected"]

        if status == "done":
            st.success(f"✅ {expected}")
        elif status == "missing":
            st.error(f"⚠️ {expected} (Missing)")
        elif status == "out_of_order":
            st.warning(f"🚨 {expected} (Out of order)")
        else:
            st.info(f"❓ {expected} (Unclear)")

    # ---- Debug / Optional ----
    with st.expander("🔎 Raw Verification Data"):
        st.json(result["verification"])
