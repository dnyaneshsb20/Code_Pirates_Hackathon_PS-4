import streamlit as st
import cv2
import tempfile
import time
from src.vllm_reasoner import run_vllm_verification

st.set_page_config(page_title="Assembly Verification", layout="wide")
st.title("🔴 Live Assembly Line Verification Demo")

# Golden steps
golden_steps = [
    "Step 1: Preparation – place all items on table",
    "Step 2: Open the charging case fully",
    "Step 3: Insert left earbud into left slot",
    "Step 4: Insert right earbud into right slot",
    "Step 5: Close the charging case completely",
    "Step 6: Connect charging cable; LED should light"
]

# Sidebar progress tracker
progress_placeholder = st.sidebar.empty()
def update_progress(verification):
    with progress_placeholder.container():
        st.subheader("📊 Progress Tracker")
        for step, info in verification.items():
            if not isinstance(info, dict):
                continue
            status = info.get("status", "uncertain")
            expected = info.get("expected", f"Step {step}")

            if status == "done":
                st.markdown(f"✅ **{expected}**")
            elif status == "missing":
                st.markdown(f"⚠️ **{expected}** (Missing)")
            elif status == "out_of_order":
                st.markdown(f"🚨 **{expected}** (Out of order)")
            else:
                st.markdown(f"❓ **{expected}** (Pending/Unclear)")

# Instructions
st.info("""
👷 **Instructions for Demo**
1. Press **Start Verification**.
2. Perform the assembly steps in order.
3. Press **Stop Verification** anytime.
""")

# ---- Session State ----
if "running" not in st.session_state:
    st.session_state.running = False

# Start / Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Verification"):
        st.session_state.running = True
        st.session_state.seen_objects = set()   # CLEAR on start
with col2:
    if st.button("⏹ Stop Verification"):
        st.session_state.running = False
        st.session_state.seen_objects = set()   # CLEAR on stop

# Webcam logic
if st.session_state.running:
    cap = cv2.VideoCapture("http://10.1.230.252:8080/video") 
    stframe = st.empty()
    last_alert_time = 0
    alert_cooldown = 3

    st.success("✅ Verification started! Perform the steps now.")

    # ensure seen_objects exists (session-wide)
    if "seen_objects" not in st.session_state:
        st.session_state.seen_objects = set()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected!")
            break

        # Show live video
        stframe.image(frame, channels="BGR", width="stretch")

        # Save temporary frame
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)

        # Run verification (simulated for demo)
        result = run_vllm_verification([temp_file.name], golden_steps, use_api=False)

        # --- Show current detections in sidebar ---
        detections = result.get("detections", [])
        if detections:
            st.sidebar.write("📷 Current detections:")
            for d in detections:
                # d is {"object": "...", "confidence": 0.xx}
                st.sidebar.write(f"- {d['object']} (conf {d['confidence']:.2f})")
        else:
            st.sidebar.write("📷 Current detections: none")

        # --- Maintain history of seen objects across frames ---
        for d in detections:
            st.session_state.seen_objects.add(d["object"])

        # show history
        st.sidebar.write("👀 Seen this session:", ", ".join(sorted(st.session_state.seen_objects)) or "none")

        # --- Override Step 1 if all required items have been seen across frames ---
        # Practical option: require case + cable + at least one earbud
        # required_all = {"case", "left_earbud", "right_earbud", "cable"}
        # required_loose = {"case", "cable"}  # plus at least one earbud
        # seen = set(st.session_state.seen_objects)

        # if required_all.issubset(seen):
        #     # all perfect
        #     result["verification"]["1"]["status"] = "done"
        #     result["verification"]["1"]["note"] = "All required items observed across frames"
        # elif required_loose.issubset(seen) and ("left_earbud" in seen or "right_earbud" in seen):
        #     # pragmatic fallback for demo: accept one earbud
        #     result["verification"]["1"]["status"] = "done"
        #     result["verification"]["1"]["note"] = "Case + cable + at least one earbud observed across frames"

        # --- Override Step 1: require items in *this* frame ---
        current_seen = {d["object"] for d in detections}

        required_all = {"case", "left_earbud", "right_earbud", "cable"}
        required_loose = {"case", "cable"}  # + at least one earbud

        if required_all.issubset(current_seen):
            result["verification"]["1"]["status"] = "done"
            result["verification"]["1"]["note"] = "All required items detected in current frame"
        elif required_loose.issubset(current_seen) and ("left_earbud" in current_seen or "right_earbud" in current_seen):
            result["verification"]["1"]["status"] = "done"
            result["verification"]["1"]["note"] = "Case + cable + at least one earbud detected in current frame"
        else:
            result["verification"]["1"]["status"] = "missing"
            result["verification"]["1"]["note"] = "Some required items missing in current frame"


        # --- Update sidebar progress tracker (visual step statuses) ---
        update_progress(result["verification"])

        # --- Parse verification results and show toasts/alerts ---
        for step, info in result["verification"].items():
            now = time.time()
            if now - last_alert_time > alert_cooldown:
                if info.get("status") == "done":
                    st.toast(f"✅ Step {step} completed: {info.get('expected')}")
                elif info.get("status") == "missing":
                    st.toast(f"⚠️ Step {step} missed: {info.get('expected')}")
                elif info.get("status") == "out_of_order":
                    st.toast(f"🚨 Step {step} out of order: {info.get('expected')}")
                elif info.get("status") == "uncertain":
                    st.toast(f"❓ Step {step} unclear: {info.get('expected')}")
                last_alert_time = now

        time.sleep(2)  # process every 2 seconds

    cap.release()
    st.warning("⏹ Verification stopped.")
