import streamlit as st
import requests
import base64
import os
import time
from pathlib import Path
from PIL import Image
import io

st.set_page_config(page_title="Human Detection Validator", page_icon="👤", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CPU_THREADS_MIN = int(os.getenv("UI_CPU_THREADS_MIN", "1"))
CPU_THREADS_MAX = int(os.getenv("UI_CPU_THREADS_MAX", "64"))
CPU_THREADS_DEFAULT = int(os.getenv("UI_CPU_THREADS_DEFAULT", "32"))

st.title("👤 Human Detection Validator")
st.markdown(f"**API Endpoint:** `{API_BASE_URL}`")

def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

def call_detection_api(image_base64: str, device: str = "cuda", cpu_threads: int | None = None) -> tuple[dict, float]:
    start_time = time.time()
    payload: dict = {"image_data": image_base64, "device": device}
    if cpu_threads is not None:
        payload["cpu_threads"] = cpu_threads
    response = requests.post(
        f"{API_BASE_URL}/detect",
        json=payload,
        timeout=10
    )
    response.raise_for_status()
    elapsed_time = time.time() - start_time
    return response.json(), elapsed_time

def display_result(result: dict, image: Image.Image, elapsed_time: float | None = None):
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.subheader("Detection Result")
        
        human_detected = result.get("humanDetected", False)
        
        if human_detected:
            st.success("✅ Human Detected")
        else:
            st.info("❌ No Human Detected")
        
        st.metric("Max Confidence", f"{result.get('maxConfidence', 0.0):.2%}")
        st.metric("Bounding Boxes", len(result.get("boundingBoxes", [])))
        if elapsed_time is not None:
            st.metric("Analysis Time", f"{elapsed_time*1000:.0f}ms")
        
        if result.get("boundingBoxes"):
            st.subheader("Bounding Boxes")
            for i, bbox in enumerate(result["boundingBoxes"], 1):
                with st.expander(f"Box {i} - Confidence: {bbox['confidence']:.2%}"):
                    st.json(bbox)

st.header("Upload Your Image")

device_option = st.selectbox(
    "Select Device",
    ["cpu", "cuda"],
    index=0,
    help="Choose whether to run inference on GPU (cuda) or CPU"
)

cpu_threads = None
if device_option == "cpu":
    cpu_threads = st.number_input(
        "CPU Threads",
        min_value=CPU_THREADS_MIN,
        max_value=CPU_THREADS_MAX,
        value=CPU_THREADS_DEFAULT,
        step=1,
        help=f"Number of CPU threads for inference (range: {CPU_THREADS_MIN}-{CPU_THREADS_MAX})"
    )

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff"],
    help="Upload an image to test human detection"
)

if uploaded_file:
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize session state for results
        if "result" not in st.session_state:
            st.session_state.result = None
            st.session_state.elapsed_time = None
            st.session_state.result_image = None
        
        if st.button("🔍 Detect Humans", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                image_base64 = encode_image(image_bytes)
                result, elapsed_time = call_detection_api(image_base64, device=device_option, cpu_threads=cpu_threads)
                st.session_state.result = result
                st.session_state.elapsed_time = elapsed_time
                st.session_state.result_image = image
        
        # Display results if they exist in session state
        if st.session_state.result is not None and st.session_state.result_image is not None:
            display_result(st.session_state.result, st.session_state.result_image, st.session_state.elapsed_time)
    except Exception as e:
        st.error(f"Error: {str(e)}")
