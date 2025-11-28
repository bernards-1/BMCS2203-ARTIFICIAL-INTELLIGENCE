import cv2
import streamlit as st
import time
from ultralytics import YOLO

# ============  ============
st.set_page_config(page_title="YOLO V8 vs V11 ", layout="wide")

# left right cam
col1, col2 = st.columns(2)
with col1:
    st.subheader("🟦 YOLOv8 ")
    camera_left = st.empty()

with col2:
    st.subheader("🟧 YOLOv11 ")
    camera_right = st.empty()

# ============ two YOLO module ============
model_v8_path = "bestv8.pt"
model_v11_path = "bestv11.pt"

try:
    model_v8 = YOLO(model_v8_path)
    st.success(f"✅ load success: {model_v8_path}")

    model_v11 = YOLO(model_v11_path)
    st.success(f"✅ load success: {model_v11_path}")

except Exception as e:
    st.error(f"❌ module load : {e}")
    st.stop()

# ============ open cam ============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ cannot open cam")
    st.stop()

# ============ loop ============
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Unable to read camera")
        break

    # -------- YOLOv8  --------
    results_v8 = model_v8(frame, stream=True)
    for r in results_v8:
        frame_v8 = r.plot()

    # -------- YOLOv11  --------
    results_v11 = model_v11(frame, stream=True)
    for r in results_v11:
        frame_v11 = r.plot()

    # left YOLOv8 
    camera_left.image(frame_v8, channels="BGR", use_container_width=True)

    # right YOLOv11 
    camera_right.image(frame_v11, channels="BGR", use_container_width=True)

    time.sleep(0.03)

cap.release()
