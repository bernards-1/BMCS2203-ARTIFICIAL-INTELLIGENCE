import cv2
import streamlit as st
import time
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd

# =============================
st.set_page_config(page_title="YOLO V8 vs V11", layout="wide")

COOLDOWN_SECONDS = 2.0  # ⭐ 同一物体冷却时间（秒）

# ===== 初始化 session memory =====
if "conf_v8_mem" not in st.session_state:
    st.session_state.conf_v8_mem = defaultdict(list)

if "conf_v11_mem" not in st.session_state:
    st.session_state.conf_v11_mem = defaultdict(list)

# ⭐ 记录每个物体上次被计数的时间
if "last_seen_v8" not in st.session_state:
    st.session_state.last_seen_v8 = {}

if "last_seen_v11" not in st.session_state:
    st.session_state.last_seen_v11 = {}

# ===== Reset 按钮 =====
if st.button("🔄 Reset Bar Chart"):
    st.session_state.conf_v8_mem.clear()
    st.session_state.conf_v11_mem.clear()
    st.session_state.last_seen_v8.clear()
    st.session_state.last_seen_v11.clear()
    st.success("Bar chart data reset!")

# ===== UI =====
col1, col2 = st.columns(2)
with col1:
    st.subheader("🟦 YOLOv8")
    camera_left = st.empty()
    st.markdown("**Count**")
    count_chart_v8 = st.empty()
    st.markdown("**Accuracy (0–1.0)**")
    acc_chart_v8 = st.empty()

with col2:
    st.subheader("🟧 YOLOv11")
    camera_right = st.empty()
    st.markdown("**Count**")
    count_chart_v11 = st.empty()
    st.markdown("**Accuracy (0–1.0)**")
    acc_chart_v11 = st.empty()

# =============================
model_v8 = YOLO("bestv8.pt")
model_v11 = YOLO("bestv11.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ cannot open cam")
    st.stop()

# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # -------- YOLOv8 --------
    results_v8 = model_v8(frame, stream=True)
    for r in results_v8:
        frame_v8 = r.plot()
        if r.boxes is not None:
            for c, conf in zip(r.boxes.cls, r.boxes.conf):
                label = model_v8.names[int(c)]

                last_time = st.session_state.last_seen_v8.get(label, 0)
                if now - last_time >= COOLDOWN_SECONDS:
                    st.session_state.conf_v8_mem[label].append(float(conf))
                    st.session_state.last_seen_v8[label] = now

    # -------- YOLOv11 --------
    results_v11 = model_v11(frame, stream=True)
    for r in results_v11:
        frame_v11 = r.plot()
        if r.boxes is not None:
            for c, conf in zip(r.boxes.cls, r.boxes.conf):
                label = model_v11.names[int(c)]

                last_time = st.session_state.last_seen_v11.get(label, 0)
                if now - last_time >= COOLDOWN_SECONDS:
                    st.session_state.conf_v11_mem[label].append(float(conf))
                    st.session_state.last_seen_v11[label] = now

    # ===== 显示画面 =====
    camera_left.image(frame_v8, channels="BGR", use_container_width=True)
    camera_right.image(frame_v11, channels="BGR", use_container_width=True)

    # ===== Bar Charts =====
    # YOLOv8
    if st.session_state.conf_v8_mem:
        count_v8 = {k: len(v) for k, v in st.session_state.conf_v8_mem.items()}
        avg_v8 = {k: sum(v) / len(v) for k, v in st.session_state.conf_v8_mem.items()}

        count_chart_v8.bar_chart(
            pd.DataFrame(count_v8.items(), columns=["Object", "Count"]).set_index("Object")
        )
        acc_chart_v8.bar_chart(
            pd.DataFrame(avg_v8.items(), columns=["Object", "Accuracy"]).set_index("Object")
        )

    # YOLOv11
    if st.session_state.conf_v11_mem:
        count_v11 = {k: len(v) for k, v in st.session_state.conf_v11_mem.items()}
        avg_v11 = {k: sum(v) / len(v) for k, v in st.session_state.conf_v11_mem.items()}

        count_chart_v11.bar_chart(
            pd.DataFrame(count_v11.items(), columns=["Object", "Count"]).set_index("Object")
        )
        acc_chart_v11.bar_chart(
            pd.DataFrame(avg_v11.items(), columns=["Object", "Accuracy"]).set_index("Object")
        )

    time.sleep(0.03)

cap.release()
