import streamlit as st
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import io

from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList

# =============================
# 1. 基础配置
# =============================
st.set_page_config(page_title="YOLO V8 vs V11 ", layout="wide")
COOLDOWN_SECONDS = 2.0 

# =============================
# 2. 初始化 Session State
# =============================
if "conf_v8_mem" not in st.session_state:
    st.session_state.conf_v8_mem = defaultdict(list)
if "conf_v11_mem" not in st.session_state:
    st.session_state.conf_v11_mem = defaultdict(list)
if "last_seen_v8" not in st.session_state:
    st.session_state.last_seen_v8 = {}
if "last_seen_v11" not in st.session_state:
    st.session_state.last_seen_v11 = {}

# =============================
# 3. 侧边栏
# =============================
st.sidebar.title("🔧 Control Panel")

run_detection = st.sidebar.checkbox("🟢 Start / Restart Detection", value=True)

if st.sidebar.button("🔄 Reset Data"):
    st.session_state.conf_v8_mem.clear()
    st.session_state.conf_v11_mem.clear()
    st.session_state.last_seen_v8.clear()
    st.session_state.last_seen_v11.clear()
    st.sidebar.success("Data reset!")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**V8 Records:** {sum(len(v) for v in st.session_state.conf_v8_mem.values())}")
st.sidebar.markdown(f"**V11 Records:** {sum(len(v) for v in st.session_state.conf_v11_mem.values())}")


def get_excel_data():
    # 1. 准备汇总数据
    def prepare_summary_df(mem):
        if not mem: return pd.DataFrame()
        count_data = {k: len(v) for k, v in mem.items()}
        # 保留2位小数
        avg_data = {k: round(sum(v)/len(v), 2) for k, v in mem.items()}
        return pd.DataFrame({
            "Object": list(count_data.keys()),
            "Count": list(count_data.values()),
            "Avg_Accuracy": list(avg_data.values())
        })



    df_v8_sum = prepare_summary_df(st.session_state.conf_v8_mem)
    df_v11_sum = prepare_summary_df(st.session_state.conf_v11_mem)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        
        # --- 柱状图 (带数字标签) ---
        def add_bar_charts(sheet_name, df):
            wb = writer.book
            ws = writer.sheets[sheet_name]
            
            # Count Chart
            chart1 = BarChart()
            chart1.title = "Object Count"
            chart1.y_axis.title = "Count"
            chart1.style = 2  
            chart1.height = 10 
            chart1.width = 18
            chart1.y_axis.majorGridlines = None 

            # 只显示数字
            chart1.dataLabels = DataLabelList()
            chart1.dataLabels.showVal = True      
            chart1.dataLabels.showSerName = False 
            chart1.dataLabels.showCatName = False 
            
            data = Reference(ws, min_col=2, min_row=1, max_row=len(df)+1)
            cats = Reference(ws, min_col=1, min_row=2, max_row=len(df)+1)
            chart1.add_data(data, titles_from_data=True)
            chart1.set_categories(cats)
            ws.add_chart(chart1, "E2")
            
            # Accuracy Chart
            chart2 = BarChart()
            chart2.title = "Avg Accuracy (0-1.0)"
            chart2.y_axis.title = "Accuracy"
            chart2.style = 2
            chart2.height = 10
            chart2.width = 18
            chart2.y_axis.majorGridlines = None

            # 只显示数字
            chart2.dataLabels = DataLabelList()
            chart2.dataLabels.showVal = True      
            chart2.dataLabels.showSerName = False 
            chart2.dataLabels.showCatName = False 
            
            data2 = Reference(ws, min_col=3, min_row=1, max_row=len(df)+1)
            chart2.add_data(data2, titles_from_data=True)
            chart2.set_categories(cats)
            ws.add_chart(chart2, "E22") 


        # === 写入数据 & 画图 (只保留 Summary) ===
        if not df_v8_sum.empty:
            df_v8_sum.to_excel(writer, sheet_name="YOLOv8_Summary", index=False)
            add_bar_charts("YOLOv8_Summary", df_v8_sum)
        else:
            pd.DataFrame({"Msg": ["No Data"]}).to_excel(writer, sheet_name="YOLOv8_Summary", index=False)
            

        if not df_v11_sum.empty:
            df_v11_sum.to_excel(writer, sheet_name="YOLOv11_Summary", index=False)
            add_bar_charts("YOLOv11_Summary", df_v11_sum)
        else:
            pd.DataFrame({"Msg": ["No Data"]}).to_excel(writer, sheet_name="YOLOv11_Summary", index=False)

  
    output.seek(0)
    return output

# 下载按钮
if st.session_state.conf_v8_mem or st.session_state.conf_v11_mem:
    excel_file = get_excel_data()
    st.sidebar.download_button(
        label="📥 Download Excel Report",
        data=excel_file,
        file_name="yolo_report_clean.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =============================
# 4. 主界面布局
# =============================
st.title("YOLOv8 vs YOLOv11 Real-Time")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟦 YOLOv8")
    camera_left = st.empty()
    st.write("---")
    st.markdown("**Count**")
    count_chart_v8 = st.empty()
    st.markdown("**Avg Accuracy**")
    acc_chart_v8 = st.empty()
    st.markdown("**📈 Confidence Trend**") 
    line_chart_v8 = st.empty()

with col2:
    st.subheader("🟧 YOLOv11")
    camera_right = st.empty()
    st.write("---")
    st.markdown("**Count**")
    count_chart_v11 = st.empty()
    st.markdown("**Avg Accuracy**")
    acc_chart_v11 = st.empty()
    st.markdown("**📈 Confidence Trend**") 
    line_chart_v11 = st.empty()

# =============================
# 5. 辅助函数：绘制图表 
# =============================
def render_charts():
    if st.session_state.conf_v8_mem:
        count_v8 = {k: len(v) for k, v in st.session_state.conf_v8_mem.items()}
        avg_v8 = {k: sum(v) / len(v) for k, v in st.session_state.conf_v8_mem.items()}
        
        count_chart_v8.bar_chart(pd.DataFrame(count_v8.items(), columns=["Object", "Count"]).set_index("Object"))
        acc_chart_v8.bar_chart(pd.DataFrame(avg_v8.items(), columns=["Object", "Accuracy"]).set_index("Object"))
        
        df_trend_v8 = pd.DataFrame.from_dict(st.session_state.conf_v8_mem, orient='index').T
        line_chart_v8.line_chart(df_trend_v8)
    
    if st.session_state.conf_v11_mem:
        count_v11 = {k: len(v) for k, v in st.session_state.conf_v11_mem.items()}
        avg_v11 = {k: sum(v) / len(v) for k, v in st.session_state.conf_v11_mem.items()}
        
        count_chart_v11.bar_chart(pd.DataFrame(count_v11.items(), columns=["Object", "Count"]).set_index("Object"))
        acc_chart_v11.bar_chart(pd.DataFrame(avg_v11.items(), columns=["Object", "Accuracy"]).set_index("Object"))
        
        df_trend_v11 = pd.DataFrame.from_dict(st.session_state.conf_v11_mem, orient='index').T
        line_chart_v11.line_chart(df_trend_v11)

# =============================
# 6. 模型加载
# =============================
@st.cache_resource
def load_models():
    try:
        return YOLO("bestv8.pt"), YOLO("bestv11.pt")
    except:
        return YOLO("yolov8n.pt"), YOLO("yolo11n.pt")

model_v8, model_v11 = load_models()

# =============================
# 7. 主循环
# =============================
render_charts()

if run_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot open camera")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame")
                break

            now = time.time()

            # YOLOv8
            results_v8 = model_v8(frame, stream=True, verbose=False)
            frame_v8 = frame.copy()
            for r in results_v8:
                frame_v8 = r.plot()
                if r.boxes:
                    for c, conf in zip(r.boxes.cls, r.boxes.conf):
                        label = model_v8.names[int(c)]
                        if now - st.session_state.last_seen_v8.get(label, 0) >= COOLDOWN_SECONDS:
                            st.session_state.conf_v8_mem[label].append(float(conf))
                            st.session_state.last_seen_v8[label] = now

            # YOLOv11
            results_v11 = model_v11(frame, stream=True, verbose=False)
            frame_v11 = frame.copy()
            for r in results_v11:
                frame_v11 = r.plot()
                if r.boxes:
                    for c, conf in zip(r.boxes.cls, r.boxes.conf):
                        label = model_v11.names[int(c)]
                        if now - st.session_state.last_seen_v11.get(label, 0) >= COOLDOWN_SECONDS:
                            st.session_state.conf_v11_mem[label].append(float(conf))
                            st.session_state.last_seen_v11[label] = now

            camera_left.image(frame_v8, channels="BGR", use_container_width=True)
            camera_right.image(frame_v11, channels="BGR", use_container_width=True)

            render_charts()
            time.sleep(0.03)

        cap.release()
else:
    st.info("⏸ Detection stopped. Check the box to Restart.")