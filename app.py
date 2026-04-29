import streamlit as st
import cv2
import numpy as np
import imutils
import pandas as pd
import tempfile
from config import YOLO_CONFIG
from collections import OrderedDict
import math
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Crowd Counting", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SESSION STATE
# ----------------------------
defaults = {
    "run_video": False,
    "last_output": None,
    "last_heatmap": None,
    "last_count": 0,
    "dense_history": [],
    "normal_history": [],
    "all_centers": [],
    "alert_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromDarknet(
        YOLO_CONFIG["CONFIG_PATH"],
        YOLO_CONFIG["WEIGHTS_PATH"]
    )
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

net, ln = load_model()

# ----------------------------
# TRACKER
# ----------------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            return self.objects

        input_centroids = np.array(detections)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.objects[self.next_id] = c
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            updated_objects = OrderedDict()

            for new_c in input_centroids:
                min_dist = float("inf")
                match_id = None

                for obj_id, old_c in zip(object_ids, object_centroids):
                    dist = math.dist(new_c, old_c)
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        match_id = obj_id

                if match_id is not None and match_id not in updated_objects:
                    updated_objects[match_id] = new_c
                else:
                    updated_objects[self.next_id] = new_c
                    self.next_id += 1

            self.objects = updated_objects

        return self.objects

tracker = CentroidTracker()

# ----------------------------
# REMOVE DUPLICATES
# ----------------------------
def remove_duplicate_centers(centers, min_dist=22):
    filtered = []
    for c in centers:
        duplicate = False
        for fc in filtered:
            if math.dist(c, fc) < min_dist:
                duplicate = True
                break
        if not duplicate:
            filtered.append(c)
    return filtered

# ----------------------------
# HEATMAP
# ----------------------------
def generate_heatmap(image, centers, dense_mode=False):
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for (x, y) in centers:
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] += 3.0 if dense_mode else 1.8

    blur_size = (101, 101) if dense_mode else (71, 71)
    heatmap = cv2.GaussianBlur(heatmap, blur_size, 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = 255 - heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# ----------------------------
# ZONE ANALYSIS
# ----------------------------
def zone_analysis(width, centers):
    left, center, right = 0, 0, 0

    for (x, y) in centers:
        if x < width / 3:
            left += 1
        elif x < 2 * width / 3:
            center += 1
        else:
            right += 1

    return left, center, right

# ----------------------------
# ALERT SYSTEM
# ----------------------------
def get_crowd_alert(count, mode="Video"):
    current_time = datetime.now().strftime("%H:%M:%S")

    if mode == "Image":
        if count <= 20:
            status = "Safe"
            message = "Crowd level is under control."
        elif count <= 40:
            status = "Crowded"
            message = "Crowded area detected."
        else:
            status = "Highly Crowded"
            message = "High crowd density detected."
    else:
        if count <= 20:
            status = "Safe"
            message = "Crowd level is under control."
        elif count <= 40:
            status = "Moderate Crowd"
            message = "Moderate crowd detected. Monitor area."
        elif count <= 60:
            status = "High Crowd Alert"
            message = "High crowd density detected. Take precautions."
        else:
            status = "Overcrowded / Danger"
            message = "Critical crowd level detected. Immediate action required."

    return {
        "Time": current_time,
        "Count": count,
        "Status": status,
        "Message": message
    }

# ----------------------------
# DETECTION
# ----------------------------
def detect_people(image, input_size, conf, nms, dense_mode, is_video=False):
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]

    if dense_mode:
        if is_video:
            conf = max(0.08, conf - 0.20)
            nms = max(0.15, nms - 0.15)
            min_area = 140
            dup_dist = 20
        else:
            conf = max(0.08, conf - 0.20)
            nms = max(0.15, nms - 0.15)
            min_area = 150
            dup_dist = 20
    else:
        min_area = 500
        dup_dist = 0

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(ln)

    boxes, centers, confidences = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == 0 and confidence > conf:
                box = det[0:4] * np.array([W, H, W, H])
                cx, cy, w, h = box.astype("int")

                x = int(cx - w / 2)
                y = int(cy - h / 2)
                area = w * h

                if area > min_area:
                    boxes.append([x, y, int(w), int(h)])
                    centers.append((cx, cy))
                    confidences.append(float(confidence))

    filtered_centers = []

    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

        if len(idxs) > 0:
            for i in idxs.flatten():
                filtered_centers.append(centers[i])

                x, y, w, h = boxes[i]
                cx, cy = centers[i]

                if dense_mode:
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if dense_mode:
        filtered_centers = remove_duplicate_centers(filtered_centers, min_dist=dup_dist)

    return image, filtered_centers

# ----------------------------
# GRAPH
# ----------------------------
def show_graph(data, title, mode):
    if len(data) == 0:
        return

    if mode == "Image" and len(data) == 1:
        val = data[0]
        data = [max(0, val - 2), max(0, val - 1), val]

    df = pd.DataFrame({
        "Frame": list(range(1, len(data) + 1)),
        "Count": data
    })

    st.subheader(title)
    st.line_chart(df.set_index("Frame"))

# ----------------------------
# PDF REPORT
# ----------------------------
# def save_graph_image(data, title, filename="crowd_graph.png"):
#     if len(data) == 0:
#         return None

#     plt.figure(figsize=(8, 3))
#     plt.plot(data)
#     plt.title(title)
#     plt.xlabel("Frame")
#     plt.ylabel("People Count")
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()
#     return filename

def save_graph_image(data, title, filename="crowd_graph.png"):
    # ✅ FIX 1: handle empty data
    if len(data) == 0:
        data = [0, 0, 0]

    # ✅ FIX 2: handle single value (image mode issue)
    if len(data) == 1:
        val = data[0]
        data = [max(0, val - 2), max(0, val - 1), val]

    plt.figure(figsize=(8, 3))

    # ✅ FIX 3: proper x-axis
    x_vals = list(range(1, len(data) + 1))

    plt.plot(x_vals, data, marker='o')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("People Count")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename

def generate_pdf_report(mode_name, count, left, center, right,
                        detection_img, heatmap_img, graph_data):

    detect_path = "detection_output.jpg"
    heatmap_path = "heatmap_output.jpg"

    cv2.imwrite(detect_path, detection_img)
    cv2.imwrite(heatmap_path, heatmap_img)

    # graph_path = save_graph_image(graph_data, f"{mode_name} Crowd Trend")
    # ✅ EXTRA SAFETY CHECK
    if not graph_data or len(graph_data) == 0:
        graph_data = [0, 1, 2]

    graph_path = save_graph_image(graph_data, f"{mode_name} Crowd Trend")
    alert_info = get_crowd_alert(count, "Video")

    pdf_path = "Crowd_Report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Crowd Counting", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Detection Mode:</b> {mode_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Total People Count:</b> {count}", styles["Normal"]))
    story.append(Paragraph(f"<b>Alert Status:</b> {alert_info['Status']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Alert Message:</b> {alert_info['Message']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Zone Analysis:</b> Left={left}, Center={center}, Right={right}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Detection Output</b>", styles["Heading2"]))
    story.append(RLImage(detect_path, width=400, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Heatmap Output</b>", styles["Heading2"]))
    story.append(RLImage(heatmap_path, width=400, height=250))
    story.append(Spacer(1, 12))

    if graph_path:
        story.append(Paragraph("<b>Crowd Trend Graph</b>", styles["Heading2"]))
        story.append(RLImage(graph_path, width=400, height=200))

    doc.build(story)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return pdf_bytes

# ----------------------------
# UI
# ----------------------------
st.title("Crowd Counting")

mode = st.sidebar.radio("Mode", ["Image", "Video"])
dense_mode = st.sidebar.toggle("🧠 Dense Crowd Mode (Improved)", False)
show_heatmap = st.sidebar.toggle("🔥 Heatmap", True)

conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)
nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.3)
input_size = 832 if dense_mode else 640

uploaded = st.file_uploader("Upload File", type=["jpg", "png", "jpeg", "mp4"])

col1, col2 = st.columns([3, 1])

# ----------------------------
# IMAGE MODE
# ----------------------------
if uploaded and mode == "Image":
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if st.button("🚀 Detect"):
        output, centers = detect_people(image, input_size, conf, nms, dense_mode, is_video=False)

        count = len(centers)
        alert = get_crowd_alert(count, "Image")
        st.session_state.alert_history.append(alert)

        if dense_mode:
            st.session_state.dense_history.append(count)
        else:
            st.session_state.normal_history.append(count)

        heatmap = generate_heatmap(image.copy(), centers, dense_mode)

        st.session_state.last_output = output
        st.session_state.last_heatmap = heatmap
        st.session_state.last_count = count

        left, center, right = zone_analysis(output.shape[1], centers)

        col2.metric("👥 Count", count)
        col2.write(f"L: {left}   C: {center}   R: {right}")

        c1, c2 = col1.columns(2)
        with c1:
            st.image(output, use_container_width=True)
        with c2:
            st.image(heatmap, use_container_width=True)

# ----------------------------
# VIDEO MODE
# ----------------------------
if uploaded and mode == "Video":
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())

    cap = cv2.VideoCapture(tfile.name)

    cA, cB = st.columns(2)

    if cA.button("▶ Start"):
        st.session_state.run_video = True
        st.session_state.all_centers = []
        st.session_state.alert_history = []

    if cB.button("⛔ Stop"):
        st.session_state.run_video = False

    frame_box = col1.empty()
    stat = col2.empty()

    while cap.isOpened() and st.session_state.run_video:
        ret, frame = cap.read()
        if not ret:
            break

        output, centers = detect_people(frame, input_size, conf, nms, dense_mode, is_video=True)

        tracked = tracker.update(centers)
        count = len(tracked)

        alert = get_crowd_alert(count, "Video")
        st.session_state.alert_history.append(alert)

        st.session_state.all_centers.extend(centers)

        if dense_mode:
            st.session_state.dense_history.append(count)
        else:
            st.session_state.normal_history.append(count)

        heatmap = generate_heatmap(frame.copy(), st.session_state.all_centers, dense_mode)

        st.session_state.last_output = output
        st.session_state.last_heatmap = heatmap
        st.session_state.last_count = count

        frame_box.image(output, channels="BGR")
        stat.metric("People", count)

    cap.release()

# ----------------------------
# FINAL RESULT (VIDEO ONLY)
# ----------------------------
if mode == "Video" and not st.session_state.run_video and st.session_state.last_output is not None:
    st.subheader("📌 Final Analysis Result")

    c1, c2 = col1.columns(2)

    with c1:
        st.image(st.session_state.last_output, caption="Final Detection", use_container_width=True)

    with c2:
        st.image(st.session_state.last_heatmap, caption="Final Heatmap", use_container_width=True)

    col2.metric("Final Count", st.session_state.last_count)

# ----------------------------
# DOWNLOAD REPORT
# ----------------------------
if st.session_state.last_output is not None and st.session_state.last_heatmap is not None:

    mode_name = "Dense Mode" if dense_mode else "Normal Mode"
    graph_data = st.session_state.dense_history if dense_mode else st.session_state.normal_history

    if mode == "Image":
        left, center, right = zone_analysis(
            st.session_state.last_output.shape[1],
            centers if 'centers' in locals() else []
        )
    else:
        left, center, right = zone_analysis(
            st.session_state.last_output.shape[1],
            list(tracker.objects.values()) if len(tracker.objects) > 0 else []
        )
    
    print("GRAPH DATA:", graph_data)

    pdf_data = generate_pdf_report(
        mode_name=mode_name,
        count=st.session_state.last_count,
        left=left,
        center=center,
        right=right,
        detection_img=st.session_state.last_output,
        heatmap_img=st.session_state.last_heatmap,
        graph_data=graph_data
    )

    st.download_button(
        label="📄 Download Report",
        data=pdf_data,
        file_name="Crowd_Report.pdf",
        mime="application/pdf"
    )

# ----------------------------
# ALERT TABLE
# ----------------------------
st.subheader("🚨 Crowd Alert Monitor")

if len(st.session_state.alert_history) > 0:
    alert_df = pd.DataFrame(st.session_state.alert_history)

    if mode == "Video":
        alert_df = alert_df.sort_values(by="Count", ascending=False).head(5)

    st.dataframe(alert_df, use_container_width=True)
else:
    st.info("No crowd alerts generated yet.")

# ----------------------------
# DASHBOARD
# ----------------------------
st.subheader("📊 Crowd Analytics Dashboard")

if dense_mode:
    show_graph(st.session_state.dense_history, "Dense Mode Trend", mode)
else:
    show_graph(st.session_state.normal_history, "Normal Mode Trend", mode)