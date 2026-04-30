# # # # # import streamlit as st
# # # # # import cv2
# # # # # import numpy as np
# # # # # import imutils
# # # # # import pandas as pd
# # # # # import tempfile
# # # # # from config import YOLO_CONFIG
# # # # # from datetime import datetime
# # # # # import matplotlib.pyplot as plt

# # # # # from reportlab.lib.pagesizes import A4
# # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # ----------------------------
# # # # # # PAGE CONFIG
# # # # # # ----------------------------
# # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # # st.title("🚶 Crowd Counting System")

# # # # # # ----------------------------
# # # # # # SESSION STATE
# # # # # # ----------------------------
# # # # # for key in ["history", "alerts", "run_video", "final_frame", "final_heatmap", "final_zones"]:
# # # # #     if key not in st.session_state:
# # # # #         st.session_state[key] = [] if key in ["history", "alerts"] else None

# # # # # # ----------------------------
# # # # # # LOAD MODEL
# # # # # # ----------------------------
# # # # # @st.cache_resource
# # # # # def load_model():
# # # # #     net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG["CONFIG_PATH"], YOLO_CONFIG["WEIGHTS_PATH"])
# # # # #     ln = net.getLayerNames()
# # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # #     return net, ln

# # # # # net, ln = load_model()

# # # # # # ----------------------------
# # # # # # SIDEBAR
# # # # # # ----------------------------
# # # # # st.sidebar.title("📊 Dashboard")
# # # # # mode = st.sidebar.radio("Select Mode", ["Image", "Video"])

# # # # # st.sidebar.markdown("---")
# # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # st.sidebar.markdown("---")
# # # # # st.sidebar.info("YOLO-based Crowd Counting")

# # # # # # ----------------------------
# # # # # # FILE UPLOAD
# # # # # # ----------------------------
# # # # # uploaded = st.file_uploader("📂 Upload Image / Video", type=["jpg", "png", "mp4"])

# # # # # # ----------------------------
# # # # # # FUNCTIONS
# # # # # # ----------------------------
# # # # # def get_crowd_alert(count):
# # # # #     t = datetime.now().strftime("%H:%M:%S")
# # # # #     if count <= 20: s, m = "Safe", "Under control"
# # # # #     elif count <= 40: s, m = "Crowded", "Moderate"
# # # # #     elif count <= 60: s, m = "High", "High density"
# # # # #     else: s, m = "Danger", "Critical"
# # # # #     return {"Time": t, "Count": count, "Status": s, "Message": m}

# # # # # def detect_people(image, size, conf, nms, video=False):
# # # # #     image = imutils.resize(image, width=800)
# # # # #     H, W = image.shape[:2]

# # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (size, size), swapRB=True)
# # # # #     net.setInput(blob)
# # # # #     outputs = net.forward(ln)

# # # # #     boxes, centers, confs = [], [], []

# # # # #     for output in outputs:
# # # # #         for det in output:
# # # # #             scores = det[5:]
# # # # #             cid = np.argmax(scores)
# # # # #             confidence = scores[cid]

# # # # #             if cid == 0 and confidence > conf:
# # # # #                 box = det[:4] * np.array([W, H, W, H])
# # # # #                 cx, cy, w, h = box.astype("int")

# # # # #                 boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
# # # # #                 centers.append((cx, cy))
# # # # #                 confs.append(float(confidence))

# # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confs, conf, nms)

# # # # #     final = []
# # # # #     if len(idxs) > 0:
# # # # #         for i in idxs.flatten():
# # # # #             final.append(centers[i])
# # # # #             cx, cy = centers[i]

# # # # #             if video:
# # # # #                 cv2.circle(image, (cx, cy), 4, (0,0,255), -1)
# # # # #             else:
# # # # #                 x,y,w,h = boxes[i]
# # # # #                 cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

# # # # #     return image, final

# # # # # # FIXED HEATMAP
# # # # # def generate_heatmap(image, centers):
# # # # #     h, w = image.shape[:2]
# # # # #     heat = np.zeros((h, w), dtype=np.float32)

# # # # #     for x, y in centers:
# # # # #         if 0 <= int(x) < w and 0 <= int(y) < h:
# # # # #             heat[int(y), int(x)] += 1

# # # # #     heat = cv2.GaussianBlur(heat, (71, 71), 0)
# # # # #     heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
# # # # #     return cv2.applyColorMap(np.uint8(heat), cv2.COLORMAP_JET)

# # # # # def zone_analysis(w, centers):
# # # # #     l=c=r=0
# # # # #     for x,_ in centers:
# # # # #         if x < w/3: l+=1
# # # # #         elif x < 2*w/3: c+=1
# # # # #         else: r+=1
# # # # #     return l,c,r

# # # # # # FIXED PDF FUNCTION
# # # # # def generate_pdf(img, heat, hist, count, zones, alerts):
# # # # #     cv2.imwrite("det.jpg", img)
# # # # #     cv2.imwrite("heat.jpg", heat)

# # # # #     plt.figure(figsize=(6,4))
# # # # #     if len(hist) == 1:
# # # # #         plt.plot([0, hist[0]], marker='o')
# # # # #         plt.xticks([0,1], ["Start","Current"])
# # # # #     else:
# # # # #         plt.plot(hist)

# # # # #     plt.title("Crowd Trend")
# # # # #     plt.xlabel("Time")
# # # # #     plt.ylabel("Count")
# # # # #     plt.grid(True)

# # # # #     plt.savefig("graph.png")
# # # # #     plt.close()

# # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # #     styles = getSampleStyleSheet()
# # # # #     story = []

# # # # #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# # # # #     story.append(Spacer(1,10))
# # # # #     story.append(Paragraph(f"<b>Final Count:</b> {count}", styles["Normal"]))
# # # # #     story.append(Spacer(1,10))

# # # # #     story.append(RLImage("det.jpg", width=400, height=250))
# # # # #     story.append(Spacer(1,10))
# # # # #     story.append(RLImage("heat.jpg", width=400, height=250))
# # # # #     story.append(Spacer(1,10))
# # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # #     pdf.build(story)

# # # # #     with open("report.pdf","rb") as f:
# # # # #         return f.read()

# # # # # # ----------------------------
# # # # # # IMAGE MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Image":
# # # # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # # # #     if st.button("🚀 Process Image"):
# # # # #         out, centers = detect_people(img,640,conf,nms)
# # # # #         count = len(centers)

# # # # #         heat = generate_heatmap(img.copy(), centers)
# # # # #         zones = zone_analysis(img.shape[1], centers)
# # # # #         alert = get_crowd_alert(count)

# # # # #         t1,t2 = st.columns(2)
# # # # #         t1.image(out, caption=f"Count: {count}")
# # # # #         t2.image(heat)

# # # # #         m1,m2 = st.columns(2)
# # # # #         m1.line_chart([0,count])
# # # # #         m2.bar_chart(pd.DataFrame({
# # # # #             "Zone":["Left","Center","Right"],
# # # # #             "Count":zones
# # # # #         }).set_index("Zone"))

# # # # #         st.subheader("⚠️ Alert")
# # # # #         st.table(pd.DataFrame([alert]))

# # # # #         pdf = generate_pdf(out, heat, [count], count, zones, [alert])
# # # # #         st.download_button("📄 Download PDF", pdf, "image_report.pdf")

# # # # # # ----------------------------
# # # # # # VIDEO MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Video":
# # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # #     tfile.write(uploaded.read())
# # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # #     c1,c2 = st.columns(2)
# # # # #     if c1.button("▶ Start"):
# # # # #         st.session_state.run_video=True
# # # # #         st.session_state.history=[]
# # # # #         st.session_state.alerts=[]

# # # # #     if c2.button("⛔ Stop"):
# # # # #         st.session_state.run_video=False

# # # # #     top1, top2 = st.columns(2)
# # # # #     mid1, mid2 = st.columns(2)

# # # # #     frame_box = top1.empty()
# # # # #     heat_box = top2.empty()
# # # # #     trend_box = mid1.empty()
# # # # #     zone_box = mid2.empty()
# # # # #     alert_box = st.empty()

# # # # #     while cap.isOpened() and st.session_state.run_video:
# # # # #         ret, frame = cap.read()
# # # # #         if not ret: break

# # # # #         out, centers = detect_people(frame,640,conf,nms,True)
# # # # #         count=len(centers)

# # # # #         heat = generate_heatmap(frame.copy(), centers)
# # # # #         zones = zone_analysis(frame.shape[1], centers)
# # # # #         alert = get_crowd_alert(count)

# # # # #         st.session_state.history.append(count)
# # # # #         st.session_state.alerts.append(alert)
# # # # #         st.session_state.final_frame=out
# # # # #         st.session_state.final_heatmap=heat
# # # # #         st.session_state.final_zones=zones

# # # # #         frame_box.image(out, caption=f"Count: {count}")
# # # # #         heat_box.image(heat)

# # # # #         trend_box.line_chart(st.session_state.history)
# # # # #         zone_box.bar_chart(pd.DataFrame({
# # # # #             "Zone":["Left","Center","Right"],
# # # # #             "Count":zones
# # # # #         }).set_index("Zone"))

# # # # #         alert_box.table(pd.DataFrame(st.session_state.alerts).tail(5))

# # # # #     cap.release()

# # # # #     # AFTER STOP
# # # # #     if not st.session_state.run_video and st.session_state.final_frame is not None:

# # # # #         t1,t2 = st.columns(2)
# # # # #         t1.image(st.session_state.final_frame)
# # # # #         t2.image(st.session_state.final_heatmap)

# # # # #         m1,m2 = st.columns(2)
# # # # #         m1.line_chart(st.session_state.history)
# # # # #         m2.bar_chart(pd.DataFrame({
# # # # #             "Zone":["Left","Center","Right"],
# # # # #             "Count":st.session_state.final_zones
# # # # #         }).set_index("Zone"))

# # # # #         st.subheader("⚠️ Alerts")
# # # # #         st.table(pd.DataFrame(st.session_state.alerts).tail(5))

# # # # #         pdf = generate_pdf(
# # # # #             st.session_state.final_frame,
# # # # #             st.session_state.final_heatmap,
# # # # #             st.session_state.history,
# # # # #             st.session_state.history[-1],
# # # # #             st.session_state.final_zones,
# # # # #             st.session_state.alerts
# # # # #         )

# # # # #         st.download_button("📄 Download Final Report", pdf, "video_report.pdf")



# # # # ----------------------------------------------------------------------------------------






import streamlit as st
import cv2
import numpy as np
import imutils
import pandas as pd
import tempfile
from config import YOLO_CONFIG
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Crowd Counting", layout="wide")

# ----------------------------
# UI STYLE (ONLY UI)
# ----------------------------
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:32px;
    font-weight:bold;
    color:#00C2FF;
    margin-bottom:10px;
}

.card {
    padding:15px;
    border-radius:12px;
    background-color:#111827;
    box-shadow:0px 2px 10px rgba(0,0,0,0.3);
}

.metric-box {
    padding:15px;
    border-radius:10px;
    background:#1f2937;
    text-align:center;
}

.small-text {
    font-size:14px;
    color:#9ca3af;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🚶 Crowd Counting Dashboard</div>", unsafe_allow_html=True)

# ----------------------------
# SESSION STATE
# ----------------------------
for key in ["history", "alerts", "run_video", "final_frame", "final_heatmap", "final_zones"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["history", "alerts"] else None

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG["CONFIG_PATH"], YOLO_CONFIG["WEIGHTS_PATH"])
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

net, ln = load_model()

# ----------------------------
# SIDEBAR UI
# ----------------------------
with st.sidebar:
    st.header("📊 Control Panel")

    mode = st.radio("Select Mode", ["Image", "Video"])

    st.markdown("---")

    uploaded = None
    if mode == "Image":
        uploaded = st.file_uploader("📷 Upload Image", type=["jpg", "png"])
    elif mode == "Video":
        uploaded = st.file_uploader("🎥 Upload Video", type=["mp4"])

    st.markdown("---")

    conf = st.slider("Confidence", 0.1, 1.0, 0.4)
    nms = st.slider("NMS", 0.1, 1.0, 0.4)

    st.markdown("---")
    st.info("💡 AI Crowd Monitoring System")

# ----------------------------
# FUNCTIONS (UNCHANGED)
# ----------------------------
def get_crowd_alert(count):
    t = datetime.now().strftime("%H:%M:%S")
    if count <= 20: s, m = "Safe", "Under control"
    elif count <= 40: s, m = "Crowded", "Moderate"
    elif count <= 60: s, m = "High", "High density"
    else: s, m = "Danger", "Critical"
    return {"Time": t, "Count": count, "Status": s, "Message": m}

def detect_people(image, size, conf, nms, video=False):
    image = imutils.resize(image, width=800)
    H, W = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (size, size), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes, centers, confs = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            cid = np.argmax(scores)
            confidence = scores[cid]

            if cid == 0 and confidence > conf:
                box = det[:4] * np.array([W, H, W, H])
                cx, cy, w, h = box.astype("int")

                boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
                centers.append((cx, cy))
                confs.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf, nms)

    final = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final.append(centers[i])
            cx, cy = centers[i]

            if video:
                cv2.circle(image, (cx, cy), 4, (0,0,255), -1)
            else:
                x,y,w,h = boxes[i]
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return image, final

def generate_heatmap(image, centers):
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for x, y in centers:
        if 0 <= int(x) < w and 0 <= int(y) < h:
            heat[int(y), int(x)] += 1

    heat = cv2.GaussianBlur(heat, (71, 71), 0)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(heat), cv2.COLORMAP_JET)

def zone_analysis(w, centers):
    l=c=r=0
    for x,_ in centers:
        if x < w/3: l+=1
        elif x < 2*w/3: c+=1
        else: r+=1
    return l,c,r

# ----------------------------
# PDF GENERATION FUNCTION
# ----------------------------

def generate_pdf(image, heatmap, counts, total, zones, alerts, history):

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------------- TITLE ----------------
    elements.append(Paragraph("Crowd Monitoring Report", styles["Title"]))
    elements.append(Spacer(1, 10))

    # ---------------- SAVE IMAGES ----------------
    cv2.imwrite("temp_img.jpg", image)
    cv2.imwrite("temp_heat.jpg", heatmap)

    # ---------------- CROWD TREND (NEW PART) ----------------
    plt.figure()
    plt.plot(history, marker="o")
    plt.title("Crowd Trend")
    plt.xlabel("Frame / Time")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("trend.jpg")
    plt.close()

    # ---------------- IMAGES ----------------
    elements.append(Paragraph("Detection Output", styles["Heading2"]))
    elements.append(RLImage("temp_img.jpg", width=400, height=250))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Heatmap", styles["Heading2"]))
    elements.append(RLImage("temp_heat.jpg", width=400, height=250))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Crowd Trend", styles["Heading2"]))
    elements.append(RLImage("trend.jpg", width=400, height=250))
    elements.append(Spacer(1, 10))

    # ---------------- SUMMARY ----------------
    elements.append(Paragraph(f"Total Count: {total}", styles["Normal"]))
    elements.append(Paragraph(
        f"Zones: Left={zones[0]} Center={zones[1]} Right={zones[2]}",
        styles["Normal"]
    ))
    elements.append(Paragraph(f"Alerts Triggered: {len(alerts)}", styles["Normal"]))

    doc.build(elements)

    with open(pdf_path, "rb") as f:
        return f.read()


# ----------------------------
# IMAGE MODE (FINAL STABLE VERSION)
# ----------------------------
if mode == "Image" and uploaded:

    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

    if st.button("🚀 Process Image"):

        # ---------------- DETECTION ----------------
        out, centers = detect_people(img, 640, conf, nms)
        count = len(centers)

        heat = generate_heatmap(img.copy(), centers)
        zones = zone_analysis(img.shape[1], centers)
        alert = get_crowd_alert(count)

        # ---------------- HISTORY ----------------
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append(count)
        history = st.session_state.history

        # ---------------- STATS ----------------
        total_today = sum(history)
        avg_count = int(total_today / len(history))
        max_count = max(history)
        alerts_triggered = sum(1 for c in history if c > 50)

        status_color = (
            "#22c55e" if alert["Status"] == "Safe"
            else "#facc15" if alert["Status"] == "Crowded"
            else "#ef4444"
        )

        # ---------------- RESIZE (IMPORTANT FIX) ----------------
        out_small = cv2.resize(out, (850, 460))
        heat_small = cv2.resize(heat, (550, 260))

        # ======================================================
        # ROW 1: IMAGE + CARDS
        # ======================================================
        st.markdown("### 📡 CROWD MONITORING DASHBOARD")

        img_col, card_col = st.columns([3.2, 1])

        with img_col:
            st.image(out_small, use_container_width=True)

        with card_col:
            st.markdown(f"""
                <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;">
                    <p style="color:#9ca3af;margin:0;">CROWD COUNT</p>
                    <h2 style="color:#22c55e;margin:0;">{count}</h2>
                </div>

                <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;border-left:5px solid {status_color};">
                    <p style="color:#9ca3af;margin:0;">STATUS</p>
                    <h3 style="color:{status_color};margin:0;">{alert['Status']}</h3>
                </div>

                <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;">
                    <p style="color:#9ca3af;margin:0;">ALERT LIMIT</p>
                    <h3 style="color:#facc15;margin:0;">50</h3>
                </div>
            """, unsafe_allow_html=True)

        # ======================================================
        # ROW 2: HEATMAP + SUMMARY
        # ======================================================
        st.markdown("---")
        heat_col, sum_col = st.columns([2.2, 1])

        with heat_col:
            st.markdown("### 🔥 HEATMAP")
            st.image(heat_small, use_container_width=True)

        with sum_col:
            st.markdown("### 📊 SUMMARY")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            def card(title, value, color):
                return f"""
                <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
                    <p style="color:#9ca3af;margin:0;font-size:12px;">{title}</p>
                    <h3 style="color:{color};margin:0;">{value}</h3>
                </div>
                """

            with c1:
                st.markdown(card("TOTAL", total_today, "#22c55e"), unsafe_allow_html=True)

            with c2:
                st.markdown(card("AVG", avg_count, "#3b82f6"), unsafe_allow_html=True)

            with c3:
                st.markdown(card("MAX", max_count, "#facc15"), unsafe_allow_html=True)

                latest_alert = alert["Status"]

            with c4:
                st.markdown(card("ALERTS", f"{alerts_triggered} ({latest_alert})", "#ef4444"), unsafe_allow_html=True)

            # with c4:
            #     st.markdown(card("ALERTS", alerts_triggered, "#ef4444"), unsafe_allow_html=True)

        # ======================================================
        # ROW 3: CHARTS
        # ======================================================
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 CROWD TREND")
            st.line_chart(history)

        with col2:
            st.markdown("### 📍 ZONE ANALYSIS")
            st.bar_chart({
                "Left": zones[0],
                "Center": zones[1],
                "Right": zones[2]
            })

        # ======================================================
        # PDF GENERATION (FIXED + TREND INCLUDED)
        # ======================================================
        pdf = generate_pdf(
        out_small,
        heat_small,
        history,                 # full history
        sum(history),           # TOTAL COUNT FIX
        zones,
        st.session_state.alerts if mode == "Video" else [alert],
        history
    )

        st.download_button(
            "⬇️ Download Full Report (PDF)",
            pdf,
            file_name="crowd_report.pdf",
            use_container_width=True
        )





# # ----------------------------
# # VIDEO MODE (PERSIST UI AFTER STOP)
# # ----------------------------
# if mode == "Video" and uploaded:

#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded.read())
#     cap = cv2.VideoCapture(tfile.name)

#     col1, col2 = st.columns(2)

#     if col1.button("▶ Start"):
#         st.session_state.run_video = True
#         st.session_state.history = []
#         st.session_state.alerts = []
#         st.session_state.final_frame = None
#         st.session_state.final_heatmap = None
#         st.session_state.final_zones = (0, 0, 0)

#     if col2.button("⛔ Stop"):
#         st.session_state.run_video = False

#     # ---------------- UI PLACEHOLDERS ----------------
#     img_col, card_col = st.columns([3.2, 1])
#     heat_col, sum_col = st.columns([2.2, 1])
#     col1, col2 = st.columns(2)

#     frame_box = img_col.empty()
#     heat_box = heat_col.empty()
#     card_box = card_col.empty()
#     summary_box = sum_col.empty()
#     trend_box = col1.empty()
#     zone_box = col2.empty()
#     alert_table = st.empty()

#     # ---------------- LIVE LOOP ----------------
#     while cap.isOpened() and st.session_state.run_video:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         out, centers = detect_people(frame, 640, conf, nms, True)
#         count = len(centers)

#         heat = generate_heatmap(frame.copy(), centers)
#         zones = zone_analysis(frame.shape[1], centers)
#         alert = get_crowd_alert(count)

#         st.session_state.history.append(count)
#         st.session_state.alerts.append(alert)

#         # 🔥 SAVE FINAL STATE EVERY FRAME
#         st.session_state.final_frame = out
#         st.session_state.final_heatmap = heat
#         st.session_state.final_zones = zones
#         st.session_state.final_count = count

#         # ---------------- LIVE UI ----------------
#         frame_box.image(out, caption=f"Count: {count}")
#         heat_box.image(heat)

#         trend_box.line_chart(st.session_state.history)

#         zone_box.bar_chart({
#             "Left": zones[0],
#             "Center": zones[1],
#             "Right": zones[2]
#         })

#         alert_table.table(pd.DataFrame(st.session_state.alerts).tail(5))

#     cap.release()

# # ======================================================
# # 🔥 FINAL UI (ALWAYS SHOWN AFTER STOP OR END)
# # ======================================================
# if "final_frame" in st.session_state and st.session_state.final_frame is not None:

#     final_count = st.session_state.get("final_count", 0)
#     history = st.session_state.history

#     total_today = sum(history)
#     avg_count = int(total_today / len(history)) if history else 0
#     max_count = max(history) if history else 0
#     alerts_triggered = sum(1 for a in st.session_state.alerts if a["Status"] != "Safe")

#     status_color = (
#         "#22c55e" if final_count <= 20
#         else "#facc15" if final_count <= 40
#         else "#ef4444"
#     )

#     st.markdown("### 📡 FINAL CROWD DASHBOARD")

#     img_col, card_col = st.columns([3.2, 1])
#     heat_col, sum_col = st.columns([2.2, 1])
#     col1, col2 = st.columns(2)

#     with img_col:
#         st.image(st.session_state.final_frame, use_container_width=True)

#     with heat_col:
#         st.image(st.session_state.final_heatmap, use_container_width=True)

#     with card_col:
#         st.markdown(f"""
#         <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;">
#             <p style="color:#9ca3af;margin:0;">CROWD COUNT</p>
#             <h2 style="color:#22c55e;margin:0;">{final_count}</h2>
#         </div>

#         <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;border-left:5px solid {status_color};">
#             <p style="color:#9ca3af;margin:0;">STATUS</p>
#             <h3 style="color:{status_color};margin:0;">{st.session_state.alerts[-1]['Status']}</h3>
#         </div>

#         <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;">
#             <p style="color:#9ca3af;margin:0;">ALERT LIMIT</p>
#             <h3 style="color:#facc15;margin:0;">50</h3>
#         </div>
#         """, unsafe_allow_html=True)

#     with sum_col:
#         st.markdown(f"""
#         <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
#             <p style="color:#9ca3af;margin:0;font-size:12px;">TOTAL</p>
#             <h3 style="color:#22c55e;margin:0;">{total_today}</h3>
#         </div>

#         <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
#             <p style="color:#9ca3af;margin:0;font-size:12px;">AVG</p>
#             <h3 style="color:#3b82f6;margin:0;">{avg_count}</h3>
#         </div>

#         <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
#             <p style="color:#9ca3af;margin:0;font-size:12px;">MAX</p>
#             <h3 style="color:#facc15;margin:0;">{max_count}</h3>
#         </div>

#         <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;">
#             <p style="color:#9ca3af;margin:0;font-size:12px;">ALERTS</p>
#             <h3 style="color:#ef4444;margin:0;">{alerts_triggered}</h3>
#         </div>
#         """, unsafe_allow_html=True)

#     with col1:
#         st.line_chart(history)

#     with col2:
#         st.bar_chart({
#             "Left": st.session_state.final_zones[0],
#             "Center": st.session_state.final_zones[1],
#             "Right": st.session_state.final_zones[2]
#         })







def generate_video_pdf(frame, heatmap, history, zones, alerts, total_count):

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------------- TITLE ----------------
    elements.append(Paragraph("Crowd Monitoring Video Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # ---------------- SAVE IMAGES ----------------
    cv2.imwrite("video_frame.jpg", frame)
    cv2.imwrite("video_heat.jpg", heatmap)

    # ---------------- TREND GRAPH ----------------
    plt.figure()
    plt.plot(history, marker="o")
    plt.title("Crowd Trend Over Video")
    plt.xlabel("Frame")
    plt.ylabel("People Count")
    plt.tight_layout()
    plt.savefig("video_trend.jpg")
    plt.close()

    # ---------------- IMAGES ----------------
    elements.append(Paragraph("Final Crowd Frame", styles["Heading2"]))
    elements.append(RLImage("video_frame.jpg", width=400, height=240))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Heatmap Analysis", styles["Heading2"]))
    elements.append(RLImage("video_heat.jpg", width=400, height=240))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Crowd Trend Graph", styles["Heading2"]))
    elements.append(RLImage("video_trend.jpg", width=400, height=240))
    elements.append(Spacer(1, 12))

    # ---------------- SUMMARY ----------------
    elements.append(Paragraph("Crowd Summary", styles["Heading2"]))
    elements.append(Paragraph(f"Total People Detected: {total_count}", styles["Normal"]))
    elements.append(Spacer(1, 8))

    # ---------------- ZONE ANALYSIS ----------------
    elements.append(Paragraph("Zone Analysis", styles["Heading2"]))
    elements.append(Paragraph(f"Left Zone: {zones[0]}", styles["Normal"]))
    elements.append(Paragraph(f"Center Zone: {zones[1]}", styles["Normal"]))
    elements.append(Paragraph(f"Right Zone: {zones[2]}", styles["Normal"]))
    elements.append(Spacer(1, 8))

    # ---------------- ALERTS ----------------
    alert_count = sum(1 for a in alerts if a["Status"] != "Safe")

    elements.append(Paragraph("Alerts Summary", styles["Heading2"]))
    elements.append(Paragraph(f"Total Alerts Triggered: {alert_count}", styles["Normal"]))

    doc.build(elements)

    with open(pdf_path, "rb") as f:
        return f.read()



# ----------------------------
# VIDEO MODE (LIVE + FINAL UI STABLE)
# ----------------------------
if mode == "Video" and uploaded:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    cap = cv2.VideoCapture(tfile.name)

    col1, col2 = st.columns(2)

    if col1.button("▶ Start"):
        st.session_state.run_video = True
        st.session_state.history = []
        st.session_state.alerts = []
        st.session_state.final_frame = None
        st.session_state.final_heatmap = None
        st.session_state.final_zones = (0, 0, 0)
        st.session_state.final_count = 0

    if col2.button("⛔ Stop"):
        st.session_state.run_video = False

    # ---------------- UI PLACEHOLDERS (SAME UI AS IMAGE MODE) ----------------
    st.markdown("### 📡 CROWD MONITORING DASHBOARD")

    img_col, card_col = st.columns([3.2, 1])
    heat_col, sum_col = st.columns([2.2, 1])
    col1, col2 = st.columns(2)

    frame_box = img_col.empty()
    heat_box = heat_col.empty()
    card_box = card_col.empty()
    summary_box = sum_col.empty()
    trend_box = col1.empty()
    zone_box = col2.empty()
    alert_box = st.empty()

    # ---------------- LIVE VIDEO LOOP ----------------
    while cap.isOpened() and st.session_state.run_video:

        ret, frame = cap.read()
        if not ret:
            break

        out, centers = detect_people(frame, 640, conf, nms, True)
        count = len(centers)

        heat = generate_heatmap(frame.copy(), centers)
        zones = zone_analysis(frame.shape[1], centers)
        alert = get_crowd_alert(count)

        # history update
        st.session_state.history.append(count)
        st.session_state.alerts.append(alert)

        # save latest state
        st.session_state.final_frame = out
        st.session_state.final_heatmap = heat
        st.session_state.final_zones = zones
        st.session_state.final_count = count

        # ---------------- LIVE CARDS (UPDATED EVERY FRAME) ----------------
        status_color = (
            "#22c55e" if count <= 20
            else "#facc15" if count <= 40
            else "#ef4444"
        )

        card_box.markdown(f"""
        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;">
            <p style="color:#9ca3af;margin:0;">CROWD COUNT</p>
            <h2 style="color:#22c55e;margin:0;">{count}</h2>
        </div>

        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;border-left:5px solid {status_color};">
            <p style="color:#9ca3af;margin:0;">STATUS</p>
            <h3 style="color:{status_color};margin:0;">{alert['Status']}</h3>
        </div>

        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;">
            <p style="color:#9ca3af;margin:0;">ALERT LIMIT</p>
            <h3 style="color:#facc15;margin:0;">50</h3>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- LIVE IMAGES ----------------
        frame_box.image(out, use_container_width=True)
        heat_box.image(heat, use_container_width=True)

        # ---------------- LIVE STATS ----------------
        trend_box.line_chart(st.session_state.history)

        zone_box.bar_chart({
            "Left": zones[0],
            "Center": zones[1],
            "Right": zones[2]
        })

        alert_box.table(pd.DataFrame(st.session_state.alerts).tail(5))

    cap.release()

# ======================================================
# 🔥 FINAL UI (AFTER STOP OR END)
# ======================================================
if st.session_state.final_frame is not None:

    history = st.session_state.history
    alerts = st.session_state.alerts
    zones = st.session_state.final_zones
    final_count = st.session_state.final_count

    total = sum(history)
    avg = int(total / len(history)) if history else 0
    maxv = max(history) if history else 0

    alerts_triggered = sum(1 for a in alerts if a["Status"] != "Safe")

    status_color = (
        "#22c55e" if final_count <= 20
        else "#facc15" if final_count <= 40
        else "#ef4444"
    )

    st.markdown("### 📌 FINAL REPORT (STOPPED STATE)")

    img_col, card_col = st.columns([3.2, 1])
    heat_col, sum_col = st.columns([2.2, 1])
    col1, col2 = st.columns(2)

    with img_col:
        st.image(st.session_state.final_frame, use_container_width=True)

    with heat_col:
        st.image(st.session_state.final_heatmap, use_container_width=True)

    with card_col:
        st.markdown(f"""
        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;">
            <p style="color:#9ca3af;">CROWD COUNT</p>
            <h2 style="color:#22c55e;">{final_count}</h2>
        </div>

        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;margin-bottom:12px;border-left:5px solid {status_color};">
            <p style="color:#9ca3af;">STATUS</p>
            <h3 style="color:{status_color};">{alerts[-1]['Status'] if alerts else "Safe"}</h3>
        </div>

        <div style="background:#111827;padding:18px;border-radius:12px;text-align:center;">
            <p style="color:#9ca3af;">ALERT LIMIT</p>
            <h3 style="color:#facc15;">50</h3>
        </div>
        """, unsafe_allow_html=True)

    with sum_col:
        st.markdown(f"""
        <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
            <p style="color:#9ca3af;">TOTAL</p>
            <h3 style="color:#22c55e;">{total}</h3>
        </div>

        <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
            <p style="color:#9ca3af;">AVG</p>
            <h3 style="color:#3b82f6;">{avg}</h3>
        </div>

        <div style="background:#1e293b;padding:14px;border-radius:10px;text-align:center;margin-bottom:10px;">
            <p style="color:#9ca3af;">MAX</p>
            <h3 style="color:#facc15;">{maxv}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.line_chart(history)

    with col2:
        st.bar_chart({
            "Left": zones[0],
            "Center": zones[1],
            "Right": zones[2]
        })
        
pdf = generate_video_pdf(
    st.session_state.final_frame,
    st.session_state.final_heatmap,
    st.session_state.history,
    st.session_state.final_zones,
    st.session_state.alerts,
    st.session_state.final_count
)

st.download_button(
    "📄 Download Video Report",
    pdf,
    file_name="crowd_video_report.pdf"
)