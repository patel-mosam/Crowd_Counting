# # # # # # # # # import streamlit as st
# # # # # # # # # import cv2
# # # # # # # # # import numpy as np
# # # # # # # # # import imutils
# # # # # # # # # import pandas as pd
# # # # # # # # # import tempfile
# # # # # # # # # from config import YOLO_CONFIG
# # # # # # # # # from collections import OrderedDict
# # # # # # # # # import math
# # # # # # # # # from datetime import datetime
# # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # from reportlab.lib.pagesizes import A4
# # # # # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # # # # ----------------------------
# # # # # # # # # # PAGE CONFIG
# # # # # # # # # # ----------------------------
# # # # # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")

# # # # # # # # # st.markdown("""
# # # # # # # # # <style>
# # # # # # # # # body { background-color: #0e1117; color: white; }
# # # # # # # # # </style>
# # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # ----------------------------
# # # # # # # # # # SESSION STATE
# # # # # # # # # # ----------------------------
# # # # # # # # # defaults = {
# # # # # # # # #     "run_video": False,
# # # # # # # # #     "last_output": None,
# # # # # # # # #     "last_heatmap": None,
# # # # # # # # #     "last_count": 0,
# # # # # # # # #     "dense_history": [],
# # # # # # # # #     "normal_history": [],
# # # # # # # # #     "all_centers": [],
# # # # # # # # #     "alert_history": []
# # # # # # # # # }

# # # # # # # # # for key, value in defaults.items():
# # # # # # # # #     if key not in st.session_state:
# # # # # # # # #         st.session_state[key] = value

# # # # # # # # # # ----------------------------
# # # # # # # # # # LOAD MODEL
# # # # # # # # # # ----------------------------
# # # # # # # # # @st.cache_resource
# # # # # # # # # def load_model():
# # # # # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # # # # #     )
# # # # # # # # #     ln = net.getLayerNames()
# # # # # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # # # # #     return net, ln

# # # # # # # # # net, ln = load_model()

# # # # # # # # # # ----------------------------
# # # # # # # # # # TRACKER
# # # # # # # # # # ----------------------------
# # # # # # # # # class CentroidTracker:
# # # # # # # # #     def __init__(self, max_distance=50):
# # # # # # # # #         self.next_id = 0
# # # # # # # # #         self.objects = OrderedDict()
# # # # # # # # #         self.max_distance = max_distance

# # # # # # # # #     def update(self, detections):
# # # # # # # # #         if len(detections) == 0:
# # # # # # # # #             return self.objects

# # # # # # # # #         input_centroids = np.array(detections)

# # # # # # # # #         if len(self.objects) == 0:
# # # # # # # # #             for c in input_centroids:
# # # # # # # # #                 self.objects[self.next_id] = c
# # # # # # # # #                 self.next_id += 1
# # # # # # # # #         else:
# # # # # # # # #             object_ids = list(self.objects.keys())
# # # # # # # # #             object_centroids = list(self.objects.values())
# # # # # # # # #             updated_objects = OrderedDict()

# # # # # # # # #             for new_c in input_centroids:
# # # # # # # # #                 min_dist = float("inf")
# # # # # # # # #                 match_id = None

# # # # # # # # #                 for obj_id, old_c in zip(object_ids, object_centroids):
# # # # # # # # #                     dist = math.dist(new_c, old_c)
# # # # # # # # #                     if dist < min_dist and dist < self.max_distance:
# # # # # # # # #                         min_dist = dist
# # # # # # # # #                         match_id = obj_id

# # # # # # # # #                 if match_id is not None and match_id not in updated_objects:
# # # # # # # # #                     updated_objects[match_id] = new_c
# # # # # # # # #                 else:
# # # # # # # # #                     updated_objects[self.next_id] = new_c
# # # # # # # # #                     self.next_id += 1

# # # # # # # # #             self.objects = updated_objects

# # # # # # # # #         return self.objects

# # # # # # # # # tracker = CentroidTracker()

# # # # # # # # # # ----------------------------
# # # # # # # # # # REMOVE DUPLICATES
# # # # # # # # # # ----------------------------
# # # # # # # # # def remove_duplicate_centers(centers, min_dist=22):
# # # # # # # # #     filtered = []
# # # # # # # # #     for c in centers:
# # # # # # # # #         duplicate = False
# # # # # # # # #         for fc in filtered:
# # # # # # # # #             if math.dist(c, fc) < min_dist:
# # # # # # # # #                 duplicate = True
# # # # # # # # #                 break
# # # # # # # # #         if not duplicate:
# # # # # # # # #             filtered.append(c)
# # # # # # # # #     return filtered

# # # # # # # # # # ----------------------------
# # # # # # # # # # HEATMAP
# # # # # # # # # # ----------------------------
# # # # # # # # # def generate_heatmap(image, centers, dense_mode=False):
# # # # # # # # #     h, w = image.shape[:2]
# # # # # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # # # # #     for (x, y) in centers:
# # # # # # # # #         if 0 <= x < w and 0 <= y < h:
# # # # # # # # #             heatmap[y, x] += 3.0 if dense_mode else 1.8

# # # # # # # # #     blur_size = (101, 101) if dense_mode else (71, 71)
# # # # # # # # #     heatmap = cv2.GaussianBlur(heatmap, blur_size, 0)
# # # # # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # # # # #     heatmap = np.uint8(heatmap)
# # # # # # # # #     heatmap = 255 - heatmap
# # # # # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # # # # #     return heatmap

# # # # # # # # # # ----------------------------
# # # # # # # # # # ZONE ANALYSIS
# # # # # # # # # # ----------------------------
# # # # # # # # # def zone_analysis(width, centers):
# # # # # # # # #     left, center, right = 0, 0, 0

# # # # # # # # #     for (x, y) in centers:
# # # # # # # # #         if x < width / 3:
# # # # # # # # #             left += 1
# # # # # # # # #         elif x < 2 * width / 3:
# # # # # # # # #             center += 1
# # # # # # # # #         else:
# # # # # # # # #             right += 1

# # # # # # # # #     return left, center, right

# # # # # # # # # # ----------------------------
# # # # # # # # # # ALERT SYSTEM
# # # # # # # # # # ----------------------------
# # # # # # # # # def get_crowd_alert(count, mode="Video"):
# # # # # # # # #     current_time = datetime.now().strftime("%H:%M:%S")

# # # # # # # # #     if mode == "Image":
# # # # # # # # #         if count <= 20:
# # # # # # # # #             status = "Safe"
# # # # # # # # #             message = "Crowd level is under control."
# # # # # # # # #         elif count <= 40:
# # # # # # # # #             status = "Crowded"
# # # # # # # # #             message = "Crowded area detected."
# # # # # # # # #         else:
# # # # # # # # #             status = "Highly Crowded"
# # # # # # # # #             message = "High crowd density detected."
# # # # # # # # #     else:
# # # # # # # # #         if count <= 20:
# # # # # # # # #             status = "Safe"
# # # # # # # # #             message = "Crowd level is under control."
# # # # # # # # #         elif count <= 40:
# # # # # # # # #             status = "Moderate Crowd"
# # # # # # # # #             message = "Moderate crowd detected. Monitor area."
# # # # # # # # #         elif count <= 60:
# # # # # # # # #             status = "High Crowd Alert"
# # # # # # # # #             message = "High crowd density detected. Take precautions."
# # # # # # # # #         else:
# # # # # # # # #             status = "Overcrowded / Danger"
# # # # # # # # #             message = "Critical crowd level detected. Immediate action required."

# # # # # # # # #     return {
# # # # # # # # #         "Time": current_time,
# # # # # # # # #         "Count": count,
# # # # # # # # #         "Status": status,
# # # # # # # # #         "Message": message
# # # # # # # # #     }

# # # # # # # # # # ----------------------------
# # # # # # # # # # DETECTION
# # # # # # # # # # ----------------------------
# # # # # # # # # def detect_people(image, input_size, conf, nms, dense_mode, is_video=False):
# # # # # # # # #     image = imutils.resize(image, width=800)
# # # # # # # # #     (H, W) = image.shape[:2]

# # # # # # # # #     if dense_mode:
# # # # # # # # #         if is_video:
# # # # # # # # #             conf = max(0.08, conf - 0.20)
# # # # # # # # #             nms = max(0.15, nms - 0.15)
# # # # # # # # #             min_area = 140
# # # # # # # # #             dup_dist = 20
# # # # # # # # #         else:
# # # # # # # # #             conf = max(0.08, conf - 0.20)
# # # # # # # # #             nms = max(0.15, nms - 0.15)
# # # # # # # # #             min_area = 150
# # # # # # # # #             dup_dist = 20
# # # # # # # # #     else:
# # # # # # # # #         min_area = 500
# # # # # # # # #         dup_dist = 0

# # # # # # # # #     blob = cv2.dnn.blobFromImage(
# # # # # # # # #         image, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False
# # # # # # # # #     )

# # # # # # # # #     net.setInput(blob)
# # # # # # # # #     outputs = net.forward(ln)

# # # # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # # # #     for output in outputs:
# # # # # # # # #         for det in output:
# # # # # # # # #             scores = det[5:]
# # # # # # # # #             classID = np.argmax(scores)
# # # # # # # # #             confidence = scores[classID]

# # # # # # # # #             if classID == 0 and confidence > conf:
# # # # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # # # #                 x = int(cx - w / 2)
# # # # # # # # #                 y = int(cy - h / 2)
# # # # # # # # #                 area = w * h

# # # # # # # # #                 if area > min_area:
# # # # # # # # #                     boxes.append([x, y, int(w), int(h)])
# # # # # # # # #                     centers.append((cx, cy))
# # # # # # # # #                     confidences.append(float(confidence))

# # # # # # # # #     filtered_centers = []

# # # # # # # # #     if len(boxes) > 0:
# # # # # # # # #         idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # # # # #         if len(idxs) > 0:
# # # # # # # # #             for i in idxs.flatten():
# # # # # # # # #                 filtered_centers.append(centers[i])

# # # # # # # # #                 x, y, w, h = boxes[i]
# # # # # # # # #                 cx, cy = centers[i]

# # # # # # # # #                 if dense_mode:
# # # # # # # # #                     cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # # # # # #                 else:
# # # # # # # # #                     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # # # # # # # #     if dense_mode:
# # # # # # # # #         filtered_centers = remove_duplicate_centers(filtered_centers, min_dist=dup_dist)

# # # # # # # # #     return image, filtered_centers

# # # # # # # # # # ----------------------------
# # # # # # # # # # GRAPH
# # # # # # # # # # ----------------------------
# # # # # # # # # def show_graph(data, title, mode):
# # # # # # # # #     if len(data) == 0:
# # # # # # # # #         return

# # # # # # # # #     if mode == "Image" and len(data) == 1:
# # # # # # # # #         val = data[0]
# # # # # # # # #         data = [max(0, val - 2), max(0, val - 1), val]

# # # # # # # # #     df = pd.DataFrame({
# # # # # # # # #         "Frame": list(range(1, len(data) + 1)),
# # # # # # # # #         "Count": data
# # # # # # # # #     })

# # # # # # # # #     st.subheader(title)
# # # # # # # # #     st.line_chart(df.set_index("Frame"))

# # # # # # # # # # ----------------------------
# # # # # # # # # # PDF REPORT
# # # # # # # # # # ----------------------------
# # # # # # # # # def save_graph_image(data, title, filename="crowd_graph.png"):
# # # # # # # # #     if len(data) == 0:
# # # # # # # # #         return None

# # # # # # # # #     plt.figure(figsize=(8, 3))
# # # # # # # # #     plt.plot(data)
# # # # # # # # #     plt.title(title)
# # # # # # # # #     plt.xlabel("Frame")
# # # # # # # # #     plt.ylabel("People Count")
# # # # # # # # #     plt.tight_layout()
# # # # # # # # #     plt.savefig(filename)
# # # # # # # # #     plt.close()
# # # # # # # # #     return filename

# # # # # # # # # def generate_pdf_report(mode_name, count, left, center, right,
# # # # # # # # #                         detection_img, heatmap_img, graph_data):

# # # # # # # # #     detect_path = "detection_output.jpg"
# # # # # # # # #     heatmap_path = "heatmap_output.jpg"

# # # # # # # # #     cv2.imwrite(detect_path, detection_img)
# # # # # # # # #     cv2.imwrite(heatmap_path, heatmap_img)

# # # # # # # # #     graph_path = save_graph_image(graph_data, f"{mode_name} Crowd Trend")
# # # # # # # # #     alert_info = get_crowd_alert(count, "Video")

# # # # # # # # #     pdf_path = "Crowd_Report.pdf"

# # # # # # # # #     doc = SimpleDocTemplate(pdf_path, pagesize=A4)
# # # # # # # # #     styles = getSampleStyleSheet()
# # # # # # # # #     story = []

# # # # # # # # #     story.append(Paragraph("Crowd Counting", styles["Title"]))
# # # # # # # # #     story.append(Spacer(1, 12))
# # # # # # # # #     story.append(Paragraph(f"<b>Detection Mode:</b> {mode_name}", styles["Normal"]))
# # # # # # # # #     story.append(Paragraph(f"<b>Total People Count:</b> {count}", styles["Normal"]))
# # # # # # # # #     story.append(Paragraph(f"<b>Alert Status:</b> {alert_info['Status']}", styles["Normal"]))
# # # # # # # # #     story.append(Paragraph(f"<b>Alert Message:</b> {alert_info['Message']}", styles["Normal"]))
# # # # # # # # #     story.append(Paragraph(f"<b>Zone Analysis:</b> Left={left}, Center={center}, Right={right}", styles["Normal"]))
# # # # # # # # #     story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
# # # # # # # # #     story.append(Spacer(1, 15))

# # # # # # # # #     story.append(Paragraph("<b>Detection Output</b>", styles["Heading2"]))
# # # # # # # # #     story.append(RLImage(detect_path, width=400, height=250))
# # # # # # # # #     story.append(Spacer(1, 12))

# # # # # # # # #     story.append(Paragraph("<b>Heatmap Output</b>", styles["Heading2"]))
# # # # # # # # #     story.append(RLImage(heatmap_path, width=400, height=250))
# # # # # # # # #     story.append(Spacer(1, 12))

# # # # # # # # #     if graph_path:
# # # # # # # # #         story.append(Paragraph("<b>Crowd Trend Graph</b>", styles["Heading2"]))
# # # # # # # # #         story.append(RLImage(graph_path, width=400, height=200))

# # # # # # # # #     doc.build(story)

# # # # # # # # #     with open(pdf_path, "rb") as f:
# # # # # # # # #         pdf_bytes = f.read()

# # # # # # # # #     return pdf_bytes

# # # # # # # # # # ----------------------------
# # # # # # # # # # UI
# # # # # # # # # # ----------------------------
# # # # # # # # # st.title("Crowd Counting")

# # # # # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # # # # # dense_mode = st.sidebar.toggle("🧠 Dense Crowd Mode (Improved)", False)
# # # # # # # # # show_heatmap = st.sidebar.toggle("🔥 Heatmap", True)

# # # # # # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)
# # # # # # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.3)
# # # # # # # # # input_size = 832 if dense_mode else 640

# # # # # # # # # uploaded = st.file_uploader("Upload File", type=["jpg", "png", "jpeg", "mp4"])

# # # # # # # # # col1, col2 = st.columns([3, 1])

# # # # # # # # # # ----------------------------
# # # # # # # # # # IMAGE MODE
# # # # # # # # # # ----------------------------
# # # # # # # # # if uploaded and mode == "Image":
# # # # # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # # # # #     if st.button("🚀 Detect"):
# # # # # # # # #         output, centers = detect_people(image, input_size, conf, nms, dense_mode, is_video=False)

# # # # # # # # #         count = len(centers)
# # # # # # # # #         alert = get_crowd_alert(count, "Image")
# # # # # # # # #         st.session_state.alert_history.append(alert)

# # # # # # # # #         if dense_mode:
# # # # # # # # #             st.session_state.dense_history.append(count)
# # # # # # # # #         else:
# # # # # # # # #             st.session_state.normal_history.append(count)

# # # # # # # # #         heatmap = generate_heatmap(image.copy(), centers, dense_mode)

# # # # # # # # #         st.session_state.last_output = output
# # # # # # # # #         st.session_state.last_heatmap = heatmap
# # # # # # # # #         st.session_state.last_count = count

# # # # # # # # #         left, center, right = zone_analysis(output.shape[1], centers)

# # # # # # # # #         col2.metric("👥 Count", count)
# # # # # # # # #         col2.write(f"L: {left}   C: {center}   R: {right}")

# # # # # # # # #         c1, c2 = col1.columns(2)
# # # # # # # # #         with c1:
# # # # # # # # #             st.image(output, use_container_width=True)
# # # # # # # # #         with c2:
# # # # # # # # #             st.image(heatmap, use_container_width=True)

# # # # # # # # # # ----------------------------
# # # # # # # # # # VIDEO MODE
# # # # # # # # # # ----------------------------
# # # # # # # # # if uploaded and mode == "Video":
# # # # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # # # #     tfile.write(uploaded.read())

# # # # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # # # #     cA, cB = st.columns(2)

# # # # # # # # #     if cA.button("▶ Start"):
# # # # # # # # #         st.session_state.run_video = True
# # # # # # # # #         st.session_state.all_centers = []
# # # # # # # # #         st.session_state.alert_history = []

# # # # # # # # #     if cB.button("⛔ Stop"):
# # # # # # # # #         st.session_state.run_video = False

# # # # # # # # #     frame_box = col1.empty()
# # # # # # # # #     stat = col2.empty()

# # # # # # # # #     while cap.isOpened() and st.session_state.run_video:
# # # # # # # # #         ret, frame = cap.read()
# # # # # # # # #         if not ret:
# # # # # # # # #             break

# # # # # # # # #         output, centers = detect_people(frame, input_size, conf, nms, dense_mode, is_video=True)

# # # # # # # # #         tracked = tracker.update(centers)
# # # # # # # # #         count = len(tracked)

# # # # # # # # #         alert = get_crowd_alert(count, "Video")
# # # # # # # # #         st.session_state.alert_history.append(alert)

# # # # # # # # #         st.session_state.all_centers.extend(centers)

# # # # # # # # #         if dense_mode:
# # # # # # # # #             st.session_state.dense_history.append(count)
# # # # # # # # #         else:
# # # # # # # # #             st.session_state.normal_history.append(count)

# # # # # # # # #         heatmap = generate_heatmap(frame.copy(), st.session_state.all_centers, dense_mode)

# # # # # # # # #         st.session_state.last_output = output
# # # # # # # # #         st.session_state.last_heatmap = heatmap
# # # # # # # # #         st.session_state.last_count = count

# # # # # # # # #         frame_box.image(output, channels="BGR")
# # # # # # # # #         stat.metric("People", count)

# # # # # # # # #     cap.release()

# # # # # # # # # # ----------------------------
# # # # # # # # # # FINAL RESULT (VIDEO ONLY)
# # # # # # # # # # ----------------------------
# # # # # # # # # if mode == "Video" and not st.session_state.run_video and st.session_state.last_output is not None:
# # # # # # # # #     st.subheader("📌 Final Analysis Result")

# # # # # # # # #     c1, c2 = col1.columns(2)

# # # # # # # # #     with c1:
# # # # # # # # #         st.image(st.session_state.last_output, caption="Final Detection", use_container_width=True)

# # # # # # # # #     with c2:
# # # # # # # # #         st.image(st.session_state.last_heatmap, caption="Final Heatmap", use_container_width=True)

# # # # # # # # #     col2.metric("Final Count", st.session_state.last_count)

# # # # # # # # # # ----------------------------
# # # # # # # # # # ALERT TABLE
# # # # # # # # # # ----------------------------
# # # # # # # # # st.subheader("🚨 Crowd Alert Monitor")

# # # # # # # # # if len(st.session_state.alert_history) > 0:
# # # # # # # # #     alert_df = pd.DataFrame(st.session_state.alert_history)

# # # # # # # # #     if mode == "Video":
# # # # # # # # #         alert_df = alert_df.sort_values(by="Count", ascending=False).head(5)

# # # # # # # # #     st.dataframe(alert_df, use_container_width=True)
# # # # # # # # # else:
# # # # # # # # #     st.info("No crowd alerts generated yet.")

# # # # # # # # # # ----------------------------
# # # # # # # # # # DASHBOARD
# # # # # # # # # # ----------------------------
# # # # # # # # # st.subheader("📊 Crowd Analytics Dashboard")

# # # # # # # # # if dense_mode:
# # # # # # # # #     show_graph(st.session_state.dense_history, "Dense Mode Trend", mode)
# # # # # # # # # else:
# # # # # # # # #     show_graph(st.session_state.normal_history, "Normal Mode Trend", mode)


# # # # # # # # # # ----------------------------
# # # # # # # # # # DOWNLOAD REPORT
# # # # # # # # # # ----------------------------
# # # # # # # # # if st.session_state.last_output is not None and st.session_state.last_heatmap is not None:

# # # # # # # # #     mode_name = "Dense Mode" if dense_mode else "Normal Mode"
# # # # # # # # #     graph_data = st.session_state.dense_history if dense_mode else st.session_state.normal_history

# # # # # # # # #     if mode == "Image":
# # # # # # # # #         left, center, right = zone_analysis(
# # # # # # # # #             st.session_state.last_output.shape[1],
# # # # # # # # #             centers if 'centers' in locals() else []
# # # # # # # # #         )
# # # # # # # # #     else:
# # # # # # # # #         left, center, right = zone_analysis(
# # # # # # # # #             st.session_state.last_output.shape[1],
# # # # # # # # #             list(tracker.objects.values()) if len(tracker.objects) > 0 else []
# # # # # # # # #         )

# # # # # # # # #     pdf_data = generate_pdf_report(
# # # # # # # # #         mode_name=mode_name,
# # # # # # # # #         count=st.session_state.last_count,
# # # # # # # # #         left=left,
# # # # # # # # #         center=center,
# # # # # # # # #         right=right,
# # # # # # # # #         detection_img=st.session_state.last_output,
# # # # # # # # #         heatmap_img=st.session_state.last_heatmap,
# # # # # # # # #         graph_data=graph_data
# # # # # # # # #     )

# # # # # # # # #     st.download_button(
# # # # # # # # #         label="📄 Download Report",
# # # # # # # # #         data=pdf_data,
# # # # # # # # #         file_name="Crowd_Report.pdf",
# # # # # # # # #         mime="application/pdf"
# # # # # # # # #     )









# # # # # # # # # # # ----------------------------------------------------










# # # # # # # # import streamlit as st
# # # # # # # # import cv2
# # # # # # # # import numpy as np
# # # # # # # # import imutils
# # # # # # # # import pandas as pd
# # # # # # # # import tempfile
# # # # # # # # from config import YOLO_CONFIG
# # # # # # # # from collections import OrderedDict
# # # # # # # # import math
# # # # # # # # from datetime import datetime
# # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # from reportlab.lib.pagesizes import A4
# # # # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # # # ----------------------------
# # # # # # # # # PAGE CONFIG
# # # # # # # # # ----------------------------
# # # # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")

# # # # # # # # st.markdown("""
# # # # # # # # <style>
# # # # # # # # body { background-color: #0e1117; color: white; }
# # # # # # # # </style>
# # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # ----------------------------
# # # # # # # # # SESSION STATE
# # # # # # # # # ----------------------------
# # # # # # # # defaults = {
# # # # # # # #     "run_video": False,
# # # # # # # #     "last_output": None,
# # # # # # # #     "last_heatmap": None,
# # # # # # # #     "last_count": 0,
# # # # # # # #     "dense_history": [],
# # # # # # # #     "normal_history": [],
# # # # # # # #     "all_centers": [],
# # # # # # # #     "alert_history": []
# # # # # # # # }

# # # # # # # # for key, value in defaults.items():
# # # # # # # #     if key not in st.session_state:
# # # # # # # #         st.session_state[key] = value

# # # # # # # # # ----------------------------
# # # # # # # # # LOAD MODEL
# # # # # # # # # ----------------------------
# # # # # # # # @st.cache_resource
# # # # # # # # def load_model():
# # # # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # # # #     )
# # # # # # # #     ln = net.getLayerNames()
# # # # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # # # #     return net, ln

# # # # # # # # net, ln = load_model()

# # # # # # # # # ----------------------------
# # # # # # # # # CROWD TREND PREDICTION (FIXED)
# # # # # # # # # ----------------------------
# # # # # # # # def predict_crowd_trend(data, steps=5):
# # # # # # # #     if len(data) < 3:
# # # # # # # #         return []

# # # # # # # #     predictions = []
# # # # # # # #     temp = data.copy()

# # # # # # # #     for _ in range(steps):
# # # # # # # #         next_val = int(sum(temp[-3:]) / 3)
# # # # # # # #         predictions.append(next_val)
# # # # # # # #         temp.append(next_val)

# # # # # # # #     return predictions

# # # # # # # # # ----------------------------
# # # # # # # # # TRACKER
# # # # # # # # # ----------------------------
# # # # # # # # class CentroidTracker:
# # # # # # # #     def __init__(self, max_distance=50):
# # # # # # # #         self.next_id = 0
# # # # # # # #         self.objects = OrderedDict()
# # # # # # # #         self.max_distance = max_distance

# # # # # # # #     def update(self, detections):
# # # # # # # #         if len(detections) == 0:
# # # # # # # #             return self.objects

# # # # # # # #         input_centroids = np.array(detections)

# # # # # # # #         if len(self.objects) == 0:
# # # # # # # #             for c in input_centroids:
# # # # # # # #                 self.objects[self.next_id] = c
# # # # # # # #                 self.next_id += 1
# # # # # # # #         else:
# # # # # # # #             object_ids = list(self.objects.keys())
# # # # # # # #             object_centroids = list(self.objects.values())
# # # # # # # #             updated_objects = OrderedDict()

# # # # # # # #             for new_c in input_centroids:
# # # # # # # #                 min_dist = float("inf")
# # # # # # # #                 match_id = None

# # # # # # # #                 for obj_id, old_c in zip(object_ids, object_centroids):
# # # # # # # #                     dist = math.dist(new_c, old_c)
# # # # # # # #                     if dist < min_dist and dist < self.max_distance:
# # # # # # # #                         min_dist = dist
# # # # # # # #                         match_id = obj_id

# # # # # # # #                 if match_id is not None and match_id not in updated_objects:
# # # # # # # #                     updated_objects[match_id] = new_c
# # # # # # # #                 else:
# # # # # # # #                     updated_objects[self.next_id] = new_c
# # # # # # # #                     self.next_id += 1

# # # # # # # #             self.objects = updated_objects

# # # # # # # #         return self.objects

# # # # # # # # tracker = CentroidTracker()

# # # # # # # # # ----------------------------
# # # # # # # # # REMOVE DUPLICATES
# # # # # # # # # ----------------------------
# # # # # # # # def remove_duplicate_centers(centers, min_dist=22):
# # # # # # # #     filtered = []
# # # # # # # #     for c in centers:
# # # # # # # #         duplicate = False
# # # # # # # #         for fc in filtered:
# # # # # # # #             if math.dist(c, fc) < min_dist:
# # # # # # # #                 duplicate = True
# # # # # # # #                 break
# # # # # # # #         if not duplicate:
# # # # # # # #             filtered.append(c)
# # # # # # # #     return filtered

# # # # # # # # # ----------------------------
# # # # # # # # # HEATMAP
# # # # # # # # # ----------------------------
# # # # # # # # def generate_heatmap(image, centers, dense_mode=False):
# # # # # # # #     h, w = image.shape[:2]
# # # # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # # # #     for (x, y) in centers:
# # # # # # # #         if 0 <= x < w and 0 <= y < h:
# # # # # # # #             heatmap[y, x] += 3.0 if dense_mode else 1.8

# # # # # # # #     blur_size = (101, 101) if dense_mode else (71, 71)
# # # # # # # #     heatmap = cv2.GaussianBlur(heatmap, blur_size, 0)
# # # # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # # # #     heatmap = np.uint8(heatmap)
# # # # # # # #     heatmap = 255 - heatmap
# # # # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # # # #     return heatmap

# # # # # # # # # ----------------------------
# # # # # # # # # ZONE ANALYSIS
# # # # # # # # # ----------------------------
# # # # # # # # def zone_analysis(width, centers):
# # # # # # # #     left, center, right = 0, 0, 0

# # # # # # # #     for (x, y) in centers:
# # # # # # # #         if x < width / 3:
# # # # # # # #             left += 1
# # # # # # # #         elif x < 2 * width / 3:
# # # # # # # #             center += 1
# # # # # # # #         else:
# # # # # # # #             right += 1

# # # # # # # #     return left, center, right

# # # # # # # # # ----------------------------
# # # # # # # # # ALERT SYSTEM
# # # # # # # # # ----------------------------
# # # # # # # # def get_crowd_alert(count, mode="Video"):
# # # # # # # #     current_time = datetime.now().strftime("%H:%M:%S")

# # # # # # # #     if mode == "Image":
# # # # # # # #         if count <= 20:
# # # # # # # #             status = "Safe"
# # # # # # # #         elif count <= 40:
# # # # # # # #             status = "Crowded"
# # # # # # # #         else:
# # # # # # # #             status = "Highly Crowded"
# # # # # # # #     else:
# # # # # # # #         if count <= 20:
# # # # # # # #             status = "Safe"
# # # # # # # #         elif count <= 40:
# # # # # # # #             status = "Moderate Crowd"
# # # # # # # #         elif count <= 60:
# # # # # # # #             status = "High Crowd Alert"
# # # # # # # #         else:
# # # # # # # #             status = "Danger"

# # # # # # # #     return {
# # # # # # # #         "Time": current_time,
# # # # # # # #         "Count": count,
# # # # # # # #         "Status": status
# # # # # # # #     }

# # # # # # # # # ----------------------------
# # # # # # # # # DETECTION
# # # # # # # # # ----------------------------
# # # # # # # # def detect_people(image, input_size, conf, nms, dense_mode, is_video=False):
# # # # # # # #     image = imutils.resize(image, width=800)
# # # # # # # #     H, W = image.shape[:2]

# # # # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # # # #                                  swapRB=True, crop=False)

# # # # # # # #     net.setInput(blob)
# # # # # # # #     outputs = net.forward(ln)

# # # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # # #     for output in outputs:
# # # # # # # #         for det in output:
# # # # # # # #             scores = det[5:]
# # # # # # # #             classID = np.argmax(scores)
# # # # # # # #             confidence = scores[classID]

# # # # # # # #             if classID == 0 and confidence > conf:
# # # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # # #                 x = int(cx - w / 2)
# # # # # # # #                 y = int(cy - h / 2)

# # # # # # # #                 if w * h > 500:
# # # # # # # #                     boxes.append([x, y, int(w), int(h)])
# # # # # # # #                     centers.append((cx, cy))
# # # # # # # #                     confidences.append(float(confidence))

# # # # # # # #     filtered_centers = []

# # # # # # # #     if len(boxes) > 0:
# # # # # # # #         idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # # # #         if len(idxs) > 0:
# # # # # # # #             for i in idxs.flatten():
# # # # # # # #                 filtered_centers.append(centers[i])

# # # # # # # #     return image, filtered_centers

# # # # # # # # # ----------------------------
# # # # # # # # # GRAPH
# # # # # # # # # ----------------------------
# # # # # # # # def show_graph(data, title):
# # # # # # # #     if len(data) == 0:
# # # # # # # #         return

# # # # # # # #     df = pd.DataFrame({
# # # # # # # #         "Frame": list(range(1, len(data) + 1)),
# # # # # # # #         "Count": data
# # # # # # # #     })

# # # # # # # #     st.subheader(title)
# # # # # # # #     st.line_chart(df.set_index("Frame"))

# # # # # # # # # ----------------------------
# # # # # # # # # UI
# # # # # # # # # ----------------------------
# # # # # # # # st.title("Crowd Counting System")

# # # # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # # # # dense_mode = st.sidebar.toggle("Dense Mode", False)

# # # # # # # # uploaded = st.file_uploader("Upload File", type=["jpg", "png", "jpeg", "mp4"])

# # # # # # # # col1, col2 = st.columns(2)

# # # # # # # # # ----------------------------
# # # # # # # # # IMAGE MODE
# # # # # # # # # ----------------------------
# # # # # # # # if uploaded and mode == "Image":

# # # # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # # # #     if st.button("Detect"):

# # # # # # # #         output, centers = detect_people(image, 640, 0.4, 0.4, dense_mode)

# # # # # # # #         count = len(centers)

# # # # # # # #         st.session_state.dense_history.append(count)

# # # # # # # #         heatmap = generate_heatmap(image.copy(), centers, dense_mode)

# # # # # # # #         st.session_state.last_output = output
# # # # # # # #         st.session_state.last_heatmap = heatmap
# # # # # # # #         st.session_state.last_count = count

# # # # # # # #         # 🔮 PREDICTION
# # # # # # # #         history = st.session_state.dense_history
# # # # # # # #         predictions = predict_crowd_trend(history, 5)

# # # # # # # #         st.subheader("🔮 Crowd Trend Prediction")
# # # # # # # #         st.write(predictions)

# # # # # # # #         combined = history + predictions
# # # # # # # #         st.line_chart(combined)

# # # # # # # #         col1.image(output)
# # # # # # # #         col2.image(heatmap)

# # # # # # # # # ----------------------------
# # # # # # # # # VIDEO MODE
# # # # # # # # # ----------------------------
# # # # # # # # if uploaded and mode == "Video":

# # # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # # #     tfile.write(uploaded.read())

# # # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # # #     if st.button("Start"):
# # # # # # # #         st.session_state.run_video = True

# # # # # # # #     frame_box = col1.empty()

# # # # # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # # # # #         ret, frame = cap.read()
# # # # # # # #         if not ret:
# # # # # # # #             break

# # # # # # # #         output, centers = detect_people(frame, 640, 0.4, 0.4, dense_mode)

# # # # # # # #         count = len(centers)
# # # # # # # #         st.session_state.dense_history.append(count)

# # # # # # # #         heatmap = generate_heatmap(frame.copy(), centers, dense_mode)

# # # # # # # #         frame_box.image(output, channels="BGR")

# # # # # # # #     cap.release()





# # # # # # # # ---------------------------------------------------------------------------------












# # # # # # # import streamlit as st
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # import imutils
# # # # # # # import pandas as pd
# # # # # # # import tempfile
# # # # # # # from config import YOLO_CONFIG
# # # # # # # import math
# # # # # # # from datetime import datetime
# # # # # # # import matplotlib.pyplot as plt

# # # # # # # from reportlab.lib.pagesizes import A4
# # # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # # ----------------------------
# # # # # # # # PAGE CONFIG
# # # # # # # # ----------------------------
# # # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # # # # st.title("Crowd Counting System")

# # # # # # # # ----------------------------
# # # # # # # # SESSION STATE
# # # # # # # # ----------------------------
# # # # # # # if "history" not in st.session_state:
# # # # # # #     st.session_state.history = []

# # # # # # # if "run_video" not in st.session_state:
# # # # # # #     st.session_state.run_video = False

# # # # # # # if "last_output" not in st.session_state:
# # # # # # #     st.session_state.last_output = None

# # # # # # # if "last_heatmap" not in st.session_state:
# # # # # # #     st.session_state.last_heatmap = None

# # # # # # # # ----------------------------
# # # # # # # # LOAD YOLO MODEL
# # # # # # # # ----------------------------
# # # # # # # @st.cache_resource
# # # # # # # def load_model():
# # # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # # #     )
# # # # # # #     ln = net.getLayerNames()
# # # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # # #     return net, ln

# # # # # # # net, ln = load_model()

# # # # # # # # ----------------------------
# # # # # # # # DETECTION (NMS + CONFIDENCE)
# # # # # # # # ----------------------------
# # # # # # # def detect_people(image, input_size, conf_thresh, nms_thresh):
# # # # # # #     image = imutils.resize(image, width=800)
# # # # # # #     H, W = image.shape[:2]

# # # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # # #                                  swapRB=True, crop=False)

# # # # # # #     net.setInput(blob)
# # # # # # #     outputs = net.forward(ln)

# # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # #     for output in outputs:
# # # # # # #         for det in output:
# # # # # # #             scores = det[5:]
# # # # # # #             classID = np.argmax(scores)
# # # # # # #             confidence = scores[classID]

# # # # # # #             if classID == 0 and confidence > conf_thresh:
# # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # #                 x = int(cx - w / 2)
# # # # # # #                 y = int(cy - h / 2)

# # # # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # # # #                 centers.append((cx, cy))
# # # # # # #                 confidences.append(float(confidence))

# # # # # # #     final_centers = []

# # # # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

# # # # # # #     if len(idxs) > 0:
# # # # # # #         for i in idxs.flatten():
# # # # # # #             final_centers.append(centers[i])

# # # # # # #             x, y, w, h = boxes[i]
# # # # # # #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # # # #     return image, final_centers

# # # # # # # # ----------------------------
# # # # # # # # HEATMAP
# # # # # # # # ----------------------------
# # # # # # # def generate_heatmap(image, centers):
# # # # # # #     h, w = image.shape[:2]
# # # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # # #     for x, y in centers:
# # # # # # #         x = max(0, min(x, w - 1))
# # # # # # #         y = max(0, min(y, h - 1))
# # # # # # #         heatmap[y, x] += 1

# # # # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # # #     heatmap = np.uint8(heatmap)
# # # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # # #     return heatmap

# # # # # # # # ----------------------------
# # # # # # # # GRAPH
# # # # # # # # ----------------------------
# # # # # # # def show_graph(data):
# # # # # # #     if len(data) == 0:
# # # # # # #         return

# # # # # # #     df = pd.DataFrame({
# # # # # # #         "Frame": list(range(1, len(data)+1)),
# # # # # # #         "Count": data
# # # # # # #     })

# # # # # # #     st.line_chart(df.set_index("Frame"))

# # # # # # # # ----------------------------
# # # # # # # # PDF REPORT
# # # # # # # # ----------------------------
# # # # # # # def generate_pdf(image, heatmap, history, count):

# # # # # # #     cv2.imwrite("det.jpg", image)
# # # # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # # # #     plt.figure()
# # # # # # #     plt.plot(history)
# # # # # # #     plt.title("Crowd Trend")
# # # # # # #     plt.savefig("graph.png")
# # # # # # #     plt.close()

# # # # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # # # #     styles = getSampleStyleSheet()
# # # # # # #     story = []

# # # # # # #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# # # # # # #     story.append(Spacer(1, 10))

# # # # # # #     story.append(Paragraph(f"Total Count: {count}", styles["Normal"]))
# # # # # # #     story.append(Spacer(1, 10))

# # # # # # #     story.append(Paragraph("Detection Output", styles["Heading2"]))
# # # # # # #     story.append(RLImage("det.jpg", width=400, height=250))

# # # # # # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # # # # # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # # # # # #     story.append(Paragraph("Trend Graph", styles["Heading2"]))
# # # # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # # # #     pdf.build(story)

# # # # # # #     with open("report.pdf", "rb") as f:
# # # # # # #         return f.read()

# # # # # # # # ----------------------------
# # # # # # # # UI
# # # # # # # # ----------------------------
# # # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])

# # # # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)
# # # # # # # input_size = 640

# # # # # # # uploaded = st.file_uploader("Upload Image/Video", type=["jpg", "png", "mp4"])

# # # # # # # col1, col2 = st.columns(2)

# # # # # # # # ----------------------------
# # # # # # # # IMAGE MODE
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Image":

# # # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # # #     if st.button("Detect"):

# # # # # # #         output, centers = detect_people(image, input_size, conf, nms)
# # # # # # #         count = len(centers)

# # # # # # #         st.session_state.history.append(count)

# # # # # # #         heatmap = generate_heatmap(image.copy(), centers)

# # # # # # #         st.session_state.last_output = output
# # # # # # #         st.session_state.last_heatmap = heatmap

# # # # # # #         col1.image(output, channels="BGR")
# # # # # # #         col2.image(heatmap, channels="BGR")

# # # # # # #         st.subheader("📊 Crowd Trend")
# # # # # # #         show_graph(st.session_state.history)

# # # # # # #         pdf = generate_pdf(output, heatmap, st.session_state.history, count)

# # # # # # #         st.download_button(
# # # # # # #             "📄 Download Report",
# # # # # # #             data=pdf,
# # # # # # #             file_name="crowd_report.pdf",
# # # # # # #             mime="application/pdf"
# # # # # # #         )

# # # # # # # # ----------------------------
# # # # # # # # VIDEO MODE (START / STOP FIXED)
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Video":

# # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # #     tfile.write(uploaded.read())

# # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # #     colA, colB = st.columns(2)

# # # # # # #     # ✅ START / STOP BUTTONS SIDE BY SIDE
# # # # # # #     if colA.button("▶ Start"):
# # # # # # #         st.session_state.run_video = True

# # # # # # #     if colB.button("⛔ Stop"):
# # # # # # #         st.session_state.run_video = False

# # # # # # #     frame_box = col1.empty()

# # # # # # #     if st.session_state.run_video:

# # # # # # #         while cap.isOpened() and st.session_state.run_video:

# # # # # # #             ret, frame = cap.read()
# # # # # # #             if not ret:
# # # # # # #                 break

# # # # # # #             output, centers = detect_people(frame, input_size, conf, nms)

# # # # # # #             count = len(centers)
# # # # # # #             st.session_state.history.append(count)

# # # # # # #             heatmap = generate_heatmap(frame.copy(), centers)

# # # # # # #             st.session_state.last_output = output
# # # # # # #             st.session_state.last_heatmap = heatmap

# # # # # # #             frame_box.image(output, channels="BGR")

# # # # # # #         cap.release()

# # # # # # #     st.subheader("📊 Crowd Trend")
# # # # # # #     show_graph(st.session_state.history)

# # # # # # #     # PDF only if data exists
# # # # # # #     if st.session_state.last_output is not None:

# # # # # # #         pdf = generate_pdf(
# # # # # # #             st.session_state.last_output,
# # # # # # #             st.session_state.last_heatmap,
# # # # # # #             st.session_state.history,
# # # # # # #             st.session_state.history[-1]
# # # # # # #         )

# # # # # # #         st.download_button(
# # # # # # #             "📄 Download Report",
# # # # # # #             data=pdf,
# # # # # # #             file_name="crowd_report.pdf",
# # # # # # #             mime="application/pdf"
# # # # # # #         )














# # # # # # # import streamlit as st
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # import imutils
# # # # # # # import pandas as pd
# # # # # # # import tempfile
# # # # # # # from config import YOLO_CONFIG
# # # # # # # from datetime import datetime
# # # # # # # import matplotlib.pyplot as plt

# # # # # # # from reportlab.lib.pagesizes import A4
# # # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # # ----------------------------
# # # # # # # # PAGE CONFIG
# # # # # # # # ----------------------------
# # # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # # # # st.title("🚶 Crowd Counting System")

# # # # # # # # ----------------------------
# # # # # # # # SESSION STATE
# # # # # # # # ----------------------------
# # # # # # # if "history" not in st.session_state:
# # # # # # #     st.session_state.history = []

# # # # # # # if "alerts" not in st.session_state:
# # # # # # #     st.session_state.alerts = []

# # # # # # # if "run_video" not in st.session_state:
# # # # # # #     st.session_state.run_video = False

# # # # # # # # ----------------------------
# # # # # # # # LOAD MODEL
# # # # # # # # ----------------------------
# # # # # # # @st.cache_resource
# # # # # # # def load_model():
# # # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # # #     )
# # # # # # #     ln = net.getLayerNames()
# # # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # # #     return net, ln

# # # # # # # net, ln = load_model()

# # # # # # # # ----------------------------
# # # # # # # # ALERT SYSTEM (YOUR LOGIC RESTORED)
# # # # # # # # ----------------------------
# # # # # # # def get_crowd_alert(count, mode="Video"):
# # # # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # # # #     if mode == "Image":
# # # # # # #         if count <= 20:
# # # # # # #             status = "Safe"
# # # # # # #             message = "Crowd level is under control."
# # # # # # #         elif count <= 40:
# # # # # # #             status = "Crowded"
# # # # # # #             message = "Crowded area detected."
# # # # # # #         else:
# # # # # # #             status = "Highly Crowded"
# # # # # # #             message = "High crowd density detected."

# # # # # # #     else:
# # # # # # #         if count <= 20:
# # # # # # #             status = "Safe"
# # # # # # #             message = "Crowd level is under control."
# # # # # # #         elif count <= 40:
# # # # # # #             status = "Moderate Crowd"
# # # # # # #             message = "Moderate crowd detected. Monitor area."
# # # # # # #         elif count <= 60:
# # # # # # #             status = "High Crowd Alert"
# # # # # # #             message = "High crowd density detected. Take precautions."
# # # # # # #         else:
# # # # # # #             status = "Danger"
# # # # # # #             message = "Critical crowd level detected. Immediate action required."

# # # # # # #     return {
# # # # # # #         "Time": time_now,
# # # # # # #         "Count": count,
# # # # # # #         "Status": status,
# # # # # # #         "Message": message
# # # # # # #     }

# # # # # # # # ----------------------------
# # # # # # # # DETECTION
# # # # # # # # ----------------------------
# # # # # # # # def detect_people(image, input_size, conf, nms):
# # # # # # # #     image = imutils.resize(image, width=800)
# # # # # # # #     H, W = image.shape[:2]

# # # # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # # # #                                  swapRB=True, crop=False)

# # # # # # # #     net.setInput(blob)
# # # # # # # #     outputs = net.forward(ln)

# # # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # # #     for output in outputs:
# # # # # # # #         for det in output:
# # # # # # # #             scores = det[5:]
# # # # # # # #             classID = np.argmax(scores)
# # # # # # # #             confidence = scores[classID]

# # # # # # # #             if classID == 0 and confidence > conf:
# # # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # # #                 x = int(cx - w / 2)
# # # # # # # #                 y = int(cy - h / 2)

# # # # # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # # # # #                 centers.append((cx, cy))
# # # # # # # #                 confidences.append(float(confidence))

# # # # # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # # # #     final_centers = []

# # # # # # # #     if len(idxs) > 0:
# # # # # # # #         for i in idxs.flatten():
# # # # # # # #             final_centers.append(centers[i])
# # # # # # # #             x, y, w, h = boxes[i]
# # # # # # # #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # # # # #     return image, final_centers

# # # # # # # # ----------------------------
# # # # # # # # DETECTION (DOTS ONLY FOR VIDEO)
# # # # # # # # ----------------------------
# # # # # # # def detect_people(image, input_size, conf, nms, video_mode=False):
# # # # # # #     image = imutils.resize(image, width=800)
# # # # # # #     H, W = image.shape[:2]

# # # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # # #                                  swapRB=True, crop=False)

# # # # # # #     net.setInput(blob)
# # # # # # #     outputs = net.forward(ln)

# # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # #     for output in outputs:
# # # # # # #         for det in output:
# # # # # # #             scores = det[5:]
# # # # # # #             classID = np.argmax(scores)
# # # # # # #             confidence = scores[classID]

# # # # # # #             if classID == 0 and confidence > conf:
# # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # #                 x = int(cx - w / 2)
# # # # # # #                 y = int(cy - h / 2)

# # # # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # # # #                 centers.append((cx, cy))
# # # # # # #                 confidences.append(float(confidence))

# # # # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # # #     final_centers = []

# # # # # # #     if len(idxs) > 0:
# # # # # # #         for i in idxs.flatten():
# # # # # # #             final_centers.append(centers[i])

# # # # # # #             cx, cy = centers[i]

# # # # # # #             # 🔵 DOTS ONLY IN VIDEO MODE
# # # # # # #             if video_mode:
# # # # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # # # #             else:
# # # # # # #                 x, y, w, h = boxes[i]
# # # # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # # # #     return image, final_centers

# # # # # # # # ----------------------------
# # # # # # # # HEATMAP
# # # # # # # # ----------------------------
# # # # # # # def generate_heatmap(image, centers):
# # # # # # #     h, w = image.shape[:2]
# # # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # # #     for x, y in centers:
# # # # # # #         x = max(0, min(x, w - 1))
# # # # # # #         y = max(0, min(y, h - 1))
# # # # # # #         heatmap[y, x] += 1

# # # # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # # #     heatmap = np.uint8(heatmap)
# # # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # # #     return heatmap

# # # # # # # # ----------------------------
# # # # # # # # GRAPH
# # # # # # # # ----------------------------
# # # # # # # def show_graph(data):
# # # # # # #     if len(data) == 0:
# # # # # # #         return

# # # # # # #     df = pd.DataFrame({
# # # # # # #         "Frame": list(range(1, len(data)+1)),
# # # # # # #         "Count": data
# # # # # # #     })

# # # # # # #     st.line_chart(df.set_index("Frame"))

# # # # # # # # ----------------------------
# # # # # # # # UI
# # # # # # # # ----------------------------
# # # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # # # uploaded = st.file_uploader("Upload Image / Video", type=["jpg", "png", "mp4"])

# # # # # # # col1, col2 = st.columns(2)

# # # # # # # # ----------------------------
# # # # # # # # IMAGE MODE
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Image":

# # # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # # #     if st.button("Detect"):

# # # # # # #         output, centers = detect_people(image, 640, conf, nms)
# # # # # # #         count = len(centers)

# # # # # # #         st.session_state.history.append(count)

# # # # # # #         heatmap = generate_heatmap(image.copy(), centers)

# # # # # # #         alert = get_crowd_alert(count, "Image")
# # # # # # #         st.session_state.alerts.append(alert)

# # # # # # #         col1.image(output, channels="BGR", caption=f"People Count: {count}")
# # # # # # #         col2.image(heatmap, channels="BGR")

# # # # # # #         st.subheader("📊 Crowd Trend")
# # # # # # #         show_graph(st.session_state.history)

# # # # # # # # ----------------------------
# # # # # # # # VIDEO MODE
# # # # # # # # ----------------------------
# # # # # # # # if uploaded and mode == "Video":

# # # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # # #     tfile.write(uploaded.read())

# # # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # # #     c1, c2 = st.columns(2)

# # # # # # # #     # START / STOP
# # # # # # # #     if c1.button("▶ Start"):
# # # # # # # #         st.session_state.run_video = True

# # # # # # # #     if c2.button("⛔ Stop"):
# # # # # # # #         st.session_state.run_video = False

# # # # # # # #     frame_box = col1.empty()
# # # # # # # #     heat_box = col2.empty()

# # # # # # # #     if st.session_state.run_video:

# # # # # # # #         while cap.isOpened() and st.session_state.run_video:

# # # # # # # #             ret, frame = cap.read()
# # # # # # # #             if not ret:
# # # # # # # #                 break

# # # # # # # #             output, centers = detect_people(frame, 640, conf, nms)
# # # # # # # #             count = len(centers)

# # # # # # # #             st.session_state.history.append(count)

# # # # # # # #             heatmap = generate_heatmap(frame.copy(), centers)

# # # # # # # #             alert = get_crowd_alert(count, "Video")
# # # # # # # #             st.session_state.alerts.append(alert)

# # # # # # # #             frame_box.image(output, channels="BGR", caption=f"Count: {count}")
# # # # # # # #             heat_box.image(heatmap, channels="BGR")

# # # # # # # #         cap.release()


# # # # # # # # ----------------------------
# # # # # # # # VIDEO MODE (UPDATED)
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Video":

# # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # #     tfile.write(uploaded.read())

# # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # #     c1, c2 = st.columns(2)

# # # # # # #     if c1.button("▶ Start"):
# # # # # # #         st.session_state.run_video = True
# # # # # # #         st.session_state.history = []
# # # # # # #         st.session_state.alerts = []

# # # # # # #     if c2.button("⛔ Stop"):
# # # # # # #         st.session_state.run_video = False

# # # # # # #     frame_box = col1.empty()
# # # # # # #     heat_box = col2.empty()

# # # # # # #     final_frame = None
# # # # # # #     final_heatmap = None

# # # # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # # # #         ret, frame = cap.read()
# # # # # # #         if not ret:
# # # # # # #             break

# # # # # # #         output, centers = detect_people(frame, 640, conf, nms, video_mode=True)

# # # # # # #         count = len(centers)
# # # # # # #         st.session_state.history.append(count)

# # # # # # #         heatmap = generate_heatmap(frame.copy(), centers)

# # # # # # #         alert = get_crowd_alert(count, "Video")
# # # # # # #         st.session_state.alerts.append(alert)

# # # # # # #         frame_box.image(output, channels="BGR")
# # # # # # #         heat_box.image(heatmap, channels="BGR")

# # # # # # #         final_frame = output
# # # # # # #         final_heatmap = heatmap

# # # # # # #     cap.release()

# # # # # # #     # ----------------------------
# # # # # # #     # SHOW FINAL OUTPUT AFTER STOP
# # # # # # #     # ----------------------------
# # # # # # #     if not st.session_state.run_video and final_frame is not None:

# # # # # # #         st.subheader("📌 Final Output After Stop")

# # # # # # #         col1.image(final_frame, caption="Final Detection (Dots)", channels="BGR")
# # # # # # #         col2.image(final_heatmap, caption="Final Heatmap", channels="BGR")
# # # # # # # # ----------------------------
# # # # # # # # ALERT TABLE (YOUR REQUEST)
# # # # # # # # ----------------------------
# # # # # # # # st.subheader("🚨 Alert Monitoring Table")

# # # # # # # # if len(st.session_state.alerts) > 0:
# # # # # # # #     alert_df = pd.DataFrame(st.session_state.alerts)
# # # # # # # #     st.dataframe(alert_df, use_container_width=True)
# # # # # # # # else:
# # # # # # # #     st.info("No alerts generated yet.")

# # # # # # # # ----------------------------
# # # # # # # # ALERT TABLE (TOP 5 ONLY FOR VIDEO)
# # # # # # # # ----------------------------
# # # # # # # st.subheader("🚨 Alert Monitoring Table")

# # # # # # # if len(st.session_state.alerts) > 0:

# # # # # # #     alert_df = pd.DataFrame(st.session_state.alerts)

# # # # # # #     # 🔥 VIDEO MODE → TOP 5 HIGHEST COUNTS
# # # # # # #     if mode == "Video":
# # # # # # #         alert_df = alert_df.sort_values(by="Count", ascending=False).head(5)

# # # # # # #     st.dataframe(alert_df, use_container_width=True)

# # # # # # # else:
# # # # # # #     st.info("No alerts generated yet.")

# # # # # # # # ----------------------------
# # # # # # # # GRAPH
# # # # # # # # ----------------------------
# # # # # # # st.subheader("📈 Crowd Trend Graph")
# # # # # # # show_graph(st.session_state.history)

















# # # # # # # import streamlit as st
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # import imutils
# # # # # # # import pandas as pd
# # # # # # # import tempfile
# # # # # # # from config import YOLO_CONFIG
# # # # # # # from datetime import datetime
# # # # # # # import matplotlib.pyplot as plt

# # # # # # # from reportlab.lib.pagesizes import A4
# # # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # # ----------------------------
# # # # # # # # PAGE CONFIG
# # # # # # # # ----------------------------
# # # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # # # # st.title("🚶 Crowd Counting System")

# # # # # # # # ----------------------------
# # # # # # # # SESSION STATE
# # # # # # # # ----------------------------
# # # # # # # if "history" not in st.session_state:
# # # # # # #     st.session_state.history = []

# # # # # # # if "alerts" not in st.session_state:
# # # # # # #     st.session_state.alerts = []

# # # # # # # if "run_video" not in st.session_state:
# # # # # # #     st.session_state.run_video = False

# # # # # # # if "final_frame" not in st.session_state:
# # # # # # #     st.session_state.final_frame = None

# # # # # # # if "final_heatmap" not in st.session_state:
# # # # # # #     st.session_state.final_heatmap = None


# # # # # # # # ----------------------------
# # # # # # # # LOAD MODEL
# # # # # # # # ----------------------------
# # # # # # # @st.cache_resource
# # # # # # # def load_model():
# # # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # # #     )
# # # # # # #     ln = net.getLayerNames()
# # # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # # #     return net, ln

# # # # # # # net, ln = load_model()


# # # # # # # # ----------------------------
# # # # # # # # ALERT SYSTEM
# # # # # # # # ----------------------------
# # # # # # # def get_crowd_alert(count, mode="Video"):
# # # # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # # # #     if mode == "Image":
# # # # # # #         if count <= 20:
# # # # # # #             status = "Safe"
# # # # # # #             message = "Crowd level is under control."
# # # # # # #         elif count <= 40:
# # # # # # #             status = "Crowded"
# # # # # # #             message = "Crowded area detected."
# # # # # # #         else:
# # # # # # #             status = "Highly Crowded"
# # # # # # #             message = "High crowd density detected."
# # # # # # #     else:
# # # # # # #         if count <= 20:
# # # # # # #             status = "Safe"
# # # # # # #             message = "Crowd level is under control."
# # # # # # #         elif count <= 40:
# # # # # # #             status = "Moderate Crowd"
# # # # # # #             message = "Moderate crowd detected. Monitor area."
# # # # # # #         elif count <= 60:
# # # # # # #             status = "High Crowd Alert"
# # # # # # #             message = "High crowd density detected."
# # # # # # #         else:
# # # # # # #             status = "Danger"
# # # # # # #             message = "Critical crowd level detected."

# # # # # # #     return {
# # # # # # #         "Time": time_now,
# # # # # # #         "Count": count,
# # # # # # #         "Status": status,
# # # # # # #         "Message": message
# # # # # # #     }


# # # # # # # # ----------------------------
# # # # # # # # DETECTION (DOTS + BOXES)
# # # # # # # # ----------------------------
# # # # # # # def detect_people(image, input_size, conf, nms, video_mode=False):
# # # # # # #     image = imutils.resize(image, width=800)
# # # # # # #     H, W = image.shape[:2]

# # # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # # #                                  swapRB=True, crop=False)

# # # # # # #     net.setInput(blob)
# # # # # # #     outputs = net.forward(ln)

# # # # # # #     boxes, centers, confidences = [], [], []

# # # # # # #     for output in outputs:
# # # # # # #         for det in output:
# # # # # # #             scores = det[5:]
# # # # # # #             classID = np.argmax(scores)
# # # # # # #             confidence = scores[classID]

# # # # # # #             if classID == 0 and confidence > conf:
# # # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # # #                 x = int(cx - w / 2)
# # # # # # #                 y = int(cy - h / 2)

# # # # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # # # #                 centers.append((cx, cy))
# # # # # # #                 confidences.append(float(confidence))

# # # # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # # #     final_centers = []

# # # # # # #     if len(idxs) > 0:
# # # # # # #         for i in idxs.flatten():
# # # # # # #             final_centers.append(centers[i])

# # # # # # #             cx, cy = centers[i]

# # # # # # #             # 🔵 DOTS FOR VIDEO
# # # # # # #             if video_mode:
# # # # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # # # #             else:
# # # # # # #                 x, y, w, h = boxes[i]
# # # # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # # # #     return image, final_centers


# # # # # # # # ----------------------------
# # # # # # # # HEATMAP
# # # # # # # # ----------------------------
# # # # # # # def generate_heatmap(image, centers):
# # # # # # #     h, w = image.shape[:2]
# # # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # # #     for x, y in centers:
# # # # # # #         x = max(0, min(x, w - 1))
# # # # # # #         y = max(0, min(y, h - 1))
# # # # # # #         heatmap[y, x] += 1

# # # # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # # #     heatmap = np.uint8(heatmap)
# # # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # # #     return heatmap


# # # # # # # # ----------------------------
# # # # # # # # GRAPH
# # # # # # # # ----------------------------
# # # # # # # def show_graph(data):
# # # # # # #     if len(data) == 0:
# # # # # # #         return

# # # # # # #     df = pd.DataFrame({
# # # # # # #         "Frame": list(range(1, len(data)+1)),
# # # # # # #         "Count": data
# # # # # # #     })

# # # # # # #     st.line_chart(df.set_index("Frame"))


# # # # # # # # ----------------------------
# # # # # # # # PDF REPORT
# # # # # # # # ----------------------------
# # # # # # # def generate_pdf(image, heatmap, history, count):

# # # # # # #     cv2.imwrite("det.jpg", image)
# # # # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # # # #     plt.figure()
# # # # # # #     plt.plot(history)
# # # # # # #     plt.title("Crowd Trend")
# # # # # # #     plt.savefig("graph.png")
# # # # # # #     plt.close()

# # # # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # # # #     styles = getSampleStyleSheet()
# # # # # # #     story = []

# # # # # # #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# # # # # # #     story.append(Spacer(1, 10))

# # # # # # #     story.append(Paragraph(f"Total Count: {count}", styles["Normal"]))
# # # # # # #     story.append(Spacer(1, 10))

# # # # # # #     story.append(Paragraph("Detection Output", styles["Heading2"]))
# # # # # # #     story.append(RLImage("det.jpg", width=400, height=250))

# # # # # # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # # # # # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # # # # # #     story.append(Paragraph("Trend Graph", styles["Heading2"]))
# # # # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # # # #     pdf.build(story)

# # # # # # #     with open("report.pdf", "rb") as f:
# # # # # # #         return f.read()


# # # # # # # # ----------------------------
# # # # # # # # UI
# # # # # # # # ----------------------------
# # # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # # # uploaded = st.file_uploader("Upload Image / Video", type=["jpg", "png", "mp4"])

# # # # # # # col1, col2 = st.columns(2)


# # # # # # # # ----------------------------
# # # # # # # # IMAGE MODE
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Image":

# # # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # # #     if st.button("Detect"):

# # # # # # #         output, centers = detect_people(image, 640, conf, nms)
# # # # # # #         count = len(centers)

# # # # # # #         st.session_state.history.append(count)

# # # # # # #         heatmap = generate_heatmap(image.copy(), centers)

# # # # # # #         alert = get_crowd_alert(count, "Image")
# # # # # # #         st.session_state.alerts.append(alert)

# # # # # # #         col1.image(output, channels="BGR", caption=f"Count: {count}")
# # # # # # #         col2.image(heatmap, channels="BGR")

# # # # # # #         st.subheader("📊 Crowd Trend")
# # # # # # #         show_graph(st.session_state.history)

# # # # # # #         pdf = generate_pdf(output, heatmap, st.session_state.history, count)

# # # # # # #         st.download_button(
# # # # # # #             "📄 Download Report",
# # # # # # #             data=pdf,
# # # # # # #             file_name="crowd_report.pdf",
# # # # # # #             mime="application/pdf"
# # # # # # #         )


# # # # # # # # ----------------------------
# # # # # # # # VIDEO MODE
# # # # # # # # ----------------------------
# # # # # # # if uploaded and mode == "Video":

# # # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # # #     tfile.write(uploaded.read())

# # # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # # #     c1, c2 = st.columns(2)

# # # # # # #     if c1.button("▶ Start"):
# # # # # # #         st.session_state.run_video = True
# # # # # # #         st.session_state.history = []
# # # # # # #         st.session_state.alerts = []

# # # # # # #     if c2.button("⛔ Stop"):
# # # # # # #         st.session_state.run_video = False

# # # # # # #     frame_box = col1.empty()
# # # # # # #     heat_box = col2.empty()

# # # # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # # # #         ret, frame = cap.read()
# # # # # # #         if not ret:
# # # # # # #             break

# # # # # # #         output, centers = detect_people(frame, 640, conf, nms, video_mode=True)

# # # # # # #         count = len(centers)
# # # # # # #         st.session_state.history.append(count)

# # # # # # #         heatmap = generate_heatmap(frame.copy(), centers)

# # # # # # #         alert = get_crowd_alert(count, "Video")
# # # # # # #         st.session_state.alerts.append(alert)

# # # # # # #         frame_box.image(output, channels="BGR")
# # # # # # #         heat_box.image(heatmap, channels="BGR")

# # # # # # #         st.session_state.final_frame = output
# # # # # # #         st.session_state.final_heatmap = heatmap

# # # # # # #     cap.release()

# # # # # # #     # FINAL OUTPUT AFTER STOP
# # # # # # #     if not st.session_state.run_video and st.session_state.final_frame is not None:

# # # # # # #         st.subheader("📌 Final Output")

# # # # # # #         col1.image(st.session_state.final_frame, channels="BGR")
# # # # # # #         col2.image(st.session_state.final_heatmap, channels="BGR")


# # # # # # # # ----------------------------
# # # # # # # # ALERT TABLE (TOP 5 FOR VIDEO)
# # # # # # # # ----------------------------
# # # # # # # st.subheader("🚨 Alert Table")

# # # # # # # if len(st.session_state.alerts) > 0:

# # # # # # #     df = pd.DataFrame(st.session_state.alerts)

# # # # # # #     if mode == "Video":
# # # # # # #         df = df.sort_values(by="Count", ascending=False).head(5)

# # # # # # #     st.dataframe(df, use_container_width=True)


# # # # # # # # ----------------------------
# # # # # # # # GRAPH
# # # # # # # # ----------------------------
# # # # # # # st.subheader("📈 Crowd Trend")
# # # # # # # show_graph(st.session_state.history)


# # # # # # # # ----------------------------
# # # # # # # # PDF (BOTTOM FOR BOTH MODES)
# # # # # # # # ----------------------------
# # # # # # # if st.session_state.final_frame is not None:

# # # # # # #     pdf = generate_pdf(
# # # # # # #         st.session_state.final_frame,
# # # # # # #         st.session_state.final_heatmap,
# # # # # # #         st.session_state.history,
# # # # # # #         st.session_state.history[-1]
# # # # # # #     )

# # # # # # #     st.download_button(
# # # # # # #         "📄 Download Final Report",
# # # # # # #         data=pdf,
# # # # # # #         file_name="crowd_report.pdf",
# # # # # # #         mime="application/pdf"
# # # # # # #     )








# # # # # # import streamlit as st
# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import imutils
# # # # # # import pandas as pd
# # # # # # import tempfile
# # # # # # from config import YOLO_CONFIG
# # # # # # from datetime import datetime
# # # # # # import matplotlib.pyplot as plt

# # # # # # from reportlab.lib.pagesizes import A4
# # # # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # # # from reportlab.lib.styles import getSampleStyleSheet

# # # # # # # ----------------------------
# # # # # # # PAGE CONFIG
# # # # # # # ----------------------------
# # # # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # # # st.title("🚶 Crowd Counting System")

# # # # # # # ----------------------------
# # # # # # # SESSION STATE
# # # # # # # ----------------------------
# # # # # # if "history" not in st.session_state:
# # # # # #     st.session_state.history = []

# # # # # # if "alerts" not in st.session_state:
# # # # # #     st.session_state.alerts = []

# # # # # # if "run_video" not in st.session_state:
# # # # # #     st.session_state.run_video = False

# # # # # # if "final_frame" not in st.session_state:
# # # # # #     st.session_state.final_frame = None

# # # # # # if "final_heatmap" not in st.session_state:
# # # # # #     st.session_state.final_heatmap = None


# # # # # # # ----------------------------
# # # # # # # LOAD MODEL
# # # # # # # ----------------------------
# # # # # # @st.cache_resource
# # # # # # def load_model():
# # # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # # #     )
# # # # # #     ln = net.getLayerNames()
# # # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # # #     return net, ln

# # # # # # net, ln = load_model()


# # # # # # # ----------------------------
# # # # # # # ALERT SYSTEM
# # # # # # # ----------------------------
# # # # # # def get_crowd_alert(count, mode="Video"):
# # # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # # #     if mode == "Image":
# # # # # #         if count <= 20:
# # # # # #             status = "Safe"
# # # # # #             message = "Crowd level is under control."
# # # # # #         elif count <= 40:
# # # # # #             status = "Crowded"
# # # # # #             message = "Crowded area detected."
# # # # # #         else:
# # # # # #             status = "Highly Crowded"
# # # # # #             message = "High crowd density detected."
# # # # # #     else:
# # # # # #         if count <= 20:
# # # # # #             status = "Safe"
# # # # # #             message = "Crowd level is under control."
# # # # # #         elif count <= 40:
# # # # # #             status = "Moderate Crowd"
# # # # # #             message = "Moderate crowd detected."
# # # # # #         elif count <= 60:
# # # # # #             status = "High Crowd Alert"
# # # # # #             message = "High crowd density detected."
# # # # # #         else:
# # # # # #             status = "Danger"
# # # # # #             message = "Critical crowd level detected."

# # # # # #     return {
# # # # # #         "Time": time_now,
# # # # # #         "Count": count,
# # # # # #         "Status": status,
# # # # # #         "Message": message
# # # # # #     }


# # # # # # # ----------------------------
# # # # # # # ZONE ANALYSIS (NEW)
# # # # # # # ----------------------------
# # # # # # def zone_analysis(width, centers):
# # # # # #     left = center = right = 0

# # # # # #     for (x, y) in centers:
# # # # # #         if x < width / 3:
# # # # # #             left += 1
# # # # # #         elif x < 2 * width / 3:
# # # # # #             center += 1
# # # # # #         else:
# # # # # #             right += 1

# # # # # #     return left, center, right


# # # # # # # ----------------------------
# # # # # # # CROWD TREND PREDICTION (NEW)
# # # # # # # ----------------------------
# # # # # # def predict_trend(data, steps=5):
# # # # # #     if len(data) < 3:
# # # # # #         return []

# # # # # #     preds = []
# # # # # #     temp = data.copy()

# # # # # #     for _ in range(steps):
# # # # # #         nxt = int(sum(temp[-3:]) / 3)
# # # # # #         preds.append(nxt)
# # # # # #         temp.append(nxt)

# # # # # #     return preds


# # # # # # # ----------------------------
# # # # # # # DETECTION
# # # # # # # ----------------------------
# # # # # # def detect_people(image, input_size, conf, nms, video_mode=False):
# # # # # #     image = imutils.resize(image, width=800)
# # # # # #     H, W = image.shape[:2]

# # # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # # #                                  swapRB=True, crop=False)

# # # # # #     net.setInput(blob)
# # # # # #     outputs = net.forward(ln)

# # # # # #     boxes, centers, confidences = [], [], []

# # # # # #     for output in outputs:
# # # # # #         for det in output:
# # # # # #             scores = det[5:]
# # # # # #             classID = np.argmax(scores)
# # # # # #             confidence = scores[classID]

# # # # # #             if classID == 0 and confidence > conf:
# # # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # # #                 cx, cy, w, h = box.astype("int")

# # # # # #                 x = int(cx - w / 2)
# # # # # #                 y = int(cy - h / 2)

# # # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # # #                 centers.append((cx, cy))
# # # # # #                 confidences.append(float(confidence))

# # # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # # #     final_centers = []

# # # # # #     if len(idxs) > 0:
# # # # # #         for i in idxs.flatten():
# # # # # #             final_centers.append(centers[i])

# # # # # #             cx, cy = centers[i]

# # # # # #             if video_mode:
# # # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # # #             else:
# # # # # #                 x, y, w, h = boxes[i]
# # # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # # #     return image, final_centers


# # # # # # # ----------------------------
# # # # # # # HEATMAP
# # # # # # # ----------------------------
# # # # # # def generate_heatmap(image, centers):
# # # # # #     h, w = image.shape[:2]
# # # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # # #     for x, y in centers:
# # # # # #         x = max(0, min(x, w - 1))
# # # # # #         y = max(0, min(y, h - 1))
# # # # # #         heatmap[y, x] += 1

# # # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # # #     heatmap = np.uint8(heatmap)
# # # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # # #     return heatmap


# # # # # # # ----------------------------
# # # # # # # GRAPH
# # # # # # # ----------------------------
# # # # # # def show_graph(data):
# # # # # #     if len(data) == 0:
# # # # # #         return

# # # # # #     df = pd.DataFrame({
# # # # # #         "Frame": list(range(1, len(data)+1)),
# # # # # #         "Count": data
# # # # # #     })

# # # # # #     st.line_chart(df.set_index("Frame"))


# # # # # # # ----------------------------
# # # # # # # PDF REPORT
# # # # # # # ----------------------------
# # # # # # def generate_pdf(image, heatmap, history, count, zone_data, prediction):

# # # # # #     cv2.imwrite("det.jpg", image)
# # # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # # #     plt.figure()
# # # # # #     plt.plot(history)
# # # # # #     plt.title("Crowd Trend")
# # # # # #     plt.savefig("graph.png")
# # # # # #     plt.close()

# # # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # # #     styles = getSampleStyleSheet()
# # # # # #     story = []

# # # # # #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# # # # # #     story.append(Spacer(1, 10))

# # # # # #     story.append(Paragraph(f"Total Count: {count}", styles["Normal"]))
# # # # # #     story.append(Paragraph(f"Zone (L/C/R): {zone_data}", styles["Normal"]))
# # # # # #     story.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
# # # # # #     story.append(Spacer(1, 10))

# # # # # #     story.append(Paragraph("Detection Output", styles["Heading2"]))
# # # # # #     story.append(RLImage("det.jpg", width=400, height=250))

# # # # # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # # # # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # # # # #     story.append(Paragraph("Trend Graph", styles["Heading2"]))
# # # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # # #     pdf.build(story)

# # # # # #     with open("report.pdf", "rb") as f:
# # # # # #         return f.read()


# # # # # # # ----------------------------
# # # # # # # UI
# # # # # # # ----------------------------
# # # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # # uploaded = st.file_uploader("Upload Image / Video", type=["jpg", "png", "mp4"])

# # # # # # col1, col2 = st.columns(2)


# # # # # # # ----------------------------
# # # # # # # IMAGE MODE
# # # # # # # ----------------------------
# # # # # # if uploaded and mode == "Image":

# # # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # # #     if st.button("Detect"):

# # # # # #         output, centers = detect_people(image, 640, conf, nms)

# # # # # #         count = len(centers)
# # # # # #         st.session_state.history.append(count)

# # # # # #         heatmap = generate_heatmap(image.copy(), centers)

# # # # # #         # ZONE + PREDICTION
# # # # # #         zone = zone_analysis(output.shape[1], centers)
# # # # # #         prediction = predict_trend(st.session_state.history)

# # # # # #         alert = get_crowd_alert(count, "Image")
# # # # # #         st.session_state.alerts.append(alert)

# # # # # #         col1.image(output, channels="BGR", caption=f"Count: {count}")
# # # # # #         col2.image(heatmap, channels="BGR")

# # # # # #         st.subheader("📊 Trend")
# # # # # #         show_graph(st.session_state.history)

# # # # # #         pdf = generate_pdf(output, heatmap, st.session_state.history,
# # # # # #                            count, zone, prediction)

# # # # # #         st.download_button("📄 Download Report", pdf, "report.pdf")


# # # # # # # ----------------------------
# # # # # # # VIDEO MODE
# # # # # # # ----------------------------
# # # # # # if uploaded and mode == "Video":

# # # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # # #     tfile.write(uploaded.read())

# # # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # # #     c1, c2 = st.columns(2)

# # # # # #     if c1.button("▶ Start"):
# # # # # #         st.session_state.run_video = True
# # # # # #         st.session_state.history = []
# # # # # #         st.session_state.alerts = []

# # # # # #     if c2.button("⛔ Stop"):
# # # # # #         st.session_state.run_video = False

# # # # # #     frame_box = col1.empty()
# # # # # #     heat_box = col2.empty()

# # # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # # #         ret, frame = cap.read()
# # # # # #         if not ret:
# # # # # #             break

# # # # # #         output, centers = detect_people(frame, 640, conf, nms, video_mode=True)

# # # # # #         count = len(centers)
# # # # # #         st.session_state.history.append(count)

# # # # # #         heatmap = generate_heatmap(frame.copy(), centers)

# # # # # #         alert = get_crowd_alert(count, "Video")
# # # # # #         st.session_state.alerts.append(alert)

# # # # # #         frame_box.image(output, channels="BGR")
# # # # # #         heat_box.image(heatmap, channels="BGR")

# # # # # #         st.session_state.final_frame = output
# # # # # #         st.session_state.final_heatmap = heatmap

# # # # # #     cap.release()


# # # # # # # ----------------------------
# # # # # # # FINAL REPORT (BOTTOM)
# # # # # # # ----------------------------
# # # # # # if st.session_state.final_frame is not None:

# # # # # #     zone = zone_analysis(
# # # # # #         st.session_state.final_frame.shape[1],
# # # # # #         centers if 'centers' in locals() else []
# # # # # #     )

# # # # # #     prediction = predict_trend(st.session_state.history)

# # # # # #     pdf = generate_pdf(
# # # # # #         st.session_state.final_frame,
# # # # # #         st.session_state.final_heatmap,
# # # # # #         st.session_state.history,
# # # # # #         st.session_state.history[-1],
# # # # # #         zone,
# # # # # #         prediction
# # # # # #     )

# # # # # #     st.download_button("📄 Final Report", pdf, "crowd_report.pdf")



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
# # # # # if "history" not in st.session_state:
# # # # #     st.session_state.history = []

# # # # # if "alerts" not in st.session_state:
# # # # #     st.session_state.alerts = []

# # # # # if "run_video" not in st.session_state:
# # # # #     st.session_state.run_video = False

# # # # # if "final_frame" not in st.session_state:
# # # # #     st.session_state.final_frame = None

# # # # # if "final_heatmap" not in st.session_state:
# # # # #     st.session_state.final_heatmap = None


# # # # # # ----------------------------
# # # # # # LOAD MODEL
# # # # # # ----------------------------
# # # # # @st.cache_resource
# # # # # def load_model():
# # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # #     )
# # # # #     ln = net.getLayerNames()
# # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # #     return net, ln

# # # # # net, ln = load_model()


# # # # # # ----------------------------
# # # # # # ALERT SYSTEM
# # # # # # ----------------------------
# # # # # def get_crowd_alert(count, mode="Video"):
# # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # #     if count <= 20:
# # # # #         status = "Safe"
# # # # #         message = "Crowd level is under control."
# # # # #     elif count <= 40:
# # # # #         status = "Moderate"
# # # # #         message = "Crowded area detected."
# # # # #     elif count <= 60:
# # # # #         status = "High"
# # # # #         message = "High crowd density detected."
# # # # #     else:
# # # # #         status = "Danger"
# # # # #         message = "Critical crowd level!"

# # # # #     return {
# # # # #         "Time": time_now,
# # # # #         "Count": count,
# # # # #         "Status": status,
# # # # #         "Message": message
# # # # #     }


# # # # # # ----------------------------
# # # # # # ZONE ANALYSIS
# # # # # # ----------------------------
# # # # # def zone_analysis(width, centers):
# # # # #     left = center = right = 0

# # # # #     for (x, y) in centers:
# # # # #         if x < width / 3:
# # # # #             left += 1
# # # # #         elif x < 2 * width / 3:
# # # # #             center += 1
# # # # #         else:
# # # # #             right += 1

# # # # #     return left, center, right


# # # # # # ----------------------------
# # # # # # PREDICTION
# # # # # # ----------------------------
# # # # # def predict_trend(data, steps=5):
# # # # #     if len(data) < 3:
# # # # #         return []

# # # # #     preds = []
# # # # #     temp = data.copy()

# # # # #     for _ in range(steps):
# # # # #         nxt = int(sum(temp[-3:]) / 3)
# # # # #         preds.append(nxt)
# # # # #         temp.append(nxt)

# # # # #     return preds


# # # # # # ----------------------------
# # # # # # DETECTION
# # # # # # ----------------------------
# # # # # def detect_people(image, input_size, conf, nms, video_mode=False):
# # # # #     image = imutils.resize(image, width=800)
# # # # #     H, W = image.shape[:2]

# # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # #                                  swapRB=True, crop=False)

# # # # #     net.setInput(blob)
# # # # #     outputs = net.forward(ln)

# # # # #     boxes, centers, confidences = [], [], []

# # # # #     for output in outputs:
# # # # #         for det in output:
# # # # #             scores = det[5:]
# # # # #             classID = np.argmax(scores)
# # # # #             confidence = scores[classID]

# # # # #             if classID == 0 and confidence > conf:
# # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # #                 cx, cy, w, h = box.astype("int")

# # # # #                 x = int(cx - w / 2)
# # # # #                 y = int(cy - h / 2)

# # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # #                 centers.append((cx, cy))
# # # # #                 confidences.append(float(confidence))

# # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # #     final_centers = []

# # # # #     if len(idxs) > 0:
# # # # #         for i in idxs.flatten():
# # # # #             final_centers.append(centers[i])

# # # # #             cx, cy = centers[i]

# # # # #             if video_mode:
# # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # #             else:
# # # # #                 x, y, w, h = boxes[i]
# # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # #     return image, final_centers


# # # # # # ----------------------------
# # # # # # HEATMAP
# # # # # # ----------------------------
# # # # # def generate_heatmap(image, centers):
# # # # #     h, w = image.shape[:2]
# # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # #     for x, y in centers:
# # # # #         x = max(0, min(x, w - 1))
# # # # #         y = max(0, min(y, h - 1))
# # # # #         heatmap[y, x] += 1

# # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # #     heatmap = np.uint8(heatmap)
# # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # #     return heatmap


# # # # # # ----------------------------
# # # # # # GRAPH
# # # # # # ----------------------------
# # # # # def show_graph(data):
# # # # #     if len(data) == 0:
# # # # #         return

# # # # #     df = pd.DataFrame({
# # # # #         "Frame": list(range(1, len(data)+1)),
# # # # #         "Count": data
# # # # #     })

# # # # #     st.line_chart(df.set_index("Frame"))


# # # # # # ----------------------------
# # # # # # PDF
# # # # # # ----------------------------
# # # # # def generate_pdf(image, heatmap, history, count, zone, prediction):

# # # # #     cv2.imwrite("det.jpg", image)
# # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # #     plt.figure()
# # # # #     plt.plot(history)
# # # # #     plt.title("Crowd Trend")
# # # # #     plt.savefig("graph.png")
# # # # #     plt.close()

# # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # #     styles = getSampleStyleSheet()
# # # # #     story = []

# # # # #     story.append(Paragraph("Crowd Report", styles["Title"]))
# # # # #     story.append(Spacer(1, 10))

# # # # #     story.append(Paragraph(f"Count: {count}", styles["Normal"]))
# # # # #     story.append(Paragraph(f"Zone L/C/R: {zone}", styles["Normal"]))
# # # # #     story.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
# # # # #     story.append(Spacer(1, 10))

# # # # #     story.append(Paragraph("Detection", styles["Heading2"]))
# # # # #     story.append(RLImage("det.jpg", width=400, height=250))

# # # # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # # # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # # # #     story.append(Paragraph("Graph", styles["Heading2"]))
# # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # #     pdf.build(story)

# # # # #     with open("report.pdf", "rb") as f:
# # # # #         return f.read()


# # # # # # ----------------------------
# # # # # # UI
# # # # # # ----------------------------
# # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # uploaded = st.file_uploader("Upload", type=["jpg", "png", "mp4"])

# # # # # col1, col2 = st.columns(2)


# # # # # # ----------------------------
# # # # # # IMAGE MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Image":

# # # # #     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
# # # # #     image = cv2.imdecode(file_bytes, 1)

# # # # #     if st.button("Detect"):

# # # # #         output, centers = detect_people(image, 640, conf, nms)

# # # # #         count = len(centers)
# # # # #         st.session_state.history.append(count)

# # # # #         heatmap = generate_heatmap(image.copy(), centers)

# # # # #         zone = zone_analysis(output.shape[1], centers)
# # # # #         prediction = predict_trend(st.session_state.history)

# # # # #         alert = get_crowd_alert(count)
# # # # #         st.session_state.alerts.append(alert)

# # # # #         col1.image(output, channels="BGR")
# # # # #         col2.image(heatmap, channels="BGR")

# # # # #         show_graph(st.session_state.history)

# # # # #         pdf = generate_pdf(output, heatmap, st.session_state.history,
# # # # #                            count, zone, prediction)

# # # # #         st.download_button("Download Report", pdf, "report.pdf")


# # # # # # ----------------------------
# # # # # # VIDEO MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Video":

# # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # #     tfile.write(uploaded.read())

# # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # #     c1, c2 = st.columns(2)

# # # # #     if c1.button("Start"):
# # # # #         st.session_state.run_video = True
# # # # #         st.session_state.history = []
# # # # #         st.session_state.alerts = []

# # # # #     if c2.button("Stop"):
# # # # #         st.session_state.run_video = False

# # # # #     frame_box = col1.empty()
# # # # #     heat_box = col2.empty()

# # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break

# # # # #         output, centers = detect_people(frame, 640, conf, nms, video_mode=True)

# # # # #         count = len(centers)
# # # # #         st.session_state.history.append(count)

# # # # #         heatmap = generate_heatmap(frame.copy(), centers)

# # # # #         alert = get_crowd_alert(count)
# # # # #         st.session_state.alerts.append(alert)

# # # # #         frame_box.image(output, channels="BGR")
# # # # #         heat_box.image(heatmap, channels="BGR")

# # # # #         st.session_state.final_frame = output
# # # # #         st.session_state.final_heatmap = heatmap

# # # # #     cap.release()


# # # # # # ----------------------------
# # # # # # FINAL OUTPUT
# # # # # # ----------------------------
# # # # # if st.session_state.final_frame is not None:

# # # # #     st.subheader("Final Output")
# # # # #     col1.image(st.session_state.final_frame, channels="BGR")
# # # # #     col2.image(st.session_state.final_heatmap, channels="BGR")


# # # # # # ----------------------------
# # # # # # ALERT TABLE (TOP 5 VIDEO)
# # # # # # ----------------------------
# # # # # st.subheader("Alert Table")

# # # # # if len(st.session_state.alerts) > 0:
# # # # #     df = pd.DataFrame(st.session_state.alerts)

# # # # #     if mode == "Video":
# # # # #         df = df.sort_values(by="Count", ascending=False).head(5)

# # # # #     st.dataframe(df)


# # # # # # ----------------------------
# # # # # # GRAPH
# # # # # # ----------------------------
# # # # # st.subheader("Crowd Trend")
# # # # # show_graph(st.session_state.history)


# # # # # # ----------------------------
# # # # # # PDF FINAL
# # # # # # ----------------------------
# # # # # if st.session_state.final_frame is not None:

# # # # #     zone = zone_analysis(st.session_state.final_frame.shape[1], [])
# # # # #     prediction = predict_trend(st.session_state.history)

# # # # #     pdf = generate_pdf(
# # # # #         st.session_state.final_frame,
# # # # #         st.session_state.final_heatmap,
# # # # #         st.session_state.history,
# # # # #         st.session_state.history[-1],
# # # # #         zone,
# # # # #         prediction
# # # # #     )

# # # # #     st.download_button("Final Report", pdf, "crowd_report.pdf")













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
# # # # # if "history" not in st.session_state:
# # # # #     st.session_state.history = []

# # # # # if "alerts" not in st.session_state:
# # # # #     st.session_state.alerts = []

# # # # # if "run_video" not in st.session_state:
# # # # #     st.session_state.run_video = False

# # # # # if "final_frame" not in st.session_state:
# # # # #     st.session_state.final_frame = None

# # # # # if "final_heatmap" not in st.session_state:
# # # # #     st.session_state.final_heatmap = None


# # # # # # ----------------------------
# # # # # # LOAD MODEL
# # # # # # ----------------------------
# # # # # @st.cache_resource
# # # # # def load_model():
# # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # #     )
# # # # #     ln = net.getLayerNames()
# # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # #     return net, ln

# # # # # net, ln = load_model()


# # # # # # ----------------------------
# # # # # # ALERT SYSTEM
# # # # # # ----------------------------
# # # # # def get_crowd_alert(count):
# # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # #     if count <= 20:
# # # # #         status = "Safe"
# # # # #     elif count <= 40:
# # # # #         status = "Crowded"
# # # # #     elif count <= 60:
# # # # #         status = "High Crowd"
# # # # #     else:
# # # # #         status = "Danger"

# # # # #     return {
# # # # #         "Time": time_now,
# # # # #         "Count": count,
# # # # #         "Status": status
# # # # #     }


# # # # # # ----------------------------
# # # # # # DETECTION
# # # # # # ----------------------------
# # # # # def detect_people(image, input_size, conf, nms, video_mode=False):

# # # # #     image = imutils.resize(image, width=800)
# # # # #     H, W = image.shape[:2]

# # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # #                                  swapRB=True, crop=False)

# # # # #     net.setInput(blob)
# # # # #     outputs = net.forward(ln)

# # # # #     boxes, centers, confidences = [], [], []

# # # # #     for output in outputs:
# # # # #         for det in output:
# # # # #             scores = det[5:]
# # # # #             classID = np.argmax(scores)
# # # # #             confidence = scores[classID]

# # # # #             if classID == 0 and confidence > conf:

# # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # #                 cx, cy, w, h = box.astype("int")

# # # # #                 x = int(cx - w / 2)
# # # # #                 y = int(cy - h / 2)

# # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # #                 centers.append((cx, cy))
# # # # #                 confidences.append(float(confidence))

# # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # #     final_centers = []

# # # # #     if len(idxs) > 0:
# # # # #         for i in idxs.flatten():
# # # # #             final_centers.append(centers[i])

# # # # #             cx, cy = centers[i]

# # # # #             if video_mode:
# # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # #             else:
# # # # #                 x, y, w, h = boxes[i]
# # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # #     return image, final_centers


# # # # # # ----------------------------
# # # # # # HEATMAP
# # # # # # ----------------------------
# # # # # def generate_heatmap(image, centers):

# # # # #     h, w = image.shape[:2]
# # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # #     for x, y in centers:
# # # # #         if 0 <= x < w and 0 <= y < h:
# # # # #             heatmap[y, x] += 1

# # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # #     heatmap = np.uint8(heatmap)
# # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # #     return heatmap


# # # # # # ----------------------------
# # # # # # ZONE ANALYSIS (FIXED)
# # # # # # ----------------------------
# # # # # def zone_analysis(width, centers):

# # # # #     left = center = right = 0

# # # # #     for x, y in centers:
# # # # #         if x < width / 3:
# # # # #             left += 1
# # # # #         elif x < 2 * width / 3:
# # # # #             center += 1
# # # # #         else:
# # # # #             right += 1

# # # # #     return left, center, right


# # # # # # ----------------------------
# # # # # # TREND PREDICTION
# # # # # # ----------------------------
# # # # # def predict_trend(data, steps=5):

# # # # #     if len(data) < 3:
# # # # #         return []

# # # # #     preds = []
# # # # #     temp = data.copy()

# # # # #     for _ in range(steps):
# # # # #         val = int(sum(temp[-3:]) / 3)
# # # # #         preds.append(val)
# # # # #         temp.append(val)

# # # # #     return preds


# # # # # # ----------------------------
# # # # # # GRAPH
# # # # # # ----------------------------
# # # # # def show_graph(data):

# # # # #     if len(data) == 0:
# # # # #         return

# # # # #     df = pd.DataFrame({
# # # # #         "Frame": list(range(1, len(data)+1)),
# # # # #         "Count": data
# # # # #     })

# # # # #     st.line_chart(df.set_index("Frame"))


# # # # # # ----------------------------
# # # # # # PDF
# # # # # # ----------------------------
# # # # # def generate_pdf(image, heatmap, history, count):

# # # # #     cv2.imwrite("det.jpg", image)
# # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # #     plt.figure()
# # # # #     plt.plot(history)
# # # # #     plt.title("Crowd Trend")
# # # # #     plt.savefig("graph.png")
# # # # #     plt.close()

# # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # #     styles = getSampleStyleSheet()
# # # # #     story = []

# # # # #     story.append(Paragraph("Crowd Report", styles["Title"]))
# # # # #     story.append(Spacer(1, 10))

# # # # #     story.append(Paragraph(f"Final Count: {count}", styles["Normal"]))

# # # # #     story.append(Spacer(1, 10))
# # # # #     story.append(Paragraph("Detection", styles["Heading2"]))
# # # # #     story.append(RLImage("det.jpg", width=400, height=250))

# # # # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # # # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # # # #     story.append(Paragraph("Trend", styles["Heading2"]))
# # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # #     pdf.build(story)

# # # # #     with open("report.pdf", "rb") as f:
# # # # #         return f.read()


# # # # # # ----------------------------
# # # # # # UI
# # # # # # ----------------------------
# # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # uploaded = st.file_uploader("Upload", type=["jpg", "png", "mp4"])

# # # # # col1, col2 = st.columns(2)


# # # # # # ----------------------------
# # # # # # IMAGE
# # # # # # ----------------------------
# # # # # # if uploaded and mode == "Image":

# # # # # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # # # # #     if st.button("Detect"):

# # # # # #         out, centers = detect_people(img, 640, conf, nms)
# # # # # #         count = len(centers)

# # # # # #         st.session_state.history.append(count)

# # # # # #         heat = generate_heatmap(img.copy(), centers)

# # # # # #         left, center, right = zone_analysis(img.shape[1], centers)

# # # # # #         st.write("### Zone Analysis")
# # # # # #         st.write({"Left": left, "Center": center, "Right": right})

# # # # # #         st.image(out)
# # # # # #         st.image(heat)

# # # # # #         st.write("### Trend")
# # # # # #         show_graph(st.session_state.history)

# # # # # #         pdf = generate_pdf(out, heat, st.session_state.history, count)

# # # # # #         st.download_button("Download PDF", pdf, "report.pdf")

# # # # # if uploaded and mode == "Image":

# # # # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # # # #     if st.button("Detect"):

# # # # #         out, centers = detect_people(img, 640, conf, nms)
# # # # #         count = len(centers)

# # # # #         st.session_state.history.append(count)

# # # # #         heat = generate_heatmap(img.copy(), centers)

# # # # #         left, center, right = zone_analysis(img.shape[1], centers)

# # # # #         # ----------------------------
# # # # #         # ZONE ANALYSIS CARD
# # # # #         # ----------------------------
# # # # #         st.markdown("### 📍 Zone-wise Crowd Analysis")

# # # # #         st.subheader("📍 Zone-wise Crowd Analysis")

# # # # #         st.write(f"Left = {left}, Center = {center}, Right = {right}")
        

# # # # #         # ----------------------------
# # # # #         # IMAGE + HEATMAP SIDE BY SIDE
# # # # #         # ----------------------------
# # # # #         col1, col2 = st.columns(2)

# # # # #         with col1:
# # # # #             st.markdown("### 📸 Detection Output")
# # # # #             st.image(out, channels="BGR", use_container_width=True)
# # # # #             st.markdown(f"**👥 Crowd Count:** {count}")

# # # # #         with col2:
# # # # #             st.markdown("### 🔥 Heatmap")
# # # # #             st.image(heat, channels="BGR", use_container_width=True)


# # # # #         # ----------------------------
# # # # #         # TREND GRAPH
# # # # #         # ----------------------------
# # # # #         st.markdown("### 📊 Crowd Trend")
# # # # #         show_graph(st.session_state.history)

# # # # #         # ----------------------------
# # # # #         # PDF REPORT
# # # # #         # ----------------------------
# # # # #         pdf = generate_pdf(out, heat, st.session_state.history, count)

# # # # #         st.download_button(
# # # # #             "📄 Download PDF Report",
# # # # #             pdf,
# # # # #             "crowd_report.pdf",
# # # # #             mime="application/pdf"
# # # # #         )


# # # # # # ----------------------------
# # # # # # VIDEO
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Video":

# # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # #     tfile.write(uploaded.read())

# # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # #     c1, c2 = st.columns(2)

# # # # #     if c1.button("▶ Start"):
# # # # #         st.session_state.run_video = True
# # # # #         st.session_state.history = []
# # # # #         st.session_state.alerts = []

# # # # #     if c2.button("⛔ Stop"):
# # # # #         st.session_state.run_video = False

# # # # #     frame_box = col1.empty()
# # # # #     heat_box = col2.empty()

# # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break

# # # # #         out, centers = detect_people(frame, 640, conf, nms, video_mode=True)
# # # # #         count = len(centers)

# # # # #         st.session_state.history.append(count)
# # # # #         st.session_state.alerts.append(get_crowd_alert(count))

# # # # #         heat = generate_heatmap(frame.copy(), centers)

# # # # #         frame_box.image(out)
# # # # #         heat_box.image(heat)

# # # # #         st.session_state.final_frame = out
# # # # #         st.session_state.final_heatmap = heat

# # # # #     cap.release()

# # # # #     if not st.session_state.run_video and st.session_state.final_frame is not None:

# # # # #         st.image(st.session_state.final_frame)
# # # # #         st.image(st.session_state.final_heatmap)

# # # # #         st.write("### Trend Prediction")
# # # # #         preds = predict_trend(st.session_state.history)
# # # # #         st.write(preds)


# # # # # # ----------------------------
# # # # # # ALERT TABLE (TOP 5 VIDEO)
# # # # # # ----------------------------
# # # # # # st.subheader("Alert Table")

# # # # # # if len(st.session_state.alerts) > 0:

# # # # # #     df = pd.DataFrame(st.session_state.alerts)

# # # # # #     if mode == "Video":
# # # # # #         df = df.sort_values("Count", ascending=False).head(5)

# # # # # #     st.dataframe(df)


# # # # # def get_crowd_alert(count, mode="Image"):
# # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # #     if count <= 20:
# # # # #         status = "Safe"
# # # # #         message = "Crowd is under control"
# # # # #     elif count <= 40:
# # # # #         status = "Crowded"
# # # # #         message = "Moderate crowd detected"
# # # # #     elif count <= 60:
# # # # #         status = "High Crowd"
# # # # #         message = "Crowd density is high"
# # # # #     else:
# # # # #         status = "Danger"
# # # # #         message = "Critical crowd level!"

# # # # #     return {
# # # # #         "Time": time_now,
# # # # #         "Count": count,
# # # # #         "Status": status,
# # # # #         "Message": message   # ✅ IMPORTANT FIX
# # # # #     }


# # # # # # ----------------------------
# # # # # # FINAL GRAPH
# # # # # # ----------------------------
# # # # # st.subheader("Trend")
# # # # # show_graph(st.session_state.history)


# # # # # # ----------------------------
# # # # # # FINAL PDF
# # # # # # ----------------------------
# # # # # if st.session_state.final_frame is not None:

# # # # #     pdf = generate_pdf(
# # # # #         st.session_state.final_frame,
# # # # #         st.session_state.final_heatmap,
# # # # #         st.session_state.history,
# # # # #         st.session_state.history[-1]
# # # # #     )

# # # # #     st.download_button("Final Report PDF", pdf, "final_report.pdf")


















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
# # # # # if "history" not in st.session_state:
# # # # #     st.session_state.history = []

# # # # # if "alerts" not in st.session_state:
# # # # #     st.session_state.alerts = []

# # # # # if "run_video" not in st.session_state:
# # # # #     st.session_state.run_video = False

# # # # # if "final_frame" not in st.session_state:
# # # # #     st.session_state.final_frame = None

# # # # # if "final_heatmap" not in st.session_state:
# # # # #     st.session_state.final_heatmap = None


# # # # # # ----------------------------
# # # # # # LOAD MODEL
# # # # # # ----------------------------
# # # # # @st.cache_resource
# # # # # def load_model():
# # # # #     net = cv2.dnn.readNetFromDarknet(
# # # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # # #     )
# # # # #     ln = net.getLayerNames()
# # # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # # #     return net, ln

# # # # # net, ln = load_model()


# # # # # # ----------------------------
# # # # # # ALERT SYSTEM
# # # # # # ----------------------------
# # # # # def get_crowd_alert(count):
# # # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # # #     if count <= 20:
# # # # #         status = "Safe"
# # # # #         message = "Crowd is under control"
# # # # #     elif count <= 40:
# # # # #         status = "Crowded"
# # # # #         message = "Moderate crowd detected"
# # # # #     elif count <= 60:
# # # # #         status = "High Crowd"
# # # # #         message = "Crowd density is high"
# # # # #     else:
# # # # #         status = "Danger"
# # # # #         message = "Critical crowd level!"

# # # # #     return {
# # # # #         "Time": time_now,
# # # # #         "Count": count,
# # # # #         "Status": status,
# # # # #         "Message": message
# # # # #     }


# # # # # # ----------------------------
# # # # # # DETECTION
# # # # # # ----------------------------
# # # # # def detect_people(image, input_size, conf, nms, video_mode=False):

# # # # #     image = imutils.resize(image, width=800)
# # # # #     H, W = image.shape[:2]

# # # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # # #                                  swapRB=True, crop=False)

# # # # #     net.setInput(blob)
# # # # #     outputs = net.forward(ln)

# # # # #     boxes, centers, confidences = [], [], []

# # # # #     for output in outputs:
# # # # #         for det in output:
# # # # #             scores = det[5:]
# # # # #             classID = np.argmax(scores)
# # # # #             confidence = scores[classID]

# # # # #             if classID == 0 and confidence > conf:

# # # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # # #                 cx, cy, w, h = box.astype("int")

# # # # #                 x = int(cx - w / 2)
# # # # #                 y = int(cy - h / 2)

# # # # #                 boxes.append([x, y, int(w), int(h)])
# # # # #                 centers.append((cx, cy))
# # # # #                 confidences.append(float(confidence))

# # # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # # #     final_centers = []

# # # # #     if len(idxs) > 0:
# # # # #         for i in idxs.flatten():
# # # # #             final_centers.append(centers[i])

# # # # #             cx, cy = centers[i]

# # # # #             if video_mode:
# # # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # # #             else:
# # # # #                 x, y, w, h = boxes[i]
# # # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # # #     return image, final_centers


# # # # # # ----------------------------
# # # # # # HEATMAP
# # # # # # ----------------------------
# # # # # def generate_heatmap(image, centers):

# # # # #     h, w = image.shape[:2]
# # # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # # #     for x, y in centers:
# # # # #         x = max(0, min(x, w - 1))
# # # # #         y = max(0, min(y, h - 1))
# # # # #         heatmap[y, x] += 1

# # # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # # #     heatmap = np.uint8(heatmap)
# # # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # # #     return heatmap


# # # # # # ----------------------------
# # # # # # ZONE ANALYSIS
# # # # # # ----------------------------
# # # # # def zone_analysis(width, centers):

# # # # #     left = center = right = 0

# # # # #     for x, y in centers:
# # # # #         if x < width / 3:
# # # # #             left += 1
# # # # #         elif x < 2 * width / 3:
# # # # #             center += 1
# # # # #         else:
# # # # #             right += 1

# # # # #     return left, center, right


# # # # # # ----------------------------
# # # # # # TREND PREDICTION
# # # # # # ----------------------------
# # # # # def predict_trend(data, steps=5):

# # # # #     if len(data) < 3:
# # # # #         return []

# # # # #     preds = []
# # # # #     temp = data.copy()

# # # # #     for _ in range(steps):
# # # # #         val = int(sum(temp[-3:]) / 3)
# # # # #         preds.append(val)
# # # # #         temp.append(val)

# # # # #     return preds


# # # # # # ----------------------------
# # # # # # GRAPH
# # # # # # ----------------------------
# # # # # def show_graph(data):

# # # # #     if len(data) == 0:
# # # # #         return

# # # # #     df = pd.DataFrame({
# # # # #         "Frame": list(range(1, len(data)+1)),
# # # # #         "Count": data
# # # # #     })

# # # # #     st.line_chart(df.set_index("Frame"))


# # # # # # ----------------------------
# # # # # # PDF
# # # # # # ----------------------------
# # # # # def generate_pdf(image, heatmap, history, count):

# # # # #     cv2.imwrite("det.jpg", image)
# # # # #     cv2.imwrite("heat.jpg", heatmap)

# # # # #     plt.figure()
# # # # #     plt.plot(history)
# # # # #     plt.title("Crowd Trend")
# # # # #     plt.savefig("graph.png")
# # # # #     plt.close()

# # # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # # #     styles = getSampleStyleSheet()
# # # # #     story = []

# # # # #     story.append(Paragraph("Crowd Report", styles["Title"]))
# # # # #     story.append(Spacer(1, 10))

# # # # #     story.append(Paragraph(f"Final Count: {count}", styles["Normal"]))
# # # # #     story.append(Spacer(1, 10))

# # # # #     story.append(Paragraph("Detection + Heatmap", styles["Heading2"]))
# # # # #     story.append(RLImage("det.jpg", width=300, height=200))
# # # # #     story.append(RLImage("heat.jpg", width=300, height=200))

# # # # #     story.append(Paragraph("Trend", styles["Heading2"]))
# # # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # # #     pdf.build(story)

# # # # #     with open("report.pdf", "rb") as f:
# # # # #         return f.read()


# # # # # # ----------------------------
# # # # # # UI
# # # # # # ----------------------------
# # # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # # uploaded = st.file_uploader("Upload Image / Video", type=["jpg", "png", "mp4"])

# # # # # col1, col2 = st.columns(2)


# # # # # # ----------------------------
# # # # # # IMAGE MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Image":

# # # # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # # # #     if st.button("Detect"):

# # # # #         out, centers = detect_people(img, 640, conf, nms)
# # # # #         count = len(centers)

# # # # #         st.session_state.history.append(count)

# # # # #         heat = generate_heatmap(img.copy(), centers)

# # # # #         left, center, right = zone_analysis(img.shape[1], centers)

# # # # #         # Zone
# # # # #         st.write(f"📍 Zone Analysis → Left = {left}, Center = {center}, Right = {right}")

# # # # #         # Images side by side
# # # # #         col1.image(out, caption=f"Count = {count}")
# # # # #         col2.image(heat)

# # # # #         # Alert
# # # # #         alert = get_crowd_alert(count)
# # # # #         st.session_state.alerts.append(alert)

# # # # #         st.dataframe(pd.DataFrame(st.session_state.alerts))

# # # # #         st.subheader("Trend")
# # # # #         show_graph(st.session_state.history)

# # # # #         pdf = generate_pdf(out, heat, st.session_state.history, count)

# # # # #         st.download_button("Download PDF", pdf, "report.pdf")


# # # # # # ----------------------------
# # # # # # VIDEO MODE
# # # # # # ----------------------------
# # # # # if uploaded and mode == "Video":

# # # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # # #     tfile.write(uploaded.read())

# # # # #     cap = cv2.VideoCapture(tfile.name)

# # # # #     c1, c2 = st.columns(2)

# # # # #     if c1.button("▶ Start"):
# # # # #         st.session_state.run_video = True
# # # # #         st.session_state.history = []
# # # # #         st.session_state.alerts = []

# # # # #     if c2.button("⛔ Stop"):
# # # # #         st.session_state.run_video = False

# # # # #     frame_box = col1.empty()
# # # # #     heat_box = col2.empty()

# # # # #     while cap.isOpened() and st.session_state.run_video:

# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break

# # # # #         out, centers = detect_people(frame, 640, conf, nms, video_mode=True)
# # # # #         count = len(centers)

# # # # #         st.session_state.history.append(count)
# # # # #         st.session_state.alerts.append(get_crowd_alert(count))

# # # # #         heat = generate_heatmap(frame.copy(), centers)

# # # # #         frame_box.image(out)
# # # # #         heat_box.image(heat)

# # # # #         st.session_state.final_frame = out
# # # # #         st.session_state.final_heatmap = heat

# # # # #     cap.release()

# # # # #     if st.session_state.final_frame is not None:

# # # # #         st.subheader("Final Output")

# # # # #         col1.image(st.session_state.final_frame)
# # # # #         col2.image(st.session_state.final_heatmap)

# # # # #         st.subheader("Trend Prediction")
# # # # #         st.write(predict_trend(st.session_state.history))


# # # # # # ----------------------------
# # # # # # ALERT TABLE
# # # # # # ----------------------------
# # # # # st.subheader("Alert Table")

# # # # # if len(st.session_state.alerts) > 0:
# # # # #     st.dataframe(pd.DataFrame(st.session_state.alerts))


# # # # # # ----------------------------
# # # # # # FINAL GRAPH
# # # # # # ----------------------------
# # # # # st.subheader("Trend")
# # # # # show_graph(st.session_state.history)


# # # # # # ----------------------------
# # # # # # FINAL PDF
# # # # # # ----------------------------
# # # # # if st.session_state.final_frame is not None:

# # # # #     pdf = generate_pdf(
# # # # #         st.session_state.final_frame,
# # # # #         st.session_state.final_heatmap,
# # # # #         st.session_state.history,
# # # # #         st.session_state.history[-1]
# # # # #     )

# # # # #     st.download_button("Final PDF", pdf, "final_report.pdf")






# # # # import streamlit as st
# # # # import cv2
# # # # import numpy as np
# # # # import imutils
# # # # import pandas as pd
# # # # import tempfile
# # # # from config import YOLO_CONFIG
# # # # from datetime import datetime
# # # # import matplotlib.pyplot as plt

# # # # from reportlab.lib.pagesizes import A4
# # # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # # from reportlab.lib.styles import getSampleStyleSheet


# # # # # ----------------------------
# # # # # PAGE CONFIG
# # # # # ----------------------------
# # # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # # st.title("🚶 Crowd Counting System")


# # # # # ----------------------------
# # # # # SESSION STATE
# # # # # ----------------------------
# # # # if "history" not in st.session_state:
# # # #     st.session_state.history = []

# # # # if "alerts" not in st.session_state:
# # # #     st.session_state.alerts = []

# # # # if "run_video" not in st.session_state:
# # # #     st.session_state.run_video = False

# # # # if "final_frame" not in st.session_state:
# # # #     st.session_state.final_frame = None

# # # # if "final_heatmap" not in st.session_state:
# # # #     st.session_state.final_heatmap = None


# # # # # ----------------------------
# # # # # LOAD MODEL
# # # # # ----------------------------
# # # # @st.cache_resource
# # # # def load_model():
# # # #     net = cv2.dnn.readNetFromDarknet(
# # # #         YOLO_CONFIG["CONFIG_PATH"],
# # # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # # #     )
# # # #     ln = net.getLayerNames()
# # # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # # #     return net, ln

# # # # net, ln = load_model()


# # # # # ----------------------------
# # # # # ALERT SYSTEM
# # # # # ----------------------------
# # # # def get_crowd_alert(count):
# # # #     time_now = datetime.now().strftime("%H:%M:%S")

# # # #     if count <= 20:
# # # #         status = "Safe"
# # # #         message = "Crowd is under control"
# # # #     elif count <= 40:
# # # #         status = "Crowded"
# # # #         message = "Moderate crowd detected"
# # # #     elif count <= 60:
# # # #         status = "High Crowd"
# # # #         message = "Crowd density is high"
# # # #     else:
# # # #         status = "Danger"
# # # #         message = "Critical crowd level!"

# # # #     return {
# # # #         "Time": time_now,
# # # #         "Count": count,
# # # #         "Status": status,
# # # #         "Message": message
# # # #     }


# # # # # ----------------------------
# # # # # DETECTION
# # # # # ----------------------------
# # # # def detect_people(image, input_size, conf, nms, video_mode=False):

# # # #     image = imutils.resize(image, width=800)
# # # #     H, W = image.shape[:2]

# # # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # # #                                  swapRB=True, crop=False)

# # # #     net.setInput(blob)
# # # #     outputs = net.forward(ln)

# # # #     boxes, centers, confidences = [], [], []

# # # #     for output in outputs:
# # # #         for det in output:
# # # #             scores = det[5:]
# # # #             classID = np.argmax(scores)
# # # #             confidence = scores[classID]

# # # #             if classID == 0 and confidence > conf:
# # # #                 box = det[0:4] * np.array([W, H, W, H])
# # # #                 cx, cy, w, h = box.astype("int")

# # # #                 x = int(cx - w / 2)
# # # #                 y = int(cy - h / 2)

# # # #                 boxes.append([x, y, int(w), int(h)])
# # # #                 centers.append((cx, cy))
# # # #                 confidences.append(float(confidence))

# # # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # # #     final_centers = []

# # # #     if len(idxs) > 0:
# # # #         for i in idxs.flatten():
# # # #             final_centers.append(centers[i])

# # # #             cx, cy = centers[i]

# # # #             if video_mode:
# # # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # # #             else:
# # # #                 x, y, w, h = boxes[i]
# # # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # # #     return image, final_centers


# # # # # ----------------------------
# # # # # HEATMAP
# # # # # ----------------------------
# # # # def generate_heatmap(image, centers):

# # # #     h, w = image.shape[:2]
# # # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # # #     for x, y in centers:
# # # #         x = max(0, min(x, w - 1))
# # # #         y = max(0, min(y, h - 1))
# # # #         heatmap[y, x] += 1

# # # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # # #     heatmap = np.uint8(heatmap)
# # # #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# # # #     return heatmap


# # # # # ----------------------------
# # # # # ZONE ANALYSIS
# # # # # ----------------------------
# # # # def zone_analysis(width, centers):

# # # #     left = center = right = 0

# # # #     for x, y in centers:
# # # #         if x < width / 3:
# # # #             left += 1
# # # #         elif x < 2 * width / 3:
# # # #             center += 1
# # # #         else:
# # # #             right += 1

# # # #     return left, center, right


# # # # # ----------------------------
# # # # # GRAPH
# # # # # ----------------------------
# # # # def show_graph(data):
# # # #     if len(data) == 0:
# # # #         return

# # # #     df = pd.DataFrame({
# # # #         "Frame": list(range(1, len(data)+1)),
# # # #         "Count": data
# # # #     })

# # # #     st.line_chart(df.set_index("Frame"))


# # # # # ----------------------------
# # # # # PDF
# # # # # ----------------------------
# # # # def generate_pdf(image, heatmap, history, count):

# # # #     cv2.imwrite("det.jpg", image)
# # # #     cv2.imwrite("heat.jpg", heatmap)

# # # #     plt.figure()
# # # #     plt.plot(history)
# # # #     plt.title("Crowd Trend")
# # # #     plt.savefig("graph.png")
# # # #     plt.close()

# # # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # # #     styles = getSampleStyleSheet()
# # # #     story = []

# # # #     story.append(Paragraph("Crowd Report", styles["Title"]))
# # # #     story.append(Spacer(1, 10))
# # # #     story.append(Paragraph(f"Final Count: {count}", styles["Normal"]))

# # # #     story.append(RLImage("det.jpg", width=300, height=200))
# # # #     story.append(RLImage("heat.jpg", width=300, height=200))
# # # #     story.append(RLImage("graph.png", width=400, height=200))

# # # #     pdf.build(story)

# # # #     with open("report.pdf", "rb") as f:
# # # #         return f.read()


# # # # # ----------------------------
# # # # # UI
# # # # # ----------------------------
# # # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # # uploaded = st.file_uploader("Upload", type=["jpg", "png", "mp4"])

# # # # col1, col2 = st.columns(2)


# # # # # ----------------------------
# # # # # IMAGE MODE
# # # # # ----------------------------
# # # # if uploaded and mode == "Image":

# # # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # # #     if st.button("Detect"):

# # # #         out, centers = detect_people(img, 640, conf, nms)
# # # #         count = len(centers)

# # # #         st.session_state.history.append(count)

# # # #         heat = generate_heatmap(img.copy(), centers)

# # # #         left, center, right = zone_analysis(img.shape[1], centers)

# # # #         st.write(f"📍 Left={left}, Center={center}, Right={right}")

# # # #         col1.image(out, caption=f"Count={count}")
# # # #         col2.image(heat)

# # # #         st.session_state.alerts.append(get_crowd_alert(count))

# # # #         show_graph(st.session_state.history)

# # # #         pdf = generate_pdf(out, heat, st.session_state.history, count)
# # # #         st.download_button("Download PDF", pdf)


# # # # # ----------------------------
# # # # # VIDEO MODE
# # # # # ----------------------------
# # # # if uploaded and mode == "Video":

# # # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # # #     tfile.write(uploaded.read())

# # # #     cap = cv2.VideoCapture(tfile.name)

# # # #     c1, c2 = st.columns(2)

# # # #     if c1.button("▶ Start"):
# # # #         st.session_state.run_video = True
# # # #         st.session_state.history = []
# # # #         st.session_state.alerts = []

# # # #     if c2.button("⛔ Stop"):
# # # #         st.session_state.run_video = False

# # # #     frame_box = col1.empty()
# # # #     heat_box = col2.empty()

# # # #     while cap.isOpened() and st.session_state.run_video:

# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break

# # # #         out, centers = detect_people(frame, 640, conf, nms, video_mode=True)
# # # #         count = len(centers)

# # # #         st.session_state.history.append(count)
# # # #         st.session_state.alerts.append(get_crowd_alert(count))

# # # #         heat = generate_heatmap(frame.copy(), centers)

# # # #         frame_box.image(out)
# # # #         heat_box.image(heat)

# # # #         st.session_state.final_frame = out
# # # #         st.session_state.final_heatmap = heat

# # # #     cap.release()


# # # # # ----------------------------
# # # # # ALERT TABLE (FINAL FIX)
# # # # # ----------------------------
# # # # st.subheader("🚨 Alert Table")

# # # # if len(st.session_state.alerts) > 0:

# # # #     df = pd.DataFrame(st.session_state.alerts)

# # # #     if mode == "Video":
# # # #         df = df.sort_values("Count", ascending=False).head(5)

# # # #     st.dataframe(df)


# # # # # ----------------------------
# # # # # FINAL GRAPH
# # # # # ----------------------------
# # # # st.subheader("📈 Crowd Trend")
# # # # show_graph(st.session_state.history)









# # # import streamlit as st
# # # import cv2
# # # import numpy as np
# # # import imutils
# # # import pandas as pd
# # # import tempfile
# # # from config import YOLO_CONFIG
# # # from datetime import datetime
# # # import matplotlib.pyplot as plt

# # # from reportlab.lib.pagesizes import A4
# # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # # from reportlab.lib.styles import getSampleStyleSheet

# # # # ----------------------------
# # # # PAGE CONFIG
# # # # ----------------------------
# # # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # # st.title("🚶 Crowd Counting System")

# # # # ----------------------------
# # # # SESSION STATE
# # # # ----------------------------
# # # for key in ["history", "alerts", "run_video", "final_frame", "final_heatmap"]:
# # #     if key not in st.session_state:
# # #         st.session_state[key] = [] if key in ["history", "alerts"] else None

# # # # ----------------------------
# # # # LOAD MODEL
# # # # ----------------------------
# # # @st.cache_resource
# # # def load_model():
# # #     net = cv2.dnn.readNetFromDarknet(
# # #         YOLO_CONFIG["CONFIG_PATH"],
# # #         YOLO_CONFIG["WEIGHTS_PATH"]
# # #     )
# # #     ln = net.getLayerNames()
# # #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# # #     return net, ln

# # # net, ln = load_model()

# # # # ----------------------------
# # # # ALERT SYSTEM
# # # # ----------------------------
# # # def get_crowd_alert(count):
# # #     time_now = datetime.now().strftime("%H:%M:%S")

# # #     if count <= 20:
# # #         status, msg = "Safe", "Crowd under control"
# # #     elif count <= 40:
# # #         status, msg = "Crowded", "Moderate crowd"
# # #     elif count <= 60:
# # #         status, msg = "High", "High density"
# # #     else:
# # #         status, msg = "Danger", "Critical crowd"

# # #     return {"Time": time_now, "Count": count, "Status": status, "Message": msg}

# # # # ----------------------------
# # # # DETECTION
# # # # ----------------------------
# # # def detect_people(image, input_size, conf, nms, video=False):

# # #     image = imutils.resize(image, width=800)
# # #     H, W = image.shape[:2]

# # #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# # #                                  swapRB=True, crop=False)

# # #     net.setInput(blob)
# # #     outputs = net.forward(ln)

# # #     boxes, centers, confidences = [], [], []

# # #     for output in outputs:
# # #         for det in output:
# # #             scores = det[5:]
# # #             classID = np.argmax(scores)
# # #             confidence = scores[classID]

# # #             if classID == 0 and confidence > conf:
# # #                 box = det[0:4] * np.array([W, H, W, H])
# # #                 cx, cy, w, h = box.astype("int")

# # #                 x = int(cx - w / 2)
# # #                 y = int(cy - h / 2)

# # #                 boxes.append([x, y, int(w), int(h)])
# # #                 centers.append((cx, cy))
# # #                 confidences.append(float(confidence))

# # #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)

# # #     final_centers = []

# # #     if len(idxs) > 0:
# # #         for i in idxs.flatten():
# # #             final_centers.append(centers[i])
# # #             cx, cy = centers[i]

# # #             if video:
# # #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# # #             else:
# # #                 x, y, w, h = boxes[i]
# # #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # #     return image, final_centers

# # # # ----------------------------
# # # # HEATMAP
# # # # ----------------------------
# # # def generate_heatmap(image, centers):
# # #     h, w = image.shape[:2]
# # #     heatmap = np.zeros((h, w), dtype=np.float32)

# # #     for x, y in centers:
# # #         x = max(0, min(x, w - 1))
# # #         y = max(0, min(y, h - 1))
# # #         heatmap[y, x] += 1

# # #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# # #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# # #     return cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

# # # # ----------------------------
# # # # ZONE ANALYSIS
# # # # ----------------------------
# # # def zone_analysis(width, centers):
# # #     left = center = right = 0
# # #     for x, _ in centers:
# # #         if x < width/3:
# # #             left += 1
# # #         elif x < 2*width/3:
# # #             center += 1
# # #         else:
# # #             right += 1
# # #     return left, center, right

# # # # ----------------------------
# # # # PDF REPORT (FULL DETAILS)
# # # # ----------------------------
# # # def generate_pdf(image, heatmap, history, count, zones, alerts):

# # #     cv2.imwrite("det.jpg", image)
# # #     cv2.imwrite("heat.jpg", heatmap)

# # #     plt.figure()
# # #     plt.plot(history)
# # #     plt.title("Crowd Trend")
# # #     plt.savefig("graph.png")
# # #     plt.close()

# # #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# # #     styles = getSampleStyleSheet()
# # #     story = []

# # #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# # #     story.append(Spacer(1, 10))

# # #     story.append(Paragraph(f"Final Count: {count}", styles["Normal"]))
# # #     story.append(Paragraph(f"Zones → Left={zones[0]}, Center={zones[1]}, Right={zones[2]}", styles["Normal"]))

# # #     story.append(Spacer(1, 10))
# # #     story.append(Paragraph("Detection", styles["Heading2"]))
# # #     story.append(RLImage("det.jpg", width=400, height=250))

# # #     story.append(Paragraph("Heatmap", styles["Heading2"]))
# # #     story.append(RLImage("heat.jpg", width=400, height=250))

# # #     story.append(Paragraph("Trend Graph", styles["Heading2"]))
# # #     story.append(RLImage("graph.png", width=400, height=200))

# # #     # Alerts
# # #     if len(alerts) > 0:
# # #         story.append(Paragraph("Top Alerts", styles["Heading2"]))
# # #         for a in alerts[:5]:
# # #             story.append(Paragraph(f"{a['Time']} - {a['Count']} ({a['Status']})", styles["Normal"]))

# # #     pdf.build(story)

# # #     with open("report.pdf", "rb") as f:
# # #         return f.read()

# # # # ----------------------------
# # # # UI
# # # # ----------------------------
# # # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # # uploaded = st.file_uploader("Upload", type=["jpg","png","mp4"])

# # # col1, col2 = st.columns(2)

# # # # ----------------------------
# # # # IMAGE MODE
# # # # ----------------------------
# # # if uploaded and mode == "Image":

# # #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# # #     if st.button("Detect"):

# # #         out, centers = detect_people(img, 640, conf, nms)
# # #         count = len(centers)

# # #         heat = generate_heatmap(img.copy(), centers)
# # #         zones = zone_analysis(img.shape[1], centers)

# # #         st.write(f"📍 Zones → Left={zones[0]}, Center={zones[1]}, Right={zones[2]}")

# # #         col1.image(out, caption=f"Count = {count}")
# # #         col2.image(heat)

# # #         alert = get_crowd_alert(count)
# # #         st.session_state.alerts = [alert]

# # #         st.dataframe(pd.DataFrame(st.session_state.alerts))

# # #         pdf = generate_pdf(out, heat, [count], count, zones, st.session_state.alerts)

# # #         st.download_button("📄 Download Report", pdf, "report.pdf")

# # # # ----------------------------
# # # # VIDEO MODE
# # # # ----------------------------
# # # if uploaded and mode == "Video":

# # #     tfile = tempfile.NamedTemporaryFile(delete=False)
# # #     tfile.write(uploaded.read())

# # #     cap = cv2.VideoCapture(tfile.name)

# # #     c1, c2 = st.columns(2)

# # #     if c1.button("▶ Start"):
# # #         st.session_state.run_video = True
# # #         st.session_state.history = []
# # #         st.session_state.alerts = []

# # #     if c2.button("⛔ Stop"):
# # #         st.session_state.run_video = False

# # #     frame_box = col1.empty()
# # #     heat_box = col2.empty()

# # #     while cap.isOpened() and st.session_state.run_video:

# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         out, centers = detect_people(frame, 640, conf, nms, video=True)
# # #         count = len(centers)

# # #         st.session_state.history.append(count)
# # #         st.session_state.alerts.append(get_crowd_alert(count))

# # #         heat = generate_heatmap(frame.copy(), centers)

# # #         frame_box.image(out)
# # #         heat_box.image(heat)

# # #         st.session_state.final_frame = out
# # #         st.session_state.final_heatmap = heat

# # #     cap.release()

# # #     if st.session_state.final_frame is not None:

# # #         zones = zone_analysis(800, centers)

# # #         col1.image(st.session_state.final_frame)
# # #         col2.image(st.session_state.final_heatmap)

# # #         # TOP 5 ALERTS ONLY
# # #         df = pd.DataFrame(st.session_state.alerts)
# # #         df = df.sort_values("Count", ascending=False).head(5)

# # #         st.subheader("Top 5 Alerts")
# # #         st.dataframe(df)

# # #         pdf = generate_pdf(
# # #             st.session_state.final_frame,
# # #             st.session_state.final_heatmap,
# # #             st.session_state.history,
# # #             st.session_state.history[-1],
# # #             zones,
# # #             df.to_dict("records")
# # #         )

# # #         st.download_button("📄 Download Final Report", pdf, "final_report.pdf")










# # import streamlit as st
# # import cv2
# # import numpy as np
# # import imutils
# # import pandas as pd
# # import tempfile
# # from config import YOLO_CONFIG
# # from datetime import datetime
# # import matplotlib.pyplot as plt

# # from reportlab.lib.pagesizes import A4
# # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# # from reportlab.lib.styles import getSampleStyleSheet

# # # ----------------------------
# # # PAGE CONFIG
# # # ----------------------------
# # st.set_page_config(page_title="Crowd Counting", layout="wide")
# # st.title("🚶 Crowd Counting System")

# # # ----------------------------
# # # SESSION STATE
# # # ----------------------------
# # for key in ["history", "alerts", "run_video", "final_frame", "final_heatmap"]:
# #     if key not in st.session_state:
# #         st.session_state[key] = [] if key in ["history", "alerts"] else None

# # # ----------------------------
# # # LOAD MODEL
# # # ----------------------------
# # @st.cache_resource
# # def load_model():
# #     net = cv2.dnn.readNetFromDarknet(
# #         YOLO_CONFIG["CONFIG_PATH"],
# #         YOLO_CONFIG["WEIGHTS_PATH"]
# #     )
# #     ln = net.getLayerNames()
# #     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# #     return net, ln

# # net, ln = load_model()

# # # ----------------------------
# # # ALERT SYSTEM
# # # ----------------------------
# # def get_crowd_alert(count):
# #     time_now = datetime.now().strftime("%H:%M:%S")

# #     if count <= 20:
# #         status, msg = "Safe", "Crowd under control"
# #     elif count <= 40:
# #         status, msg = "Crowded", "Moderate crowd"
# #     elif count <= 60:
# #         status, msg = "High", "High density"
# #     else:
# #         status, msg = "Danger", "Critical crowd"

# #     return {"Time": time_now, "Count": count, "Status": status, "Message": msg}

# # # ----------------------------
# # # DETECTION
# # # ----------------------------
# # def detect_people(image, input_size, conf, nms, video=False):
# #     image = imutils.resize(image, width=800)
# #     H, W = image.shape[:2]

# #     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size),
# #                                  swapRB=True, crop=False)

# #     net.setInput(blob)
# #     outputs = net.forward(ln)

# #     boxes, centers, confidences = [], [], []

# #     for output in outputs:
# #         for det in output:
# #             scores = det[5:]
# #             classID = np.argmax(scores)
# #             confidence = scores[classID]

# #             if classID == 0 and confidence > conf:
# #                 box = det[0:4] * np.array([W, H, W, H])
# #                 cx, cy, w, h = box.astype("int")

# #                 x = int(cx - w / 2)
# #                 y = int(cy - h / 2)

# #                 boxes.append([x, y, int(w), int(h)])
# #                 centers.append((cx, cy))
# #                 confidences.append(float(confidence))

# #     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)
# #     final_centers = []

# #     if len(idxs) > 0:
# #         for i in idxs.flatten():
# #             final_centers.append(centers[i])
# #             cx, cy = centers[i]

# #             if video:
# #                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
# #             else:
# #                 x, y, w, h = boxes[i]
# #                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# #     return image, final_centers

# # # ----------------------------
# # # HEATMAP
# # # ----------------------------
# # def generate_heatmap(image, centers):
# #     h, w = image.shape[:2]
# #     heatmap = np.zeros((h, w), dtype=np.float32)

# #     for x, y in centers:
# #         x = max(0, min(x, w - 1))
# #         y = max(0, min(y, h - 1))
# #         heatmap[y, x] += 1

# #     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
# #     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# #     return cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

# # # ----------------------------
# # # ZONE ANALYSIS
# # # ----------------------------
# # def zone_analysis(width, centers):
# #     left = center = right = 0
# #     for x, _ in centers:
# #         if x < width/3:
# #             left += 1
# #         elif x < 2*width/3:
# #             center += 1
# #         else:
# #             right += 1
# #     return left, center, right

# # # ----------------------------
# # # PDF REPORT
# # # ----------------------------
# # def generate_pdf(image, heatmap, history, count, zones, alerts):
# #     cv2.imwrite("det.jpg", image)
# #     cv2.imwrite("heat.jpg", heatmap)

# #     plt.figure()
# #     plt.plot(history)
# #     plt.title("Crowd Trend")
# #     plt.xlabel("Frame/Time")
# #     plt.ylabel("Person Count")
# #     plt.savefig("graph.png")
# #     plt.close()

# #     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
# #     styles = getSampleStyleSheet()
# #     story = []

# #     story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
# #     story.append(Spacer(1, 10))
# #     story.append(Paragraph(f"Final Count: {count}", styles["Normal"]))
# #     story.append(Paragraph(f"Zones → Left={zones[0]}, Center={zones[1]}, Right={zones[2]}", styles["Normal"]))
# #     story.append(Spacer(1, 10))
    
# #     story.append(Paragraph("Detection Visualization", styles["Heading2"]))
# #     story.append(RLImage("det.jpg", width=400, height=250))
# #     story.append(Paragraph("Heatmap Visualization", styles["Heading2"]))
# #     story.append(RLImage("heat.jpg", width=400, height=250))
# #     story.append(Paragraph("Crowd Trend Graph", styles["Heading2"]))
# #     story.append(RLImage("graph.png", width=400, height=200))

# #     if len(alerts) > 0:
# #         story.append(Paragraph("Key Alerts", styles["Heading2"]))
# #         for a in alerts[:5]:
# #             story.append(Paragraph(f"{a['Time']} - {a['Count']} ({a['Status']})", styles["Normal"]))

# #     pdf.build(story)
# #     with open("report.pdf", "rb") as f:
# #         return f.read()

# # # ----------------------------
# # # UI LAYOUT
# # # ----------------------------
# # mode = st.sidebar.radio("Mode", ["Image", "Video"])
# # conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# # nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)

# # uploaded = st.file_uploader("Upload", type=["jpg","png","mp4"])

# # col1, col2 = st.columns(2)
# # graph_col = st.container() # Fixed container for graphs

# # # ----------------------------
# # # IMAGE MODE
# # # ----------------------------
# # if uploaded and mode == "Image":
# #     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

# #     if st.button("Detect"):
# #         out, centers = detect_people(img, 640, conf, nms)
# #         count = len(centers)
# #         heat = generate_heatmap(img.copy(), centers)
# #         zones = zone_analysis(img.shape[1], centers)

# #         col1.image(out, caption=f"Detection (Count: {count})")
# #         col2.image(heat, caption="Crowd Heatmap")

# #         with graph_col:
# #             st.subheader("📊 Zone Analysis")
# #             zone_data = pd.DataFrame({
# #                 'Zone': ['Left', 'Center', 'Right'],
# #                 'Count': [zones[0], zones[1], zones[2]]
# #             })
# #             st.bar_chart(zone_data.set_index('Zone'))

# #         alert = get_crowd_alert(count)
# #         st.session_state.alerts = [alert]
# #         st.dataframe(pd.DataFrame(st.session_state.alerts))

# #         pdf = generate_pdf(out, heat, [count], count, zones, st.session_state.alerts)
# #         st.download_button("📄 Download Report", pdf, "report.pdf")

# # # ----------------------------
# # # VIDEO MODE
# # # ----------------------------
# # # ----------------------------
# # # VIDEO MODE (FIXED)
# # # ----------------------------
# # if uploaded and mode == "Video":
# #     tfile = tempfile.NamedTemporaryFile(delete=False)
# #     tfile.write(uploaded.read())
# #     cap = cv2.VideoCapture(tfile.name)

# #     c1, c2 = st.columns(2)
# #     if c1.button("▶ Start Analysis"):
# #         st.session_state.run_video = True
# #         st.session_state.history = []
# #         st.session_state.alerts = []
# #         # Reset final frame data
# #         st.session_state.final_frame = None
# #         st.session_state.final_heatmap = None

# #     if c2.button("⛔ Stop"):
# #         st.session_state.run_video = False

# #     frame_box = col1.empty()
# #     heat_box = col2.empty()
    
# #     with graph_col:
# #         st.subheader("📈 Live Crowd Trend")
# #         live_chart = st.empty()

# #     # Initialize centers to avoid NameError if loop is skipped
# #     centers = [] 

# #     while cap.isOpened() and st.session_state.run_video:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         out, centers = detect_people(frame, 640, conf, nms, video=True)
# #         count = len(centers)

# #         st.session_state.history.append(count)
# #         st.session_state.alerts.append(get_crowd_alert(count))

# #         heat = generate_heatmap(frame.copy(), centers)

# #         frame_box.image(out)
# #         heat_box.image(heat)
        
# #         live_chart.line_chart(st.session_state.history)

# #         st.session_state.final_frame = out
# #         st.session_state.final_heatmap = heat

# #     cap.release()

# #     # Only run this if we actually processed at least one frame
# #     if st.session_state.final_frame is not None:
# #         # Use the width of the final processed frame
# #         h_final, w_final = st.session_state.final_frame.shape[:2]
# #         zones = zone_analysis(w_final, centers) 

# #         st.success("Analysis Complete")
        
# #         df = pd.DataFrame(st.session_state.alerts)
# #         df_display = df.sort_values("Count", ascending=False).head(5)

# #         st.subheader("⚠️ Top Crowd Density Alerts")
# #         st.dataframe(df_display, use_container_width=True)

# #         final_count = st.session_state.history[-1] if st.session_state.history else 0

# #         pdf = generate_pdf(
# #             st.session_state.final_frame,
# #             st.session_state.final_heatmap,
# #             st.session_state.history,
# #             final_count,
# #             zones,
# #             df_display.to_dict("records")
# #         )

# #         st.download_button("📄 Download Final Report", pdf, "final_report.pdf")



# import streamlit as st
# import cv2
# import numpy as np
# import imutils
# import pandas as pd
# import tempfile
# from config import YOLO_CONFIG
# from datetime import datetime
# import matplotlib.pyplot as plt

# from reportlab.lib.pagesizes import A4
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# from reportlab.lib.styles import getSampleStyleSheet

# # ----------------------------
# # PAGE CONFIG
# # ----------------------------
# st.set_page_config(page_title="Crowd Counting", layout="wide")
# st.title("🚶 Crowd Counting System")

# # ----------------------------
# # SESSION STATE
# # ----------------------------
# for key in ["history", "alerts", "run_video", "final_frame", "final_heatmap"]:
#     if key not in st.session_state:
#         st.session_state[key] = [] if key in ["history", "alerts"] else None

# # ----------------------------
# # LOAD MODEL
# # ----------------------------
# @st.cache_resource
# def load_model():
#     net = cv2.dnn.readNetFromDarknet(
#         YOLO_CONFIG["CONFIG_PATH"],
#         YOLO_CONFIG["WEIGHTS_PATH"]
#     )
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#     return net, ln

# net, ln = load_model()

# # ----------------------------
# # ALERT SYSTEM
# # ----------------------------
# def get_crowd_alert(count):
#     time_now = datetime.now().strftime("%H:%M:%S")
#     if count <= 20:
#         status, msg = "Safe", "Crowd under control"
#     elif count <= 40:
#         status, msg = "Crowded", "Moderate crowd"
#     elif count <= 60:
#         status, msg = "High", "High density"
#     else:
#         status, msg = "Danger", "Critical crowd"
#     return {"Time": time_now, "Count": count, "Status": status, "Message": msg}

# # ----------------------------
# # DETECTION
# # ----------------------------
# def detect_people(image, input_size, conf, nms, video=False):
#     image = imutils.resize(image, width=800)
#     H, W = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(ln)
#     boxes, centers, confidences = [], [], []

#     for output in outputs:
#         for det in output:
#             scores = det[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]
#             if classID == 0 and confidence > conf:
#                 box = det[0:4] * np.array([W, H, W, H])
#                 cx, cy, w, h = box.astype("int")
#                 x = int(cx - w / 2)
#                 y = int(cy - h / 2)
#                 boxes.append([x, y, int(w), int(h)])
#                 centers.append((cx, cy))
#                 confidences.append(float(confidence))

#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)
#     final_centers = []
#     if len(idxs) > 0:
#         for i in idxs.flatten():
#             final_centers.append(centers[i])
#             cx, cy = centers[i]
#             if video:
#                 cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
#             else:
#                 x, y, w, h = boxes[i]
#                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return image, final_centers

# # ----------------------------
# # HEATMAP
# # ----------------------------
# def generate_heatmap(image, centers):
#     h, w = image.shape[:2]
#     heatmap = np.zeros((h, w), dtype=np.float32)
#     for x, y in centers:
#         x = max(0, min(x, w - 1))
#         y = max(0, min(y, h - 1))
#         heatmap[y, x] += 1
#     heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
#     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
#     return cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

# # ----------------------------
# # ZONE ANALYSIS
# # ----------------------------
# def zone_analysis(width, centers):
#     left = center = right = 0
#     for x, _ in centers:
#         if x < width/3: left += 1
#         elif x < 2*width/3: center += 1
#         else: right += 1
#     return left, center, right

# # ----------------------------
# # PDF REPORT
# # ----------------------------
# def generate_pdf(image, heatmap, history, count, zones, alerts):
#     cv2.imwrite("det.jpg", image)
#     cv2.imwrite("heat.jpg", heatmap)
#     plt.figure()
#     plt.plot(history)
#     plt.title("Crowd Trend")
#     plt.savefig("graph.png")
#     plt.close()
#     pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
#     styles = getSampleStyleSheet()
#     story = [Paragraph("Crowd Analysis Report", styles["Title"]), Spacer(1, 10)]
#     story.append(Paragraph(f"Final Count: {count} | Zones: L:{zones[0]} C:{zones[1]} R:{zones[2]}", styles["Normal"]))
#     story.append(RLImage("det.jpg", width=400, height=250))
#     story.append(RLImage("heat.jpg", width=400, height=250))
#     story.append(RLImage("graph.png", width=400, height=200))
#     pdf.build(story)
#     with open("report.pdf", "rb") as f: return f.read()

# # ----------------------------
# # UI
# # ----------------------------
# mode = st.sidebar.radio("Mode", ["Image", "Video"])
# conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
# nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)
# uploaded = st.file_uploader("Upload", type=["jpg","png","mp4"])

# # ----------------------------
# # IMAGE MODE
# # ----------------------------
# if uploaded and mode == "Image":
#     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
#     if st.button("Detect"):
#         out, centers = detect_people(img, 640, conf, nms)
#         heat = generate_heatmap(img.copy(), centers)
        
#         c1, c2 = st.columns(2)
#         c1.image(out, caption=f"Count: {len(centers)}")
#         c2.image(heat, caption="Heatmap")
        
#         zones = zone_analysis(img.shape[1], centers)
#         st.bar_chart(pd.DataFrame({'Zone':['Left','Center','Right'], 'Count':list(zones)}).set_index('Zone'))
        
#         pdf = generate_pdf(out, heat, [len(centers)], len(centers), zones, [])
#         st.download_button("📄 Download Report", pdf, "report.pdf")

# # ----------------------------
# # VIDEO MODE
# # ----------------------------
# if uploaded and mode == "Video":
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded.read())
#     cap = cv2.VideoCapture(tfile.name)

#     btn_col1, btn_col2 = st.columns(2)
#     if btn_col1.button("▶ Start"):
#         st.session_state.run_video = True
#         st.session_state.history = []
#     if btn_col2.button("⛔ Stop"):
#         st.session_state.run_video = False

#     # Side-by-side video placeholders
#     v_col1, v_col2 = st.columns(2)
#     frame_box = v_col1.empty()
#     heat_box = v_col2.empty()
    
#     # Full-width graph placeholder below videos
#     st.markdown("---")
#     st.subheader("📈 Real-time Crowd Trend")
#     graph_box = st.empty()

#     centers = [] # Initialize to avoid NameError

#     while cap.isOpened() and st.session_state.run_video:
#         ret, frame = cap.read()
#         if not ret: break

#         out, centers = detect_people(frame, 640, conf, nms, video=True)
#         count = len(centers)
#         st.session_state.history.append(count)
#         heat = generate_heatmap(frame.copy(), centers)

#         # Update visuals
#         frame_box.image(out, caption="Live Detection")
#         heat_box.image(heat, caption="Live Heatmap")
#         graph_box.line_chart(st.session_state.history)

#         st.session_state.final_frame = out
#         st.session_state.final_heatmap = heat

#     cap.release()

#     if st.session_state.final_frame is not None:
#         h_f, w_f = st.session_state.final_frame.shape[:2]
#         zones = zone_analysis(w_f, centers)
#         pdf = generate_pdf(st.session_state.final_frame, st.session_state.final_heatmap, st.session_state.history, st.session_state.history[-1], zones, [])
#         st.download_button("📄 Download Final Report", pdf, "final_report.pdf")








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
st.title("🚶 Crowd Counting System")

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
# UTILITY FUNCTIONS
# ----------------------------
def get_crowd_alert(count):
    time_now = datetime.now().strftime("%H:%M:%S")
    if count <= 20: status, msg = "Safe", "Crowd under control"
    elif count <= 40: status, msg = "Crowded", "Moderate crowd"
    elif count <= 60: status, msg = "High", "High density"
    else: status, msg = "Danger", "Critical crowd"
    return {"Time": time_now, "Count": count, "Status": status, "Message": msg}

def detect_people(image, input_size, conf, nms, video=False):
    image = imutils.resize(image, width=800)
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)
    boxes, centers, confidences = [], [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == 0 and confidence > conf:
                box = det[0:4] * np.array([W, H, W, H]); cx, cy, w, h = box.astype("int")
                boxes.append([int(cx - w/2), int(cy - h/2), int(w), int(h)])
                centers.append((cx, cy)); confidences.append(float(confidence))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms)
    final_centers = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_centers.append(centers[i]); cx, cy = centers[i]
            if video: cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            else: x, y, w, h = boxes[i]; cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image, final_centers

def generate_heatmap(image, centers):
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for x, y in centers:
        x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
        heatmap[y, x] += 1
    heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

def zone_analysis(width, centers):
    l, c, r = 0, 0, 0
    for x, _ in centers:
        if x < width/3: l += 1
        elif x < 2*width/3: c += 1
        else: r += 1
    return l, c, r

# def generate_pdf(image, heatmap, history, count, zones, alerts):
#     cv2.imwrite("det.jpg", image); cv2.imwrite("heat.jpg", heatmap)
#     plt.figure(); plt.plot(history); plt.title("Crowd Trend"); plt.savefig("graph.png"); plt.close()
#     pdf = SimpleDocTemplate("report.pdf", pagesize=A4); styles = getSampleStyleSheet()
#     story = [Paragraph("Crowd Analysis Report", styles["Title"]), Spacer(1, 10)]
#     story.append(RLImage("det.jpg", width=400, height=250))
#     story.append(RLImage("heat.jpg", width=400, height=250))
#     story.append(RLImage("graph.png", width=400, height=200))
#     pdf.build(story)
#     with open("report.pdf", "rb") as f: return f.read()

def generate_pdf(image, heatmap, history, count, zones, alerts):
    cv2.imwrite("det.jpg", image)
    cv2.imwrite("heat.jpg", heatmap)

    plt.figure(figsize=(6, 4))
    
    # If it's a single image, we plot two points (0 and the count) to show the "jump"
    # If it's a video, we plot the whole history.
    if len(history) == 1:
        plot_data = [0, history[0]]
        plt.plot(plot_data, marker='o', linestyle='-', color='b', label="Current Detection")
        plt.xticks([0, 1], ["Start", "Current"])
    else:
        plt.plot(history, linestyle='-', color='b', label="Crowd Trend")
    
    plt.title("Crowd Trend Analysis")
    plt.ylabel("Person Count")
    plt.xlabel("Time Progression")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graph.png")
    plt.close()

    # --- PDF Building ---
    pdf = SimpleDocTemplate("report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Crowd Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Stats Table
    stats = f"<b>Final Count:</b> {count} | <b>Zones:</b> Left: {zones[0]}, Center: {zones[1]}, Right: {zones[2]}"
    story.append(Paragraph(stats, styles["Normal"]))
    story.append(Spacer(1, 15))

    # Images
    story.append(Paragraph("Visual Analysis (Detection & Heatmap)", styles["Heading2"]))
    story.append(RLImage("det.jpg", width=220, height=140))
    story.append(RLImage("heat.jpg", width=220, height=140))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Trend Growth Analysis", styles["Heading2"]))
    story.append(RLImage("graph.png", width=400, height=220))

    # Alerts Table in PDF
    if alerts:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Density Alerts Log", styles["Heading2"]))
        for a in alerts[:5]:
            msg = f"• {a['Time']} | Count: {a['Count']} | {a['Status']} - {a['Message']}"
            story.append(Paragraph(msg, styles["Normal"]))

    pdf.build(story)
    with open("report.pdf", "rb") as f:
        return f.read()

# ----------------------------
# UI CONTROLS
# ----------------------------
mode = st.sidebar.radio("Mode", ["Image", "Video"])
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
nms = st.sidebar.slider("NMS", 0.1, 1.0, 0.4)
uploaded = st.file_uploader("Upload File", type=["jpg","png","mp4"])

# ----------------------------
# SHARED DISPLAY COMPONENTS
# ----------------------------
def display_analytics(history, alerts, zones):
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Crowd Trend")
        st.line_chart(history)
    with col_b:
        st.subheader("📊 Zone Analysis")
        st.bar_chart(pd.DataFrame({'Zone':['Left','Center','Right'], 'Count':list(zones)}).set_index('Zone'))
    
    st.subheader("⚠️ Top 5 Density Alerts")
    df = pd.DataFrame(alerts)
    if not df.empty:
        st.table(df.sort_values("Count", ascending=False).head(5))

# ----------------------------
# IMAGE MODE
# ----------------------------
# if uploaded and mode == "Image":
#     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
#     if st.button("Process Image"):
#         out, centers = detect_people(img, 640, conf, nms)
#         heat = generate_heatmap(img.copy(), centers)
#         zones = zone_analysis(img.shape[1], centers)
#         count = len(centers)
        
#         c1, c2 = st.columns(2)
#         c1.image(out, caption=f"Detection (Count: {count})")
#         c2.image(heat, caption="Heatmap")
        
#         # In Image mode, history is just the single count
#         display_analytics([count], [get_crowd_alert(count)], zones)
        
#         pdf = generate_pdf(out, heat, [count], count, zones, [])
#         st.download_button("📄 Download Report", pdf, "report.pdf")

# ----------------------------
# IMAGE MODE (FIXED FOR TREND GRAPH)
# ----------------------------
# if uploaded and mode == "Image":
#     img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
#     if st.button("Process Image"):
#         out, centers = detect_people(img, 640, conf, nms)
#         heat = generate_heatmap(img.copy(), centers)
#         zones = zone_analysis(img.shape[1], centers)
#         count = len(centers)
        
#         # Display Visuals
#         c1, c2 = st.columns(2)
#         c1.image(out, caption=f"Detection (Count: {count})")
#         c2.image(heat, caption="Heatmap")
        
#         st.markdown("---")
#         col_a, col_b = st.columns(2)
        
#         with col_a:
#             st.subheader("📈 Crowd Trend")
#             # To prevent "null" graph, we show the point relative to 0
#             image_history = pd.DataFrame({"Count": [0, count]})
#             st.line_chart(image_history)
            
#         with col_b:
#             st.subheader("📊 Zone Analysis")
#             st.bar_chart(pd.DataFrame({'Zone':['Left','Center','Right'], 'Count':list(zones)}).set_index('Zone'))
        
#         st.subheader("⚠️ Alert Message")
#         alert = get_crowd_alert(count)
#         # Display as a table to match video mode
#         st.table(pd.DataFrame([alert])) 
        
#         pdf = generate_pdf(out, heat, [count], count, zones, [alert])
#         st.download_button("📄 Download Report", pdf, "report.pdf")


# ----------------------------
# IMAGE MODE (COMPLETE)
# ----------------------------
if uploaded and mode == "Image":
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

    if st.button("🚀 Process Image"):
        # 1. Run Detection
        out, centers = detect_people(img, 640, conf, nms)
        count = len(centers)
        
        # 2. Generate Analytics
        heat = generate_heatmap(img.copy(), centers)
        zones = zone_analysis(img.shape[1], centers)
        alert = get_crowd_alert(count)
        
        # 3. Display Visuals Side-by-Side
        c1, c2 = st.columns(2)
        with c1:
            st.image(out, caption=f"Detection Results (Count: {count})", use_container_width=True)
        with c2:
            st.image(heat, caption="Crowd Heatmap Density", use_container_width=True)

        # 4. Display Analytics Dashboard
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📈 Crowd Trend")
            # Creating a 2-point data frame to show the "Jump" from 0 to current count
            # This prevents the graph from looking "null"
            image_trend_data = pd.DataFrame({
                "Status": ["Initial", "Current"],
                "Person Count": [0, count]
            })
            st.line_chart(image_trend_data.set_index("Status"))
            
        with col_b:
            st.subheader("📊 Zone Analysis")
            zone_data = pd.DataFrame({
                'Zone': ['Left', 'Center', 'Right'],
                'Count': [zones[0], zones[1], zones[2]]
            })
            st.bar_chart(zone_data.set_index('Zone'))
        
        # 5. Display Alert Message
        st.subheader("⚠️ Crowd Density Alert")
        alert_df = pd.DataFrame([alert])
        st.table(alert_df) 
        
        # 6. PDF Generation and Download
        # We pass [count] as a list so the PDF logic recognizes it as history
        pdf_data = generate_pdf(
            image=out, 
            heatmap=heat, 
            history=[count], 
            count=count, 
            zones=zones, 
            alerts=[alert]
        )
        
        st.download_button(
            label="📄 Download Detailed PDF Report",
            data=pdf_data,
            file_name=f"Crowd_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )


# ----------------------------
# VIDEO MODE
# ----------------------------
if uploaded and mode == "Video":
    tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded.read())
    cap = cv2.VideoCapture(tfile.name)

    b1, b2 = st.columns(2)
    if b1.button("▶ Start"):
        st.session_state.run_video = True
        st.session_state.history, st.session_state.alerts = [], []
    if b2.button("⛔ Stop"):
        st.session_state.run_video = False

    v_col1, v_col2 = st.columns(2)
    f_box, h_box = v_col1.empty(), v_col2.empty()
    
    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    trend_box = g_col1.empty()
    zone_box = g_col2.empty()
    alert_box = st.empty()

    centers = []
    while cap.isOpened() and st.session_state.run_video:
        ret, frame = cap.read()
        if not ret: break
        out, centers = detect_people(frame, 640, conf, nms, video=True)
        count = len(centers)
        heat = generate_heatmap(frame.copy(), centers)
        zones = zone_analysis(frame.shape[1], centers)

        st.session_state.history.append(count)
        st.session_state.alerts.append(get_crowd_alert(count))
        st.session_state.final_zones = zones
        
        f_box.image(out, caption="Live Feed")
        h_box.image(heat, caption="Live Heatmap")
        trend_box.line_chart(st.session_state.history)
        zone_box.bar_chart(pd.DataFrame({'Zone':['Left','Center','Right'], 'Count':list(zones)}).set_index('Zone'))
        
        st.session_state.final_frame, st.session_state.final_heatmap = out, heat

    cap.release()

    # Stable Display after Stop
    if not st.session_state.run_video and st.session_state.final_frame is not None:
        f_box.image(st.session_state.final_frame)
        h_box.image(st.session_state.final_heatmap)
        trend_box.line_chart(st.session_state.history)
        zone_box.bar_chart(pd.DataFrame({'Zone':['Left','Center','Right'], 'Count':list(st.session_state.final_zones)}).set_index('Zone'))
        
        with alert_box.container():
            st.subheader("⚠️ Top 5 Density Alerts")
            df = pd.DataFrame(st.session_state.alerts)
            st.table(df.sort_values("Count", ascending=False).head(5))
            
        pdf = generate_pdf(st.session_state.final_frame, st.session_state.final_heatmap, st.session_state.history, st.session_state.history[-1], st.session_state.final_zones, [])
        st.download_button("📄 Download Final Report", pdf, "final_report.pdf")