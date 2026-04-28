import time
import datetime
import numpy as np
import imutils
import cv2
from scipy.spatial.distance import euclidean

from tracking import detect_human
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS

from config import (
    SHOW_DETECT, DATA_RECORD, RE_CHECK, RE_START_TIME, RE_END_TIME,
    SD_CHECK, SHOW_VIOLATION_COUNT, SHOW_TRACKING_ID, SOCIAL_DISTANCE,
    SHOW_PROCESSING_OUTPUT, VIDEO_CONFIG, DATA_RECORD_RATE,
    ABNORMAL_CHECK, ABNORMAL_ENERGY, ABNORMAL_THRESH, ABNORMAL_MIN_PEOPLE
)

IS_CAM = VIDEO_CONFIG["IS_CAM"]
HIGH_CAM = VIDEO_CONFIG["HIGH_CAM"]


# -------------------- HELPERS --------------------

def _record_movement_data(writer, movement):
    data = [movement.track_id, movement.entry, movement.exit] + list(np.array(movement.positions).flatten())
    writer.writerow(data)


def _record_crowd_data(t, count, violate, restricted, abnormal, writer):
    writer.writerow([t, count, int(violate), int(restricted), int(abnormal)])


def _end_video(tracker, frame_count, writer):
    for t in tracker.tracks:
        if t.is_confirmed():
            t.exit = frame_count
            _record_movement_data(writer, t)


# -------------------- MAIN FUNCTION --------------------

def video_process(cap, frame_size, net, ln, encoder, tracker, movement_writer, crowd_writer):

    frame_count = 0
    display_frame_count = 0

    # ================= SAFE FPS =================
    VID_FPS = cap.get(cv2.CAP_PROP_FPS)
    if VID_FPS is None or VID_FPS <= 1:
        print("Warning: FPS is invalid, setting default FPS = 25")
        VID_FPS = 25

    DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE) if DATA_RECORD_RATE > 0 else 1
    DATA_RECORD_FRAME = max(1, DATA_RECORD_FRAME)
    TIME_STEP = max(1e-6, DATA_RECORD_FRAME / VID_FPS)

    t0 = time.time()

    def _calculate_FPS():
        return frame_count / (time.time() - t0 + 1e-6)

    # ================= VIDEO WRITER =================
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # -------------------- LOOP --------------------
    while True:
        ret, frame = cap.read()

        if not ret:
            _end_video(tracker, frame_count, movement_writer)
            VID_FPS = _calculate_FPS()
            break

        frame_count += 1

        if frame_count % DATA_RECORD_FRAME != 0:
            continue

        display_frame_count += 1

        frame = imutils.resize(frame, width=frame_size)

        # 🔥 Initialize writer once
        if out is None:
            (h, w) = frame.shape[:2]
            out = cv2.VideoWriter("output.mp4", fourcc, 20, (w, h))
            print("Saving output video...")

        current_time = datetime.datetime.now()
        record_time = current_time if IS_CAM else frame_count

        # ---------------- DETECTION ----------------
        humans, expired = detect_human(net, ln, frame, encoder, tracker, record_time)

        for m in expired:
            _record_movement_data(movement_writer, m)

        # ---------------- SOCIAL DISTANCE ----------------
        violate_set = set()
        violate_count = np.zeros(len(humans))

        if SD_CHECK:
            for i, t1 in enumerate(humans):
                x1, y1, w1, h1 = map(int, t1.to_tlbr())
                cx1, cy1 = map(int, t1.positions[-1])

                for j, t2 in enumerate(humans[i+1:], start=i+1):
                    if HIGH_CAM:
                        cx2, cy2 = map(int, t2.positions[-1])
                        dist = euclidean((cx1, cy1), (cx2, cy2))
                    else:
                        x2, y2, w2, h2 = map(int, t2.to_tlbr())
                        dist = rect_distance((x1, y1, w1, h1), (x2, y2, w2, h2))

                    if dist < SOCIAL_DISTANCE:
                        violate_set.update([i, j])
                        violate_count[i] += 1
                        violate_count[j] += 1

        # ---------------- ABNORMAL ----------------
        abnormal_ids = []
        ABNORMAL = False

        if ABNORMAL_CHECK and len(humans) > 2:
            for t in humans:
                if len(t.positions) >= 2:
                    ke = kinetic_energy(t.positions[-1], t.positions[-2], TIME_STEP)
                    if ke > ABNORMAL_ENERGY:
                        abnormal_ids.append(t.track_id)

            if len(abnormal_ids) / len(humans) > ABNORMAL_THRESH:
                ABNORMAL = True

        # ---------------- DRAW ----------------
        for i, track in enumerate(humans):
            x, y, w, h = map(int, track.to_tlbr())

            color = (0, 255, 0)
            if i in violate_set:
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if SHOW_TRACKING_ID:
                cv2.putText(frame, str(track.track_id), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---------------- SAVE FRAME ----------------
        if out is not None:
            out.write(frame)

        # ---------------- DISPLAY ----------------
        if SHOW_PROCESSING_OUTPUT:
            cv2.imshow("Output", frame)
        else:
            progress(display_frame_count)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            _end_video(tracker, frame_count, movement_writer)
            VID_FPS = _calculate_FPS()
            break

    # ================= CLEANUP =================
    if out is not None:
        out.release()
        print("✅ Output video saved as output.mp4")

    cv2.destroyAllWindows()
    return VID_FPSc