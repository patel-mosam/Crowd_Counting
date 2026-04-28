# import cv2
# import imutils
# from config import YOLO_CONFIG

# # Load YOLO
# net = cv2.dnn.readNetFromDarknet(
#     YOLO_CONFIG["CONFIG_PATH"],
#     YOLO_CONFIG["WEIGHTS_PATH"]
# )

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# # Load image
# image_path = "images/" # 👈 change image name here
# image = cv2.imread(image_path)

# if image is None:
#     print("❌ Image not found")
#     exit()

# # image = imutils.resize(image, width=800)

# image = imutils.resize(image, width=1000)
# (H, W) = image.shape[:2]

# # Create blob
# # blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
# blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
# net.setInput(blob)
# outputs = net.forward(ln)

# boxes = []
# confidences = []
# classIDs = []

# # Loop detections
# for output in outputs:
#     for detection in output:
#         scores = detection[5:]
#         classID = scores.argmax()
#         confidence = scores[classID]

#         # Only detect PERSON class (class 0 in COCO)
#         # if classID == 0 and confidence > 0.5:
#         if classID == 0 and confidence > 0.25:
#             box = detection[0:4] * [W, H, W, H]
#             (centerX, centerY, width, height) = box.astype("int")

#             x = int(centerX - (width / 2))
#             y = int(centerY - (height / 2))

#             boxes.append([x, y, int(width), int(height)])
#             confidences.append(float(confidence))

# # Apply NMS
# # idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
# idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

# count = 0

# # Draw boxes
# if len(idxs) > 0:
#     for i in idxs.flatten():
#         (x, y, w, h) = boxes[i]
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         count += 1

# # Show count
# cv2.putText(image, f"People Count: {count}", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # Show image
# cv2.imshow("Image Output", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import os
import cv2
import imutils
from config import YOLO_CONFIG

# Load YOLO
net = cv2.dnn.readNetFromDarknet(
    YOLO_CONFIG["CONFIG_PATH"],
    YOLO_CONFIG["WEIGHTS_PATH"]
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# 👇 Dataset folder
dataset_path = "images"

# Create output folder
if not os.path.exists("output"):
    os.makedirs("output")

for file in os.listdir(dataset_path):

    if file.endswith(".jpg") or file.endswith(".png"):

        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image = imutils.resize(image, width=1000)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (640, 640), swapRB=True, crop=False
        )

        net.setInput(blob)
        outputs = net.forward(ln)

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = scores.argmax()
                confidence = scores[classID]

                if classID == 0 and confidence > 0.25:
                    box = detection[0:4] * [W, H, W, H]
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

        count = 0

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count += 1

        cv2.putText(image, f"People Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save output
        output_path = os.path.join("output", file)
        cv2.imwrite(output_path, image)

        print(f"{file} → Count: {count}")