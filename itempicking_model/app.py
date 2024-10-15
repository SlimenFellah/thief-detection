import os
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layers correctly
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create an uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video
    output_video_path = process_video(video_path)

    return {"filename": file.filename, "output_video": output_video_path}

def process_video(video_path: str):
    output_path = video_path.replace(".mp4", "_processed.mp4")
    cap = cv2.VideoCapture(video_path)

    # Video writer for saving the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Reduce size by 50%
        height, width = frame.shape[:2]

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Loop through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Store box coordinates, confidence, and class ID
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Loop through the indices and draw boxes for detected objects
        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, np.ndarray) else i
                box = boxes[i]
                x, y, w, h = box
                obj_name = classes[class_ids[i]]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, obj_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Perform pose estimation only for detected persons
                if obj_name == "person":
                    # Get the region of interest (ROI) for pose estimation
                    person_roi = frame[y:y + h, x:x + w]
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    result = pose.process(person_roi_rgb)

                    # Draw pose landmarks on the person
                    if result.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # Extract relevant landmarks
                        landmarks = result.pose_landmarks.landmark
                        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                        # Calculate distance for action recognition
                        wrist_to_hip_distance = np.sqrt((right_wrist.x - right_hip.x) ** 2 + (right_wrist.y - right_hip.y) ** 2)

                        # Define a threshold for action detection
                        threshold = 0.05

                        # Check if the right wrist is near the right hip (pocket)
                        if wrist_to_hip_distance < threshold:
                            action_text = "Hand to Pocket!"
                            cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return output_path

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>Upload Video</title>
        </head>
        <body>
            <h1>Upload Video for Processing</h1>
            <form action="/uploadvideo/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="video/mp4">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

# Mount the uploads folder to access processed videos
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
