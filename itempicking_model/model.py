import cv2
import numpy as np
import mediapipe as mp

# Load the video
video_path = '1036719695-preview.mp4'

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

# Load the video
cap = cv2.VideoCapture(video_path)

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

    # Loop through detections and collect boxes, confidences, and class IDs
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
    if len(indices) > 0:  # Check if there are any indices returned
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
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

                    # Calculate distances to determine action
                    wrist_to_hip_distance = np.sqrt((right_wrist.x - right_hip.x) ** 2 + (right_wrist.y - right_hip.y) ** 2)

                    # Define a threshold for the action detection
                    threshold = 0.05  # Adjust based on video size and distance

                    # Check if the right wrist is near the right hip (pocket)
                    if wrist_to_hip_distance < threshold:
                        action_text = "Hand to Pocket!"
                        cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Object Detection with Pose Estimation and Action Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
