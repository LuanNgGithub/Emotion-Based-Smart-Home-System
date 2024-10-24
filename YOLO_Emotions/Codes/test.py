import cv2
import time
from ultralytics import YOLO

# Load YOLO models: face detection model and emotion recognition model
face_model = YOLO('/YOLO_Emotions/Codes/weights/yolo_face.pt')  # Face detection model
emotion_model = YOLO('/YOLO_Emotions/Codes/weights/yolo_emotion.pt')  # Emotion recognition model

# Open the camera
cap = cv2.VideoCapture(0)

# Set the video stream resolution to 1080p (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def process_frame(frame):
    # Run face detection on the frame, set confidence threshold and NMS IOU threshold
    face_results = face_model(frame, conf=0.7, iou=0.5)
    face_regions = []
    boxes = []
    
    # Iterate through each detected face box
    for face in face_results:
        for box in face.boxes:
            # Get the coordinates of the face bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Extract the face region
            face_regions.append(frame[y1:y2, x1:x2])
            boxes.append((x1, y1, x2, y2))
    
    # Run emotion detection on the face regions, set confidence threshold and NMS IOU threshold
    if len(face_regions) > 0:
        emotion_results = emotion_model(face_regions, conf=0.7, iou=0.5)
        
        # Display the emotion detection results on the face regions
        for i, emotion_result in enumerate(emotion_results):
            annotated_face = emotion_result.plot()
            x1, y1, x2, y2 = boxes[i]
            # Put the annotated face region back into the original frame
            frame[y1:y2, x1:x2] = annotated_face
    
    return frame

# Loop through video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Calculate frame processing time to display FPS
    start_time = time.time()

    # Process the current frame
    frame = process_frame(frame)

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the annotated frame (with faces and emotions)
    cv2.imshow("Emotions Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
