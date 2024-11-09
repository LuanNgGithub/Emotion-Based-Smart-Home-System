import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch
import time

# 初始化 Mediapipe 的人脸检测
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 加载 YOLO 表情识别模型
emotion_model = YOLO('/Users/muse/Documents/YOLO_Emotions/YOLO_Emotions/weights/yolo11s/best.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置为1080p分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 初始化FPS计数
fps = 0
prev_time = time.time()

# 使用 Mediapipe 的人脸检测器
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 计算当前帧时间，并更新FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 将帧转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

                # 提取人脸区域
                face_region = frame[y1:y2, x1:x2]

                # 使用 torch.no_grad() 禁用梯度，优化 YOLO 表情识别
                with torch.no_grad():
                    emotion_results = emotion_model(face_region, conf=0.5, iou=0.5)

                if len(emotion_results) > 0:
                    annotated_face = emotion_results[0].plot()
                    frame[y1:y2, x1:x2] = annotated_face

                # 可选：在图像上绘制人脸检测框
                mp_drawing.draw_detection(frame, detection)

        # 在图像上显示 FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示带注释的帧（包含人脸和表情）
        cv2.imshow("Emotions Detection", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()