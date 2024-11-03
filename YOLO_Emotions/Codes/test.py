import cv2
import mediapipe as mp
from ultralytics import YOLO

# 初始化 Mediapipe 的人脸检测器
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 加载表情识别模型
emotion_model = YOLO('/Users/muse/Documents/YOLO_Emotions/Test/weights/yolo11s/best.pt')  # 表情识别模型

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置视频流的分辨率为1080p (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 使用 Mediapipe 的人脸检测器
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 将帧转换为 RGB 格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 在帧上运行 Mediapipe 人脸检测
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    # 获取人脸的边界框
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                    x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

                    # 提取人脸区域
                    face_region = frame[y1:y2, x1:x2]

                    # 在人脸区域内运行表情检测
                    emotion_results = emotion_model(face_region, conf=0.5, iou=0.5)

                    # 展示表情检测的结果
                    if len(emotion_results) > 0:
                        annotated_face = emotion_results[0].plot()

                        # 将注释后的脸部区域重新放回到原图上
                        frame[y1:y2, x1:x2] = annotated_face

                    # 可选：在图像上绘制人脸检测框
                    mp_drawing.draw_detection(frame, detection)

            # 展示带注释的帧（包含人脸和表情）
            cv2.imshow("Emotions Detection", frame)

            # 如果按下 'q' 键则退出循环
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break
        else:
            break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()