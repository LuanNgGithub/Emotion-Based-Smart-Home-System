import cv2
from ultralytics import YOLO  # 确保这个import语句与您的实际环境相匹配

# 加载YOLOv8模型：人脸检测模型和表情识别模型
face_model = YOLO('/Users/muse/Documents/YOLO_Emotions/Test/weights/yolov8n_face.pt')  # 人脸检测模型
emotion_model = YOLO('/Users/muse/Documents/YOLO_Emotions/Test/weights/best_yolo11x.pt')  # 表情识别模型

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置视频流的分辨率为1080p (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行人脸检测，并设置置信度阈值和NMS IOU阈值
        face_results = face_model(frame, conf=0.7, iou=0.5)

        # 迭代每个检测到的人脸框
        for face in face_results:
            # 获取人脸的坐标框信息
            for box in face.boxes:
                # 提取人脸区域
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_region = frame[y1:y2, x1:x2]

                # 在人脸区域内运行表情检测，并设置置信度阈值和NMS IOU阈值
                emotion_results = emotion_model(face_region, conf=0.7, iou=0.5)

                # 在人脸区域上展示表情检测的结果
                if len(emotion_results) > 0:
                    annotated_face = emotion_results[0].plot()

                    # 将注释后的脸部区域重新放回到原图上
                    frame[y1:y2, x1:x2] = annotated_face

        # 展示带注释的帧（包含人脸和表情）
        cv2.imshow("Emotions Detection", frame)

        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
    else:
        # 如果视频结束则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
