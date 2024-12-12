import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch
import time
from collections import deque  
from collections import Counter
import Jetson.GPIO as GPIO
import time
#test_log.txt

############################################初始化############################################
# 初始化 窗口队列
Confidence_queue = []
WINDOW_SIZE = 30
Activation_threshold = 0.1
Resolution_ratio = [1920, 1080]
Model_address = '/home/jetson/Documents/test/weights/yolo11s/best.pt'
# 定义GPIO引脚
GPIO_LED1 = 29
GPIO_LED2 =31
GPIO_LED3 = 33
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(GPIO_LED1 , GPIO.OUT, initial=GPIO.LOW) 
GPIO.setup(GPIO_LED2 , GPIO.OUT, initial=GPIO.LOW) 
GPIO.setup(GPIO_LED3 , GPIO.OUT, initial=GPIO.LOW) 


def vote(queue, Activation_threshold, WINDOW_SIZE):
    # 使用 Counter 对置信队列进行计数
    emotion_counts = Counter(queue)

    # 找到出现次数最多的情绪标签和它的次数
    predominant_emotion, count = emotion_counts.most_common(1)[0]

    # 输出出现次数最多的情绪和它的次数
    print(f"出现次数最多的情绪是：{predominant_emotion}, 次数是: {count}")
        ###这里结果存在一个很大的问题：有非常多的unknown！是否考虑删掉这些unknown再做考虑？还需要进行实验
    # 判断是否超过激活阈值
    if count / WINDOW_SIZE >= Activation_threshold:
        print(f"情绪 '{predominant_emotion}' 达到激活阈值 {Activation_threshold}，执行相关操作。")
        # 灯控制
        light_control(predominant_emotion)
    else:
        print(f"情绪 '{predominant_emotion}' 未达到激活阈值 {Activation_threshold}，不执行任何操作。")
        #return None


def light_control(predominant_emotion):
    if predominant_emotion == 'Unknown': # 这个为测试用暂时保留
        pass
        # delay
        print("Unknown Keep the normal light is ok")
    elif predominant_emotion == 'Sad':
        GPIO_control(GPIO_LED1 )  # 开灯
        pass
    elif predominant_emotion == 'Happy':
        GPIO_control(GPIO_LED2 )  # 开灯

    elif predominant_emotion == 'Anger':
        GPIO_control(GPIO_LED3 )  # 开灯

    elif predominant_emotion == 'Surprise':
        GPIO_control_double(GPIO_LED1,GPIO_LED2)
        
    elif predominant_emotion == 'Contempt':
        GPIO_control_tri(GPIO_LED1,GPIO_LED2,GPIO_LED3)
    
    elif predominant_emotion == 'Fear':
        GPIO_control_double(GPIO_LED2,GPIO_LED3)
    
    elif predominant_emotion == 'Disgust':
        GPIO_control_double(GPIO_LED1,GPIO_LED3)
    
    else:
        pass   #关灯


def GPIO_control(led_pin):
    GPIO.output(led_pin, GPIO.HIGH) 
    print("LED is ON")
    time.sleep(0.5) 
    GPIO.output(led_pin, GPIO.LOW)

def GPIO_control_double(led_pin1,led_pin2):
    GPIO.output(led_pin1, GPIO.HIGH) 
    GPIO.output(led_pin2, GPIO.HIGH) 
    print("LED is ON")
    time.sleep(0.5) 
    GPIO.output(led_pin1, GPIO.LOW)
    GPIO.output(led_pin2, GPIO.LOW)

def GPIO_control_tri(led_pin1,led_pin2, led_pin3):
    GPIO.output(led_pin1, GPIO.HIGH) 
    GPIO.output(led_pin2, GPIO.HIGH) 
    GPIO.output(led_pin3, GPIO.HIGH) 
    print("LED is ON")
    time.sleep(0.5) 
    GPIO.output(led_pin1, GPIO.LOW)
    GPIO.output(led_pin2, GPIO.LOW)
    GPIO.output(led_pin3, GPIO.LOW) 



##################################################################YOLO PROCEDURE##################################################################


# 初始化Mediapipe人脸检测
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 加载YOLO表情识别模型
emotion_model = YOLO(Model_address, task="detect")

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# 初始化FPS计数
fps = 0
prev_time = time.time()

# 使用Mediapipe人脸检测器
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 将帧转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # 检查是否检测到人脸 
        if not results.detections: 
            # 如果没有检测到人脸，跳出循环 此接口是为了后续的超声波雷达检测使用
            print("No faces detected, exiting...") 
        #elif:
            
            break

        faces = []
        face_positions = []

        # 遍历所有检测到的人脸，提取每个人脸的区域
        if results.detections:
            face_id = 1  # 用于编号每张人脸
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

                # 检查坐标边界，确保人脸区域非空
                if x2 > x1 and y2 > y1:
                    face_region = frame[y1:y2, x1:x2]

                    # 检查 face_region 的非零尺寸
                    if face_region.size != 0:
                        faces.append(face_region)
                        face_positions.append((x1, y1, x2, y2))

            # 批量推理人脸区域
            if faces:
                with torch.no_grad():
                    emotion_results = emotion_model.predict(faces, conf=0.5, iou=0.5)

                # 对每个情绪结果和人脸位置进行逐一对应
                for i, result in enumerate(emotion_results):
                    x1, y1, x2, y2 = face_positions[i]
                    
                    # 提取情绪标签索引和置信度分数
                    if result.boxes is not None and len(result.boxes) > 0:
                        label_index = int(result.boxes.cls[0])
                        emotion_label = result.names[label_index]
                        emotion_score = float(result.boxes.conf[0])

                        # 输出到终端
                        print(f"Face {face_id}: {emotion_label} ({emotion_score:.2f})")
                    else:
                        emotion_label = "Unknown"
                        emotion_score = 0.0
                        print(f"Face {face_id}: {emotion_label} ({emotion_score:.2f})")


                    # 将当前情绪结果加入到置信队列中(删去 unknown)
                    #if emotion_label != "Unknown":
                    Confidence_queue.append(emotion_label)

                    if len(Confidence_queue) > WINDOW_SIZE:
                        # 如果队列超出了窗口大小，移除最早的情绪检测结果
                        Confidence_queue.pop(0)
                    #投票(虚拟)
                    vote(Confidence_queue, Activation_threshold, WINDOW_SIZE)



                    # 将情绪结果绘制在相应的人脸区域上
                    annotated_face = result.plot()
                    frame[y1:y2, x1:x2] = annotated_face

                    # 绘制人脸编号和情绪标签
                    cv2.putText(frame, f"Face {face_id}: {emotion_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    face_id += 1

                    # 在图像上绘制人脸检测框
                    mp_drawing.draw_detection(frame, results.detections[i])

        # 显示FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示带注释的帧
        cv2.imshow("Emotions Detection", frame)


        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # 清理GPIO
