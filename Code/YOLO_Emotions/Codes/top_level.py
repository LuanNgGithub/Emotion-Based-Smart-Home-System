import os
import time
import subprocess
import logging
import Jetson.GPIO as GPIO

#sudo systemctl enable top_level.service
#sudo systemctl stop top_level.service



# 配置日志
logging.basicConfig(filename="/home/jetson/Desktop/recording.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 初始化 GPIO, 用于状态显示和超声波传感器
LED_STATUS_PIN = 15  # 显示进程状态的 LED 引脚
GPIO_TRIGGER = 7  # 超声波传感器 Trig 引脚###################Test the trigger first
GPIO_ECHO = 32  # 超声波传感器 Echo 引脚


GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_STATUS_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# 超声波测距函数
def measure_distance():
    # 发送高电平信号到 Trig 引脚
    GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, GPIO.LOW)

    # 记录发送超声波的时刻
    start_time = time.time()
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()

    # 记录接收到返回超声波的时刻
    stop_time = time.time()
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()

    # 计算往返时间并转换为距离 (cm)
    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2
    return distance





# 控制 LED 状态
#def set_led_status(is_running):
 #   GPIO.output(LED_STATUS_PIN, GPIO.HIGH if is_running else GPIO.LOW)
  #  GPIO.output(LED_PIN, GPIO.HIGH)  # 设置 LED 引脚为高电平，保持亮




# 主进程循环
def main_loop():
    yolo_process = None
    GPIO.output(LED_STATUS_PIN, GPIO.HIGH)  # 设置 LED 引脚为高电平，保持亮
    try:
        while True:
            # 检查 YOLO 子进程是否已退出
            if yolo_process is not None and yolo_process.poll() is None:
                logging.info("YOLO process is still running.")
                time.sleep(1)  # 延迟检查
                continue

            # 测量距离
            dist = measure_distance()
            logging.info(f"Measured distance: {dist:.2f} cm")

            # 检查距离触发条件80cm
            if dist < 80:
                logging.info("Trigger condition met (distance < 80cm). Starting YOLO process...")
                yolo_process = subprocess.Popen(["python3", "/home/jetson/Documents/test/programm.py"])
          
            else:

                logging.info("No trigger condition met. Waiting...")

            time.sleep(1)  # 主循环间隔

    except KeyboardInterrupt:
        logging.info("Process manager stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if yolo_process is not None:
            yolo_process.terminate()
        GPIO.cleanup()


if __name__ == "__main__":
    main_loop()
