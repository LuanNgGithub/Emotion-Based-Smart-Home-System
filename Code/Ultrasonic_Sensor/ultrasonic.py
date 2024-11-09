# ECE 411 FA24 - Team 6
import RPi.GPIO as GPIO
import time

# Set GPIO mode to BOARD
GPIO.setmode(GPIO.BOARD)

# Define GPIO pins for the sensor
TRIG = 31  # Trigger pin
ECHO = 7  # Echo pin

# Set up the GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def distance():
    # Ensure the trigger pin is low
    GPIO.output(TRIG, False)
    time.sleep(2)  # Wait for the sensor to settle

    # Send a 10us pulse to trigger the sensor
    GPIO.output(TRIG, True)
    time.sleep(0.00001)  # 10 microseconds
    GPIO.output(TRIG, False)

    # Measure the duration of the echo pulse
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Calculate the duration of the pulse
    pulse_duration = pulse_end - pulse_start

    # Calculate the distance (speed of sound is 34300 cm/s)
    distance = pulse_duration * 34300 / 2

    return distance

try:
    while True:
        dist = distance()
        print(f"Measured Distance = {dist:.2f} cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
    GPIO.cleanup()
