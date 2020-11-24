import time
import RPi.GPIO as GPIO
import atexit

# 初始化舵机
atexit.register(GPIO.cleanup)
servopin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servopin, GPIO.OUT, initial=False)
pwm = GPIO.PWM(servopin, 50)  # 50HZ


def ControlMotor(angle):
    # 舵机的频率为50HZ，占空比为2.5%-12.5%，线性对应舵机转动角度的0-180度
    pwm.ChangeDutyCycle(2.5 + angle / 360 * 20)

print("unlock!")

pwm.start(0)
ControlMotor(170)
time.sleep(5)
ControlMotor(0)
#pwm.stop()
time.sleep(1)

