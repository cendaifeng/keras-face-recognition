import os
import time

import RPi.GPIO as GPIO
import threading
import atexit

atexit.register(GPIO.cleanup)
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN)
GPIO.setup(17, GPIO.OUT, initial=True)

event = threading.Event()
password = [1, 1, 0, 0]
Input = []


def run():
    GPIO.add_event_detect(4, GPIO.BOTH)
    # LOCK WAIT
    while True:
        event.wait()
        print("\33[42;1mPASSWORD ON! unlocking...\033[0m")
        #with os.popen('python ' + os.path.join(os.path.split( os.path.realpath('__file__') )[0], './unlock.py')) as f:
        with os.popen('python /home/pi/git_dir/keras-face-recognition/unlock.py') as f:
            print(f.read())
        event.clear()


def psw():
    global password
    global Input
    psw_size = len(password)
    while True:
        GPIO.wait_for_edge(4, GPIO.BOTH)
        print('#into#')
        detect_binary(Input)
        for _ in range(psw_size-1):
            edge = GPIO.wait_for_edge(4, GPIO.BOTH, timeout=5000)
            if edge is None:
                print('break')
                break
            detect_binary(Input)
        print(Input)
        if Input == password:
            event.set()  # 设置信号量，异步调用开锁函数
        Input.clear()


def detect_binary(arr):
    t0 = time.perf_counter()
    GPIO.wait_for_edge(4, GPIO.BOTH)
    T = time.perf_counter() - t0
    print(T)
    if T > 0.6:
        print(1)
        arr.append(1)
    else:
        print(0)
        arr.append(0)


if __name__ == "__main__":
    # 多线程调用
    run = threading.Thread(target=run)
    run.start()
    psw = threading.Thread(target=psw)
    psw.start()
