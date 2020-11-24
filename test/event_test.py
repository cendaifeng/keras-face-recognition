import threading
import time

# 利用Event类模拟红绿灯
event = threading.Event()


def lighter():
    count = 0
    while True:

        print("\33[41;1mred light is on...\033[0m")

        time.sleep(1)
        count += 1
        event.set()
        print("\33[42;1mgreen light is on...\033[0m")


def car(name):
    while True:
        if event.is_set():  # 判断是否设置了标志位
            print("[%s] running..." % name)
            time.sleep(1)
        else:
            print("[%s] sees red light,waiting..." % name)
            event.wait()  # 阻塞
            print("[%s] green light is on,start going..." % name)
            time.sleep(1)
            event.clear()  # 红灯 清除标志位


light = threading.Thread(target=lighter, )
light.start()

car = threading.Thread(target=car, args=("MINI",))
car.start()
