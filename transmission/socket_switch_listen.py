import os
import threading
from socket import *

HOST = ''
PORT = 9902
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  # 端口可复用
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

# 利用Event类模拟信号量
event = threading.Event()


def run():
    while True:
        print('waiting for connection...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('...connnecting from:', addr)
        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data:
                break
            if data.decode('utf-8') == 'open650':
                print('==recv==')
                event.set()  # 设置信号量，异步调用开锁函数

        tcpCliSock.close()
    tcpSerSock.close()


def lock_wait():
    while True:
        event.wait()
        print("\33[42;1mswitch on! unlocking...\033[0m")
        with os.popen('python ' + os.path.join(os.path.split( os.path.realpath('__file__') )[0], '../unlock.py')) as f:
            print(f.read())
        event.clear()


if __name__ == "__main__":
    # 多线程调用
    run = threading.Thread(target=run)
    run.start()
    lock_wait = threading.Thread(target=lock_wait)
    lock_wait.start()
