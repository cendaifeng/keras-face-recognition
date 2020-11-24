import RPi.GPIO as GPIO
import time
import atexit
import os
from socket import *

 
atexit.register(GPIO.cleanup)
servopin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servopin, GPIO.IN)

HOST = '192.168.0.106'
PORT = 9902
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET,SOCK_STREAM)
tcpCliSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
tcpCliSock.connect(ADDR)

while True:
    INPUT = GPIO.input(servopin)
    if INPUT is 0:
        # 发送 socket 信号
        print('==send==')
        data = 'open650'
        tcpCliSock.send(data.encode())
        time.sleep(1)
tcpCliSock.close()
