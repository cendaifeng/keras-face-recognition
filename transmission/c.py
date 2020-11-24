from socket import *

HOST = '192.168.0.105'
PORT = 9902
BUFSIZ =1024
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET,SOCK_STREAM)
tcpCliSock.connect(ADDR)
while True:
     data1 = input('>')
     #data = str(data)
     if not data1:
         break
     tcpCliSock.send(data1.encode())
     data1 = tcpCliSock.recv(BUFSIZ)
     if not data1:
         break
     print(data1.decode('utf-8'))
tcpCliSock.close()