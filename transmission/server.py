from socket import *
import threading
import struct
import time
import cv2
import numpy


class Carame_Accept_Object:
    def __init__(self, S_addr_port=("", 9901)):
        self.resolution = (0, 0)  # 分辨率
        self.img_fps = 0  # 每秒传输多少帧数

        # 设置套接字
        self.server = socket(AF_INET, SOCK_STREAM)
        self.server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  # 端口可复用
        self.server.bind(S_addr_port)
        self.server.listen(5)
        print("the process work in the port:%d" % S_addr_port[1])


def check_option(object, client):
    # 按格式解码，依照客服的要求帧数和分辨率赋值
    info = struct.unpack('lhh', client.recv(8))
    if info[0] > 0:
        object.img_quality = int(info[0])  # 获取图片传输质量
        object.resolution = list(object.resolution)
        # 获取分辨率
        object.resolution[0] = info[1]
        object.resolution[1] = info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        print("check_option 格式解码错误")
        return 0


def RT_Image(object, client):
    if (check_option(object, client) == 0):
        return
    vc = cv2.VideoCapture(0)  # 从摄像头中获取视频
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), object.img_quality]  # 设置传送质量(0-100)
    while True:
        # time.sleep(0.1)  # 推迟线程运行0.1s
        _, img = vc.read()  # 读取视频每一帧
        # print(vc.get(cv2.CAP_PROP_FPS))
        # cv2.imshow("local test", img)

        img = cv2.resize(img, object.resolution)  # 按要求调整图像大小(resolution必须为元组)
        _, img_encode = cv2.imencode('.jpg', img, img_param)  # 按格式生成图片
        img_code = numpy.array(img_encode)  # 转换成矩阵
        img_data = img_code.tobytes()  # 生成相应的字符串(字节数据)
        try:
            # 流程中第二次发送信息，按照相应的格式进行打包发送图片
            client.send(
                struct.pack(
                    "l", len(img_data))
                     + img_data)
        except:
            vc.release()  # 释放资源
            return
        finally:
            if cv2.waitKey(10) == 27:  # 每10ms刷新一次图片，按‘ESC’（27）退出
                break


if __name__ == '__main__':
    camera = Carame_Accept_Object()
    while True:
        client, D_addr = camera.server.accept()
        clientThread = threading.Thread(None, target=RT_Image, args=(camera, client,))
        clientThread.start()
