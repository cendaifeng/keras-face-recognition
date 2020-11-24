from socket import *
import cv2
import threading
import struct
import numpy


class Camera_Connect_Object:
    def __init__(self, D_addr_port=("192.168.0.105", 9901)):
        self.resolution = [640, 480]
        self.addr_port = D_addr_port
        self.fps_protocol = 30  # 传输帧数
        self.interval = 0  # 图片播放时间间隔
        self.img_quality = 100  # 图片传输质量

    def Socket_Connect(self):
        self.client = socket(AF_INET, SOCK_STREAM)
        self.client.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.client.connect(self.addr_port)  # 传入一个元组
        print("IP is %s:%d" % (self.addr_port[0], self.addr_port[1]))

    def RT_Image(self):
        # 按照格式打包发送帧数和分辨率
        self.name = self.addr_port[0] + " Camera"
        # 将元组按 lhh 的方式打包成 字节对象
        # 第一次发送信息确定客户端接收协议
        self.client.send(struct.pack("lhh", self.img_quality, self.resolution[0], self.resolution[1]))
        while True:
            # 这里只需要传输一个4字节的int
            info = struct.unpack("l", self.client.recv(4))
            buf_size = info[0]  # 获取读的图片总长度
            if buf_size:
                try:
                    self.buf = b""  # 代表bytes类型
                    temp_buf = self.buf
                    while (buf_size):  # 读取每一张图片的长度
                        # 接收剩余的信息
                        temp_buf = self.client.recv(buf_size)
                        buf_size -= len(temp_buf)
                        self.buf += temp_buf  # 获取图片
                        data = numpy.frombuffer(self.buf, dtype='uint8')  # 按uint8转换为图像矩阵
                        self.image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 图像解码
                        cv2.imshow(self.name, self.image)  # 展示图片
                except:
                    pass;
                finally:
                    if cv2.waitKey(10) == 27:  # 每10ms刷新一次图片，按‘ESC’（27）退出
                        self.client.close()
                        cv2.destroyAllWindows()
                        break

    def run(self):
        showThread = threading.Thread(target=self.RT_Image)
        showThread.start()


if __name__ == '__main__':
    camera = Camera_Connect_Object()
    camera.Socket_Connect()
    camera.run()
