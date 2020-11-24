import cv2
import os
import time
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
import utils.time_utils as time_utils
from net.inception import InceptionResNetV1
# import RPi.GPIO as GPIO
import atexit
import threading
from socket import *
import struct
import numpy


# # 初始化舵机
# atexit.register(GPIO.cleanup)
# servopin = 17
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(servopin, GPIO.OUT, initial=False)
# pwm = GPIO.PWM(servopin, 50)  # 50HZ
# pwm.start(0)
# # 利用Event类模拟信号量
# event = threading.Event()


def ControlMotor(angle):
    pass
    # # 舵机的频率为50HZ，占空比为2.5%-12.5%，线性对应舵机转动角度的0-180度
    # pwm.ChangeDutyCycle(2.5 + angle / 360 * 20)


def lock_wait():
    pass
    # while True:
    #     event.wait()
    #     print("\33[42;1mdetected face! unlocking...\033[0m")
    #
    #     ControlMotor(180)
    #     time.sleep(5)
    #     ControlMotor(0)
    #     time.sleep(10)  # 10s 内不再次开锁
    #
    #     event.clear()


class face_rec:
    def __init__(self):
        # 创建 mtcnn 对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5, 0.8, 0.9]

        # 载入 facenet
        # 将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        # model.summary()
        # 加载模型权重只需要15s 而加载图像文件夹则需30s以上
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        time2 = time.time()
        # ----------------------------------------------- #
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        # ----------------------------------------------- #
        face_list = os.listdir("face_dataset")

        self.known_face_encodings = []

        self.known_face_names = []

        for face in face_list:
            name = face.split(".")[0]

            img = cv2.imread("./face_dataset/" + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # 转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)

            new_img = np.expand_dims(new_img, 0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

        time3 = time.time()
        print('init faceDataSet use: ', end='')
        print("{:.2f}".format(time3 - time2))

    def recognize(self, draw):
        # -----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        # -----------------------------------------------#
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles) == 0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)
        # -----------------------------------------------#
        #   对检测到的人脸进行编码
        # -----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
            face_encodings.append(face_encoding)

        face_names = []
        distances = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            if time_utils.is_night():  # 如果是晚上，将容忍度提高
                tolerance = 0.8
            else:
                tolerance = 0.7
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            dis = face_distances[best_match_index]
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                print(name + " ---> " + str(dis))

                # event.set()  # 设置信号量，异步调用开锁函数

            face_names.append(name)
            distances.append(dis)

        rectangles = rectangles[:, 0:4]
        # -----------------------------------------------#
        #   画框~!~
        # -----------------------------------------------#
        for (left, top, right, bottom), name, dis in zip(rectangles, face_names, distances):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)
            cv2.putText(draw, str(dis), (right, bottom - 15), font, 0.75, (0, 0, 255), 2)
            if name == "Unknown":
                print("Unknown: " + str(dis))
        return draw


class Camera_Connect_Object:
    def __init__(self, D_addr_port=("192.168.0.105", 9901)):
        self.resolution = [640, 480]
        self.addr_port = D_addr_port
        self.fps_protocol = 30  # 传输帧数
        self.interval = 0  # 图片播放时间间隔
        self.img_quality = 100  # 图片传输质量
        self.image = None

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
                    pass
                finally:
                    if cv2.waitKey(10) & 0xFF == ord('q'):  # 每10ms刷新一次图片，按‘ESC’（27）退出
                        self.client.close()
                        cv2.destroyAllWindows()
                        break

    def run(self):
        showThread = threading.Thread(target=self.RT_Image)
        showThread.start()


def run():
    """ 图传通信部分 """
    camera = Camera_Connect_Object()
    camera.Socket_Connect()
    camera.run()

    """ 图像检测部分 """
    diudiudiu = face_rec()
    # video_capture = cv2.VideoCapture(0)
    num_frames = 0
    since = time.time()
    frame_compare = np.zeros((480, 640))

    while True:
        time.sleep(0.7)
        """
        ret, draw = video_capture.read()
        """
        # # 降维
        # gray = cv2.cvtColor(camera.image, cv2.COLOR_BGR2GRAY)
        # d = int(np.linalg.norm(frame_compare - gray[:np.shape(gray)[0]][:np.shape(gray)[1]]) / 10000 - 6)
        # print(d)
        # # 重置检测位
        # frame_compare = gray[:np.shape(gray)[0]][:np.shape(gray)[1]]

        draw = camera.image
        if not draw is None:
            diudiudiu.recognize(draw)
            # put the text into video
            num_frames += 1
            cv2.putText(draw, f'FPS{num_frames / (time.time() - since):.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 2)
            cv2.imshow('Video', draw)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 多线程调用
    run = threading.Thread(target=run)
    run.start()
    lock_wait = threading.Thread(target=lock_wait)
    lock_wait.start()
