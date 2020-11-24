import cv2
import os
import time
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
import utils.time_utils as time_utils
from net.inception import InceptionResNetV1
import threading


# 利用Event类模拟信号量
event = threading.Event()


def lock_wait():
    while True:
        event.wait()
        print("\33[42;1mdetected face! unlocking...\033[0m")
        with os.popen('python ' + os.path.join(os.path.split(os.path.realpath('__file__'))[0], 'unlock.py')) as f:
            print(f.read())
        time.sleep(20)  # 20s 内不再次开锁
        event.clear()


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

            img = cv2.imread("./face_dataset/"+face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # 转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)

            new_img = np.expand_dims(new_img,0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

        time3 = time.time()
        print('init faceDataSet use: ', end='')
        print("{:.2f}".format(time3-time2))



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
            if time_utils.is_night():   # 如果是晚上，将容忍度提高
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
                print(name+" ---> "+str(dis))

                event.set()  # 设置信号量，异步调用开锁函数

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
            print(name + str(dis))
        return draw


def run():
    diudiudiu = face_rec()
    video_capture = cv2.VideoCapture(0)

    num_frames = 0
    since = time.time()
    frame_compare = np.zeros((480, 640))
    while True:
        time.sleep(0.7)
        ret, draw = video_capture.read()
        # 降维
        gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        d = int(np.linalg.norm(frame_compare - gray[:np.shape(gray)[0]][:np.shape(gray)[1]]) / 10000 - 6)
        print(d)
        # 重置检测位
        frame_compare = gray[:np.shape(gray)[0]][:np.shape(gray)[1]]
        diudiudiu.recognize(draw)
        # put the text into video
        num_frames += 1
        cv2.putText(draw, f'FPS{num_frames / (time.time() - since):.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 2)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 多线程调用
    run = threading.Thread(target=run)
    run.start()
    lock_wait = threading.Thread(target=lock_wait)
    lock_wait.start()
