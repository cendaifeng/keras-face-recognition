## 基于 Keras 人脸识别算法实现的树莓派智能门锁应用

---

*** Fork 自 https://github.com/bubbliiiing/keras-face-recognition ***

### 目录

1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [使用方法 Usage](#使用方法)

### 所需环境
tensorflow-gpu==1.13.1
keras==2.1.5

### 文件下载
训练所需的facenet_keras.h5可以在Release里面下载。
也可以去百度网盘下载
链接: https://pan.baidu.com/s/1A9jCJa_sQ4D3ejelgXX2RQ 提取码: tkhg

### 使用方法
1、先将整个仓库download下来。
2、下载完之后解压，同时下载facenet_keras.h5文件。
3、将facenet_keras.h5放入model_data中。
4、将自己想要识别的人脸放入到face_dataset中。
5、运行face_recognize.py即可。
6、align.py可以查看人脸对齐的效果。