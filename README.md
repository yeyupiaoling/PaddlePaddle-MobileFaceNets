# MobileFaceNet

本项目参考了[ArcFace](https://arxiv.org/abs/1801.07698)的损失函数，同时参考了[PP-OCRv2](https://arxiv.org/abs/2109.03144)模型结构，意在开发一个模型较小，但识别准确率较高且推理速度快的一种人脸识别项目，该项目训练数据使用emore数据集，一共有85742个人，共5822653张图片，使用lfw-align-128数据集作为测试数据。

# 数据集准备
本项目提供了标注文件，存放在`dataset`目录下，解压即可。另外需要下载下面这两个数据集，下载完解压到`dataset`目录下。
 - emore数据集[百度网盘](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)
 - lfw-align-128下载地址：[百度网盘](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA) 提取码：b2ec

然后执行下面命令，将提取人脸图片到`dataset/images`，并把整个数据集打包为二进制文件，这样可以大幅度的提高训练时数据的读取速度。
```shell
python create_dataset.py
```

# 训练

执行`train.py`即可，更多训练参数请查看代码。
```shell
python train.py
```

# 评估

执行`eval.py`即可，更多训练参数请查看代码。
```shell
python eval.py
```

# 预测

本项目已经不教提供了模预测，模型文件可以直接用于预测。在执行预测之前，先要在face_db目录下存放人脸图片，每张图片只包含一个人脸，并以该人脸的名称命名，这建立一个人脸库。之后的识别都会跟这些图片对比，找出匹配成功的人脸。

如果是通过图片路径预测的，请执行下面命令。
```shell
python infer.py --image_path=temp/test.jpg
```

如果是通过相机预测的，请执行下面命令。
```shell
python infer_camera.py --camera_id=0
```