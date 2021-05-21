# MobileFaceNet

# 数据集
本项目提供了标注文件，存放在`dataset`目录下，解压即可。另外的图片文件需要自行下载，下载解压到`dataset`目录下即可。
 - CASIA-WebFace下载地址：[百度网盘](https://pan.baidu.com/s/1OjyZRhZhl__tOvhLnXeapQ) 提取码：nf6i
 - lfw-align-128下载地址：[百度网盘](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA) 提取码：b2ec
 - 另外提供emore数据集[百度网盘](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ) 下载地址，但本项目不提供使用方式，读者自行使用。

执行下面命令，将CASIA-WebFace裁剪人脸并对齐。
```shell
python create_dataset.py
```

# 训练

执行`train.py`即可，更多训练参数请查看代码。
```shell
python train.py
```