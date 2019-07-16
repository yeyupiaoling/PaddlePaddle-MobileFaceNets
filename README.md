# 人脸识别 MobileFaceNets

## 获取数据
1. 数据集下载地址：https://pan.baidu.com/s/1eSNpdRG#list/path=%2F
2. 把`img_celeba.7z`解压到`data`目录下
3. 把`identity_CelebA.txt`文件复制到`data`目录下
4. 把`list_bbox_celeba.txt`文件复制到`data`目录下
5. 把`list_landmarks_celeba.txt`文件复制到`data`目录下


## 训练
1. 执行`python3 train/create_data_list.py` 创建训练的数据列表
2. 执行`python3 train/train.py` 执行训练


## 预测
1. 执行`python3 infer.py` 识别这里两张人脸的相似度。
2. 如果要其他的人脸图片，需要同时提供人脸的五个关键点，可以使用 https://github.com/yeyupiaoling/PaddlePaddle-MTCNN 模型识别